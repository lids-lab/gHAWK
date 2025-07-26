"""RGCN.py: OGBN-MAG dataset with different feature .
"""

# =========================
# Imports and Setup Section
# =========================

import argparse
import copy
import glob
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Parameter,
    ModuleDict,
    ModuleList,
    Linear,
    ParameterDict,
)
from torch_sparse import SparseTensor

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger


# =========================
# RGCNConv, RGCN, train, test (Unchanged)
# =========================


class RGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, node_types, edge_types):
        super(RGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # `ModuleDict` does not allow tuples :(
        self.rel_lins = ModuleDict({
            f'{key[0]}_{key[1]}_{key[2]}': Linear(in_channels, out_channels, bias=False)
            for key in edge_types
        })

        self.root_lins = ModuleDict({
            key: Linear(in_channels, out_channels, bias=True)
            for key in node_types
        })

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins.values():
            lin.reset_parameters()
        for lin in self.root_lins.values():
            lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        out_dict = {}
        for key, x in x_dict.items():
            out_dict[key] = self.root_lins[key](x)

        for key, adj_t in adj_t_dict.items():
            key_str = f'{key[0]}_{key[1]}_{key[2]}'
            x = x_dict[key[0]]
            out = self.rel_lins[key_str](adj_t.matmul(x, reduce='mean'))
            out_dict[key[2]].add_(out)

        return out_dict


class RGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        num_nodes_dict,
        x_types,
        edge_types,
    ):
        super(RGCN, self).__init__()

        node_types = list(num_nodes_dict.keys())

        self.embs = ParameterDict({
            key: Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        self.convs = ModuleList()
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, node_types, edge_types)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, node_types, edge_types)
            )
        self.convs.append(
            RGCNConv(hidden_channels, out_channels, node_types, edge_types)
        )

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embs.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        x_dict = copy.copy(x_dict)
        for key, emb in self.embs.items():
            x_dict[key] = emb
        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, adj_t_dict)
            for key, x in x_dict.items():
                x_dict[key] = F.relu(x)
                x_dict[key] = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x_dict, adj_t_dict)


def train(model, x_dict, adj_t_dict, y_true, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(x_dict, adj_t_dict)['paper'].log_softmax(dim=-1)
    loss = F.nll_loss(out[train_idx], y_true[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, x_dict, adj_t_dict, y_true, split_idx, evaluator):
    model.eval()
    out = model(x_dict, adj_t_dict)['paper']
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']
    return train_acc, valid_acc, test_acc


# =========================
# LoggerWriter for Timestamped Output
# =========================

class LoggerWriter:
    """Redirects stdout/stderr, prefixing each line with a timestamp."""
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile
        self._buffer = ""

    def write(self, message):
        # Buffer lines to prefix each with timestamp
        self._buffer += message
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            if line.strip() == "" and not self.terminal.isatty():
                continue
            ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            msg = f"{ts} {line}\n"
            self.terminal.write(msg)
            self.logfile.write(msg)
            self.terminal.flush()
            self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


# =========================
# Main Function
# =========================

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="OGBN-MAG (Full-Batch) with Feature Type Selection")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument(
        '--feature_type',
        type=str,
        default='fused',
        help="Feature type: one of ['w2v', 'bloom', 'transe', 'fused']"
    )
    args = parser.parse_args()

    # Validate feature_type argument (case-insensitive)
    allowed_types = ['w2v', 'bloom', 'transe', 'fused']
    feature_type = args.feature_type.lower()
    if feature_type not in allowed_types:
        raise ValueError(f"Invalid --feature_type: {args.feature_type}. Allowed: {allowed_types}")

    # =========================
    # Logging Setup
    # =========================
    os.makedirs('logs', exist_ok=True)
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_fname = os.path.join('logs', f"{script_name}-mag-{feature_type}-{ts}.log")
    _log_file = open(log_fname, "a")
    sys.stdout = LoggerWriter(sys.stdout, _log_file)
    sys.stderr = LoggerWriter(sys.stderr, _log_file)
    print(args)
    print(f"Logging to {log_fname}")

    # =========================
    # Device and Data Loading
    # =========================
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dataset = PygNodePropPredDataset(name='ogbn-mag')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    # =========================
    # Feature Loading Section
    # =========================
    w2v = data.x_dict['paper']
    w2v = (w2v - w2v.mean(dim=0)) / (w2v.std(dim=0) + 1e-6)
    w2v = w2v.to(device)

    # Locate bloom and transe .npy files using glob, raise if not found
    bloom_paths = glob.glob("Bloom/paper_bloom_filters*.npy")
    transe_paths = glob.glob("TransE/paper_transe_embeddings*.npy")
    if not bloom_paths:
        raise FileNotFoundError("No bloom filter .npy file found in Bloom/ (pattern: paper_bloom_filters*.npy)")
    if not transe_paths:
        raise FileNotFoundError("No transe embedding .npy file found in TransE/ (pattern: paper_transe_embeddings*.npy)")
    bloom_path = bloom_paths[0]
    transe_path = transe_paths[0]
    bloom = np.load(bloom_path)
    bloom = torch.from_numpy(bloom).float()
    transe = np.load(transe_path)
    transe = torch.from_numpy(transe).float()
    transe = (transe - transe.mean(dim=0)) / (transe.std(dim=0) + 1e-6)
    transe = transe.to(device)

    # Auto-detect bloom and transe dimensions
    bloom_dim = bloom.shape[1]
    transe_dim = transe.shape[1]

    # Project bloom if necessary
    # If bloom_dim == 63, project to 116; else, identity projection
    bloom_proj_dim = 116 if bloom_dim == 63 else bloom_dim
    bloom_proj_layer = nn.Linear(bloom_dim, bloom_proj_dim).to(device)
    bloom_feat = F.relu(bloom_proj_layer(bloom.to(device))).detach()

    # =========================
    # Feature Selection
    # =========================
    # Compose features as per feature_type
    if feature_type == 'w2v':
        paper_feat = w2v
    elif feature_type == 'bloom':
        paper_feat = bloom_feat
    elif feature_type == 'transe':
        paper_feat = transe
    elif feature_type == 'fused':
        # Concatenate all three (w2v, transe, projected bloom)
        paper_feat = torch.cat([w2v, transe, bloom_feat], dim=1)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    # Replace data.x_dict['paper'] with the selected feature
    data.x_dict.clear()
    data.x_dict['paper'] = paper_feat

    print(f"Feature type: {feature_type}, paper feature shape: {paper_feat.shape}")

    # =========================
    # Graph Construction Section
    # =========================
    # Convert to new transposed `SparseTensor` format and add reverse edges.
    data.adj_t_dict = {}
    for keys, (row, col) in data.edge_index_dict.items():
        sizes = (data.num_nodes_dict[keys[0]], data.num_nodes_dict[keys[2]])
        adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
        if keys[0] != keys[2]:
            data.adj_t_dict[keys] = adj.t()
            data.adj_t_dict[(keys[2], 'to', keys[0])] = adj
        else:
            data.adj_t_dict[keys] = adj.to_symmetric()
    data.edge_index_dict = None

    x_types = list(data.x_dict.keys())
    edge_types = list(data.adj_t_dict.keys())

    # =========================
    # Model, Optimizer, Evaluator Setup
    # =========================
    model = RGCN(
        paper_feat.size(1),
        args.hidden_channels,
        dataset.num_classes,
        args.num_layers,
        args.dropout,
        data.num_nodes_dict,
        x_types,
        edge_types,
    )

    data = data.to(device)
    model = model.to(device)
    train_idx = split_idx['train']['paper'].to(device)
    evaluator = Evaluator(name='ogbn-mag')
    logger = Logger(1, args)

    # =========================
    # Training Loop (Single Run)
    # =========================
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        loss = train(
            model,
            data.x_dict,
            data.adj_t_dict,
            data.y_dict['paper'],
            train_idx,
            optimizer,
        )
        result = test(
            model,
            data.x_dict,
            data.adj_t_dict,
            data.y_dict['paper'],
            split_idx,
            evaluator,
        )
        logger.add_result(0, result)
        if epoch % args.log_steps == 0 or epoch == 1 or epoch == args.epochs:
            train_acc, valid_acc, test_acc = result
            print(
                f"Epoch: {epoch:03d}, "
                f"Loss: {loss:.4f}, "
                f"Train: {100 * train_acc:.2f}%, "
                f"Valid: {100 * valid_acc:.2f}%, "
                f"Test: {100 * test_acc:.2f}%"
            )
    logger.print_statistics(0)
    logger.print_statistics()


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()