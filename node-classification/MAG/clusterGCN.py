#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clusterGCN.py — Node-Level-Tasks 
"""

import os
import sys
import glob
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, ClusterData, ClusterLoader
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# ---------------------------------------------------------------------------- #
#  Logging utilities                                                          #
# ---------------------------------------------------------------------------- #
class LoggerWriter:
    def __init__(self, stream, logger_name):
        self.stream = stream
        self.logger = logging.getLogger(logger_name)
    def write(self, message):
        message = message.rstrip()
        if message:
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            self.logger.info(f"{timestamp} {message}")
            self.stream.write(message + "\n")
    def flush(self):
        self.stream.flush()

# ---------------------------------------------------------------------------- #
#  Model definitions                                              #
# ---------------------------------------------------------------------------- #
class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types, num_edge_types):
        super().__init__(aggr='mean')
        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])
        self.root_lins = ModuleList([
            Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])
        self.reset_parameters()
    def reset_parameters(self):
        for lin in (*self.rel_lins, *self.root_lins):
            lin.reset_parameters()
    def forward(self, x, edge_index, edge_type, node_type):
        out = x.new_zeros(x.size(0), self.rel_lins[0].out_features)
        for i, lin in enumerate(self.rel_lins):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))
        for i, lin in enumerate(self.root_lins):
            mask = node_type == i
            out[mask] += lin(x[mask])
        return out
    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)

class RGCN(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super().__init__()
        self.in_channels = in_ch
        self.convs = ModuleList()
        num_node_types = len(num_nodes_dict)
        # embeddings for types without features
        self.emb_dict = ParameterDict({
            str(k): Parameter(torch.Tensor(n, in_ch))
            for k, n in num_nodes_dict.items() if k not in x_types
        })
        # layers
        self.convs.append(RGCNConv(in_ch, hid_ch, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hid_ch, hid_ch, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(hid_ch, out_ch, num_node_types, num_edge_types))
        self.dropout = dropout
        self.reset_parameters()
    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()
    def group_input(self, x_dict, node_type, local_node_idx):
        h = torch.zeros(node_type.size(0), self.in_channels,
                        device=node_type.device)
        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]
        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]
        return h
    def forward(self, x_dict, edge_index, edge_type, node_type, local_node_idx):
        x = self.group_input(x_dict, node_type, local_node_idx)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, node_type)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)
    @torch.no_grad()
    def inference(self, x_dict, edge_index_dict, key2int):
        device = next(iter(x_dict.values())).device
        x_dict = {**x_dict, **{int(k): emb for k, emb in self.emb_dict.items()}}
        # build SparseTensor dict
        adj_t = {}
        for k, (r, c) in edge_index_dict.items():
            adj_t[k] = SparseTensor(row=c, col=r).to(device)
        for i, conv in enumerate(self.convs):
            out_dict = {}
            # root
            for j in range(len(self.emb_dict) + len(x_dict)):
                out_dict[j] = conv.root_lins[j](x_dict[j])
            # message
            for k, mat in adj_t.items():
                src, tgt = k[0], k[-1]
                tmp = mat.matmul(x_dict[key2int[src]], reduce='mean')
                out_dict[key2int[tgt]].add_(conv.rel_lins[key2int[k]](tmp))
            # activation
            if i != len(self.convs) - 1:
                for j in out_dict:
                    F.relu_(out_dict[j])
            x_dict = out_dict
        return x_dict

# ---------------------------------------------------------------------------- #
#  Training & evaluation routines (only minor tqdm tweaks)                      #
# ---------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, x_dict, device):
    model.train()
    total_loss = total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(x_dict, batch.edge_index, batch.edge_attr,
                    batch.node_type, batch.local_node_idx)
        mask = batch.train_mask
        loss = F.nll_loss(out[mask], batch.y[mask].squeeze())
        loss.backward()
        optimizer.step()
        n = int(mask.sum())
        total_loss += float(loss) * n
        total_examples += n
    return total_loss / total_examples

@torch.no_grad()
def evaluate(model, data, split_idx, evaluator, x_dict,
             edge_index_dict, key2int, device):
    model.eval()
    out_dict = model.inference(x_dict, edge_index_dict, key2int)
    out = out_dict[key2int['paper']].cpu()
    y_true = data.y_dict['paper']
    results = {}
    for split in ['train','valid','test']:
        mask = split_idx[split]['paper']
        results[split] = evaluator.eval({
            'y_true': y_true[mask],
            'y_pred': out[mask].argmax(dim=-1, keepdim=True)
        })['acc']
    return results

# ---------------------------------------------------------------------------- #
#  Main execution                                                               #
# ---------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='OGBN-MAG (Cluster-GCN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument(
        '--feature_type', type=str, default='word2vec+bloom+transe',
        choices=['word2vec','bloom','word2vec+bloom','word2vec+bloom+transe'],
    )
    args = parser.parse_args()

    # set up logging to file & console
    os.makedirs('logs', exist_ok=True)
    script = os.path.splitext(os.path.basename(__file__))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"logs/{script}-mag-{args.feature_type}-{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(fname),
            logging.StreamHandler(sys.stdout)
        ])
    sys.stdout = LoggerWriter(sys.stdout, script)
    sys.stderr = LoggerWriter(sys.stderr, script)

    print(f"Running with feature_type={args.feature_type}, epochs={args.epochs}")

    # data & dataset
    dataset = PygNodePropPredDataset(name='ogbn-mag')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-mag')

    # ------------------------------------------------------------------------ #
    # feature loading                                                          #
    # ------------------------------------------------------------------------ #
    # Word2Vec
    w2v = data.x_dict['paper']  # [N × 128]
    w2v = (w2v - w2v.mean(0)) / (w2v.std(0) + 1e-6)

    # Bloom (dynamic)
    bloom_files = glob.glob('Bloom/paper_bloom_filters*.npy')
    if not bloom_files:
        raise FileNotFoundError("No Bloom .npy files found in Bloom/")
    bloom = torch.from_numpy(np.load(bloom_files[0])).float()
    bloom_dim = bloom.size(1)
    bloom_proj = Linear(bloom_dim, 116 if bloom_dim == 63 else bloom_dim)
    bloom_feat = F.relu(bloom_proj(bloom)).detach()

    # TransE (dynamic)
    transe_files = glob.glob('TransE/paper_transe_embeddings*.npy')
    if not transe_files:
        raise FileNotFoundError("No TransE .npy files found in TransE/")
    transe = torch.from_numpy(np.load(transe_files[0])).float()
    transe_dim = transe.size(1)
    transe_proj = Linear(transe_dim, transe_dim)
    transe_feat = F.relu(transe_proj(transe)).detach()

    # fuse/select features
    ft = args.feature_type.lower()
    if ft == 'word2vec':
        fused = w2v
    elif ft == 'bloom':
        fused = bloom_feat
    elif ft == 'transe':
        fused = transe_feat
    elif ft == 'word2vec+bloom':
        fused = torch.cat([w2v, bloom_feat], dim=1).detach()
    else:
        fused = torch.cat([w2v, bloom_feat, transe_feat], dim=1).detach()
    data.x_dict['paper'] = fused

    print(data)

    # graph preprocess 
    edge_index_dict = data.edge_index_dict
    # add reverse and undirected edges...
    # (same as original clusterGCN.py)...
    # [omitted here for brevity — copy your original reverse-edge logic]

    # group hetero graph
    out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
    edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
    homo_data = Data(
        edge_index=edge_index, edge_attr=edge_type,
        node_type=node_type, local_node_idx=local_node_idx,
        num_nodes=node_type.size(0)
    )
    homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
    homo_data.y[local2global['paper']] = data.y_dict['paper']
    homo_data.train_mask = torch.zeros(node_type.size(0), dtype=torch.bool)
    homo_data.train_mask[local2global['paper'][split_idx['train']['paper']]] = True

    cluster_data = ClusterData(
        homo_data, num_parts=5000, recursive=True,
        save_dir=dataset.processed_dir
    )
    train_loader = ClusterLoader(
        cluster_data, batch_size=500, shuffle=True, num_workers=12
    )

    x_dict = {k: v for k, v in data.x_dict.items()}
    num_nodes_dict = {
        key2int[k]: v for k, v in data.num_nodes_dict.items()
    }

    # device, model, optimizer
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    model = RGCN(
        fused.size(1), args.hidden_channels, dataset.num_classes,
        args.num_layers, args.dropout,
        num_nodes_dict, list(x_dict.keys()),
        len(edge_index_dict)
    ).to(device)
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # single-run training
    logger = Evaluator(name='dummy')  # placeholder; keep your original Logger if needed
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, x_dict, device)
        results = evaluate(
            model, data, split_idx, evaluator,
            x_dict, edge_index_dict, key2int, device
        )
        if epoch == 1 or epoch % args.log_steps == 0 or epoch == args.epochs:
            train_acc = results['train'] * 100
            valid_acc = results['valid'] * 100
            test_acc  = results['test']  * 100
            print(f"Epoch {epoch:03d} | "
                  f"Loss: {loss:.4f} | "
                  f"Train: {train_acc:.2f}% | "
                  f"Valid: {valid_acc:.2f}% | "
                  f"Test: {test_acc:.2f}%")

    # final stats (if you use Logger, call logger.print_statistics())

if __name__ == "__main__":
    main()