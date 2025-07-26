import argparse
import glob
import os
import sys
from datetime import datetime
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Parameter, ParameterDict
from torch_sparse import SparseTensor
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected
from torch_geometric.utils.hetero import group_hetero_graph

from ogb.nodeproppred import Evaluator, PygNodePropPredDataset


class LoggerWriter:
    def __init__(self, file_obj, stream):
        self.file_obj = file_obj
        self.stream = stream
        self._buffer = ''

    def write(self, message):
        self._buffer += message
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            formatted_line = timestamp + line + '\n'
            self.file_obj.write(formatted_line)
            self.file_obj.flush()
            self.stream.write(formatted_line)
            self.stream.flush()

    def flush(self):
        if self._buffer:
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            formatted_line = timestamp + self._buffer + '\n'
            self.file_obj.write(formatted_line)
            self.file_obj.flush()
            self.stream.write(formatted_line)
            self.stream.flush()
            self._buffer = ''


def setup_logging(feature_type: str):
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_filename = f'{script_name}-mag-{feature_type}-{timestamp}.log'
    log_path = os.path.join('logs', log_filename)
    log_file = open(log_path, 'a', encoding='utf-8')
    sys.stdout = LoggerWriter(log_file, sys.__stdout__)
    sys.stderr = LoggerWriter(log_file, sys.__stderr__)


def load_bloom_features():
    bloom_files = glob.glob('Bloom/*.npy')
    if not bloom_files:
        raise FileNotFoundError("Bloom features .npy file not found in 'Bloom/' directory.")
    bloom_path = bloom_files[0]
    bloom = np.load(bloom_path)
    bloom = torch.from_numpy(bloom).float()
    bloom_dim = bloom.size(1)
    proj_dim = 116 if bloom_dim == 63 else bloom_dim
    bloom_proj = Linear(bloom_dim, proj_dim)
    bloom_feat = F.relu(bloom_proj(bloom))
    return bloom_feat, bloom_dim, proj_dim


def load_transe_features():
    transe_files = glob.glob('TransE/*.npy')
    if not transe_files:
        raise FileNotFoundError("TransE features .npy file not found in 'TransE/' directory.")
    transe_path = transe_files[0]
    transe = np.load(transe_path)
    transe = torch.from_numpy(transe).float()
    transe_dim = transe.size(1)
    transe_proj = Linear(transe_dim, 128)
    transe_feat = F.relu(transe_proj(transe))
    return transe_feat, transe_dim


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types):
        super().__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

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
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, target_node_type):
        x_src, x_target = x

        out = x_target.new_zeros(x_target.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = target_node_type == i
            out[mask] += self.root_lins[i](x_target[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx, n_id=None):
        # Create global node feature matrix.
        if n_id is not None:
            node_type = node_type[n_id]
            local_node_idx = local_node_idx[n_id]

        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, n_id, x_dict, adjs, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx, n_id)
        node_type = node_type[n_id]

        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target node embeddings.
            node_type = node_type[:size[1]]  # Target node types.
            conv = self.convs[i]
            x = conv((x, x_target), edge_index, edge_type[e_id], node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x.log_softmax(dim=-1)

    def inference(self, x_dict, edge_index_dict, key2int):
        # We can perform full-batch inference on GPU.

        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)
        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                out.add_(conv.rel_lins[key2int[keys]](tmp))

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict


def main():
    # ─── Argument Parsing ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='OGBN-MAG (SAGE)')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument(
        '--feature_type',
        type=str,
        default='word2vec+bloom+transe',
        choices=['word2vec', 'bloom', 'word2vec+bloom', 'word2vec+bloom+transe'],
        help='Feature type to use (case-insensitive)',
    )
    args = parser.parse_args()
    args.feature_type = args.feature_type.lower()

    # ─── Setup Logging ─────────────────────────────────────────────────────
    setup_logging(args.feature_type)
    print(args)

    # ─── Load Dataset ──────────────────────────────────────────────────────
    dataset = PygNodePropPredDataset(name='ogbn-mag')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-mag')

    # We do not consider those attributes for now.
    data.node_year_dict = None
    data.edge_reltype_dict = None

    # ─── Load and Fuse Features ────────────────────────────────────────────
    # Word2Vec features (128-d)
    fused_features = []
    w2v = None
    if 'word2vec' in args.feature_type:
        w2v = data.x_dict['paper']  # [num_papers × 128]
        w2v = (w2v - w2v.mean(dim=0)) / (w2v.std(dim=0) + 1e-6)
        fused_features.append(w2v)

    if 'bloom' in args.feature_type:
        bloom_feat, bloom_dim, bloom_proj_dim = load_bloom_features()
        fused_features.append(bloom_feat)

    if 'transe' in args.feature_type:
        transe_feat, transe_dim = load_transe_features()
        transe_feat = (transe_feat - transe_feat.mean(dim=0)) / (transe_feat.std(dim=0) + 1e-6)
        fused_features.append(transe_feat)

    if not fused_features:
        raise ValueError("No features selected. Please specify a valid --feature_type.")

    fused = torch.cat(fused_features, dim=1)  # Concatenate features

    data.x_dict.clear()
    data.x_dict['paper'] = fused

    print(data)

    # ─── Graph Setup ───────────────────────────────────────────────────────
    edge_index_dict = data.edge_index_dict

    # Add reverse edges to the heterogeneous graph.
    r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
    edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])

    r, c = edge_index_dict[('author', 'writes', 'paper')]
    edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])

    r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
    edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])

    # Convert to undirected paper <-> paper relation.
    edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
    edge_index_dict[('paper', 'cites', 'paper')] = edge_index

    # Convert individual graphs into a single big one.
    out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
    edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

    # Map informations to their canonical type.
    x_dict = {}
    for key, x in data.x_dict.items():
        x_dict[key2int[key]] = x

    num_nodes_dict = {}
    for key, N in data.num_nodes_dict.items():
        num_nodes_dict[key2int[key]] = N

    # Create train sampler iterating over paper training nodes.
    paper_idx = local2global['paper']
    paper_train_idx = paper_idx[split_idx['train']['paper']]

    train_loader = NeighborSampler(
        edge_index,
        node_idx=paper_train_idx,
        sizes=[25, 20],
        batch_size=1024,
        shuffle=True,
        num_workers=12,
    )

    # ─── Model Setup ───────────────────────────────────────────────────────
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    model = RGCN(
        fused.size(1),
        args.hidden_channels,
        dataset.num_classes,
        args.num_layers,
        args.dropout,
        num_nodes_dict,
        list(x_dict.keys()),
        len(edge_index_dict.keys()),
    ).to(device)

    # Create global label vector.
    y_global = node_type.new_full((node_type.size(0), 1), -1)
    y_global[local2global['paper']] = data.y_dict['paper']

    # Move everything to the device.
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_type = edge_type.to(device)
    node_type = node_type.to(device)
    local_node_idx = local_node_idx.to(device)
    y_global = y_global.to(device)

    # ─── Training and Testing Functions ────────────────────────────────────
    def train(epoch):
        model.train()
        total_loss = 0
        for batch_size, n_id, adjs in train_loader:
            n_id = n_id.to(device)
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
            y = y_global[n_id][:batch_size].squeeze()
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
        loss = total_loss / paper_train_idx.size(0)
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}")
        return loss

    @torch.no_grad()
    def test():
        model.eval()
        out = model.inference(x_dict, edge_index_dict, key2int)
        out = out[key2int['paper']]

        y_pred = out.argmax(dim=-1, keepdim=True).cpu()
        y_true = data.y_dict['paper']

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

        print(
            f"Train: {100 * train_acc:.2f}%, "
            f"Valid: {100 * valid_acc:.2f}%, "
            f"Test: {100 * test_acc:.2f}%"
        )
        return train_acc, valid_acc, test_acc

    # ─── Training Loop ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    test()  # Test if inference on GPU succeeds.

    for epoch in range(1, 1 + args.epochs):
        train(epoch)
        test()


if __name__ == "__main__":
    main()