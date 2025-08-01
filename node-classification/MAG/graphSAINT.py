from copy import copy
import argparse
import os
import sys
import glob
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data.data import DataEdgeAttr
torch.serialization.add_safe_globals([DataEdgeAttr])
import functools
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, GraphSAINTRandomWalkSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class LoggerWriter:
    def __init__(self, stream):
        self.stream = stream

    def write(self, message):
        if message.strip() == '':
            return
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for line in message.rstrip().splitlines():
            self.stream.write(f'[{now}] {line}\n')
        self.flush()

    def flush(self):
        self.stream.flush()


def load_bloom_feature():
    bloom_files = glob.glob('Bloom/paper_bloom_filters*.npy')
    if not bloom_files:
        raise FileNotFoundError("Bloom feature file not found in 'Bloom/' directory.")
    bloom = np.load(bloom_files[0])
    bloom = torch.from_numpy(bloom).float()
    bloom_dim = bloom.size(1)
    proj_dim = 116 if bloom_dim == 63 else bloom_dim
    bloom_proj = nn.Linear(bloom_dim, proj_dim)
    bloom_feat = F.relu(bloom_proj(bloom)).detach()
    return bloom_feat, proj_dim


def load_transe_feature():
    transe_files = glob.glob('TransE/paper_transe_embeddings*.npy')
    if not transe_files:
        raise FileNotFoundError("TransE feature file not found in 'TransE/' directory.")
    transe = np.load(transe_files[0])
    transe = torch.from_numpy(transe).float()
    transe_dim = transe.size(1)
    transe = (transe - transe.mean(dim=0)) / (transe.std(dim=0) + 1e-6)
    transe_proj = nn.Linear(transe_dim, 128)
    transe_feat = F.relu(transe_proj(transe)).detach()
    return transe_feat


def main():
    # ----------------- Argument Parsing -----------------
    parser = argparse.ArgumentParser(description='OGBN-MAG (GraphSAINT)')
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument(
        '--feature_type',
        type=str,
        default='word2vec+bloom+transe',
        choices=['word2vec', 'bloom', 'word2vec+bloom', 'word2vec+bloom+transe'],
        help='Feature type to use (case-insensitive).',
    )
    args = parser.parse_args()
    args.feature_type = args.feature_type.lower()

    # ----------------- Logging Setup -----------------
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_filename = f'{script_name}-mag-{args.feature_type}-{timestamp}.log'
    log_path = os.path.join('logs', log_filename)
    log_file = open(log_path, 'w')
    sys.stdout = LoggerWriter(log_file)
    sys.stderr = LoggerWriter(log_file)

    print(args)

    # ----------------- Dataset and Evaluator -----------------
    dataset = PygNodePropPredDataset(name='ogbn-mag')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-mag')
    logger = Logger(1, args)

    # We do not consider those attributes for now.
    data.node_year_dict = None
    data.edge_reltype_dict = None

    # ----------------- Feature Loading and Fusion -----------------
    fused_features = []
    fused_dim = 0

    if 'word2vec' in args.feature_type:
        w2v = data.x_dict['paper']  # [num_papers × 128]
        w2v = (w2v - w2v.mean(dim=0)) / (w2v.std(dim=0) + 1e-6)
        fused_features.append(w2v)
        fused_dim += w2v.size(1)

    if 'bloom' in args.feature_type:
        bloom_feat, bloom_proj_dim = load_bloom_feature()
        fused_features.append(bloom_feat)
        fused_dim += bloom_proj_dim

    if 'transe' in args.feature_type:
        transe_feat = load_transe_feature()
        fused_features.append(transe_feat)
        fused_dim += transe_feat.size(1)

    fused = torch.cat(fused_features, dim=1).detach()
    data.x_dict.clear()
    data.x_dict['paper'] = fused

    print(data)

    # ----------------- Graph Preprocessing -----------------
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

    # Convert the individual graphs into a single big one.
    out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
    edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

    homo_data = Data(
        edge_index=edge_index,
        edge_attr=edge_type,
        node_type=node_type,
        local_node_idx=local_node_idx,
        num_nodes=node_type.size(0),
    )

    homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
    homo_data.y[local2global['paper']] = data.y_dict['paper']

    homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
    homo_data.train_mask[local2global['paper'][split_idx['train']['paper']]] = True

    print(homo_data)

    train_loader = GraphSAINTRandomWalkSampler(
        homo_data,
        batch_size=args.batch_size,
        walk_length=args.num_layers,
        num_steps=args.num_steps,
        sample_coverage=0,
        save_dir=dataset.processed_dir,
        disable=True,
    )

    # Map informations to their canonical type.
    x_dict = {}
    for key, x in data.x_dict.items():
        x_dict[key2int[key]] = x

    num_nodes_dict = {}
    for key, N in data.num_nodes_dict.items():
        num_nodes_dict[key2int[key]] = N

    # ----------------- Model Definitions -----------------
    class RGCNConv(MessagePassing):
        def __init__(self, in_channels, out_channels, num_node_types, num_edge_types):
            super(RGCNConv, self).__init__(aggr='mean')

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.num_node_types = num_node_types
            self.num_edge_types = num_edge_types

            self.rel_lins = ModuleList(
                [Linear(in_channels, out_channels, bias=False) for _ in range(num_edge_types)]
            )

            self.root_lins = ModuleList(
                [Linear(in_channels, out_channels, bias=True) for _ in range(num_node_types)]
            )

            self.reset_parameters()

        def reset_parameters(self):
            for lin in self.rel_lins:
                lin.reset_parameters()
            for lin in self.root_lins:
                lin.reset_parameters()

        def forward(self, x, edge_index, edge_type, node_type):
            out = x.new_zeros(x.size(0), self.out_channels)

            for i in range(self.num_edge_types):
                mask = edge_type == i
                out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

            for i in range(self.num_node_types):
                mask = node_type == i
                out[mask] += self.root_lins[i](x[mask])

            return out

        def message(self, x_j, edge_type: int):
            return self.rel_lins[edge_type](x_j)


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
            num_edge_types,
        ):
            super(RGCN, self).__init__()

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
            self.emb_dict = ParameterDict(
                {
                    f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
                    for key in set(node_types).difference(set(x_types))
                }
            )

            I, H, O = in_channels, hidden_channels, out_channels  # noqa

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

        def group_input(self, x_dict, node_type, local_node_idx):
            # Create global node feature matrix.
            h = torch.zeros((node_type.size(0), self.in_channels), device=node_type.device)

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
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

            return x.log_softmax(dim=-1)

        def inference(self, x_dict, edge_index_dict, key2int):
            x_dict = copy(x_dict)
            for key, emb in self.emb_dict.items():
                x_dict[int(key)] = emb

            # Save original device and prepare CPU inference
            orig_device = next(self.parameters()).device

            adj_t_dict = {}
            for key, (row, col) in edge_index_dict.items():
                # Build sparse adjacency on CPU
                adj_t_dict[key] = SparseTensor(row=col, col=row)
            # Move model and data to CPU for sparse operations
            self.to('cpu')
            for k in x_dict:
                x_dict[k] = x_dict[k].cpu()

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

            # Restore model to original device
            self.to(orig_device)

            return x_dict

    # ----------------- Device Setup -----------------
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

    x_dict = {k: v.to(device) for k, v in x_dict.items()}

    # ----------------- Training and Testing -----------------
    def train(epoch):
        model.train()

        total_loss = total_examples = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(
                x_dict, data.edge_index, data.edge_attr, data.node_type, data.local_node_idx
            )
            out = out[data.train_mask]
            y = data.y[data.train_mask].squeeze()
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()

            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        avg_loss = total_loss / total_examples if total_examples > 0 else 0
        print(f"Epoch {epoch:02d}: Loss {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def test():
        model.eval()

        out = model.inference(x_dict, edge_index_dict, key2int)
        out = out[key2int['paper']]

        y_pred = out.argmax(dim=-1, keepdim=True).cpu()
        y_true = data.y_dict['paper']

        train_acc = evaluator.eval(
            {'y_true': y_true[split_idx['train']['paper']], 'y_pred': y_pred[split_idx['train']['paper']]}
        )['acc']
        valid_acc = evaluator.eval(
            {'y_true': y_true[split_idx['valid']['paper']], 'y_pred': y_pred[split_idx['valid']['paper']]}
        )['acc']
        test_acc = evaluator.eval(
            {'y_true': y_true[split_idx['test']['paper']], 'y_pred': y_pred[split_idx['test']['paper']]}
        )['acc']

        return train_acc, valid_acc, test_acc

    # ----------------- Main Training Loop -----------------
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Test inference on GPU before training
    test()

    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        torch.cuda.empty_cache()
        train_acc, valid_acc, test_acc = test()
        logger.add_result(0, (train_acc, valid_acc, test_acc))
        print(
            f"Epoch {epoch:02d} summary: "
            f"Train: {100 * train_acc:.2f}%, "
            f"Valid: {100 * valid_acc:.2f}%, "
            f"Test: {100 * test_acc:.2f}%"
        )

    logger.print_statistics(0)


if __name__ == "__main__":
    main()