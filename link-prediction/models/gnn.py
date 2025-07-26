import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, RGCNConv

'''
class GNN(torch.nn.Module):
    """
    Two-layer GraphSAGE module for message passing.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        #assert out_dim % 2 == 0, "out_dim must be even for RotatE decoder"

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x'''


class GNN(torch.nn.Module):
    """
    Graph module: defaults to GraphSAGE, or can use relation-aware R-GCN.

    conv_type: 'sage' for GraphSAGE (default), 'rgcn' for R-GCN.
    If conv_type='rgcn', supply num_relations and optionally num_bases.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        conv_type: str = 'sage',  # default = GraphSAGE
        num_relations: int = None,
        num_bases: int = None,
    ):
        super().__init__()
        self.conv_type = conv_type.lower()

        if self.conv_type == 'sage':
            # GraphSAGE layers
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, out_dim)

        elif self.conv_type == 'rgcn':
            # Relation-aware R-GCN layers
            assert num_relations is not None, "num_relations required for RGCN"
            self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations, num_bases=num_bases)
            self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations, num_bases=num_bases)

        else:
            raise ValueError(f"Unknown conv_type: {conv_type}. Use 'sage' or 'rgcn'.")

        # gnn out_dim must be even for RotatE decoder compatibility
        assert out_dim % 2 == 0, "out_dim must be even for RotatE decoder"

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.conv_type == 'sage':
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)

        else:  # 'rgcn'
            assert edge_type is not None, "edge_type required for RGCNConv"
            x = self.conv1(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_type)

        return x

