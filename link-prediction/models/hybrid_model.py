import torch
from torch import nn

from .encoder import HybridNodeEncoder
from .decoder import RotatEDecoder
from .gnn import GNN
# at the top of models/hybrid_model.py
'''from models.encoder import HybridNodeEncoder
from models.decoder import RotatEDecoder
from models.gnn     import GNN'''



def build_global_edge_index(heads: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
    """
    Build an undirected global edge index tensor from train triples.
    """
    heads_tensor = torch.tensor(heads) if not torch.is_tensor(heads) else heads
    tails_tensor = torch.tensor(tails) if not torch.is_tensor(tails) else tails
    edge_index = torch.stack([heads_tensor, tails_tensor], dim=0)
    flipped = edge_index.flip(0)
    return torch.cat([edge_index, flipped], dim=1)


class HybridLinkPredictorWithGNN_PyG(nn.Module):
    """
    End-to-end link predictor:
      1. Encode nodes with Bloom+TransE via HybridNodeEncoder
      2. Extract subgraph over batch nodes
      3. Run a two-layer GNN (GraphSAGE or R-GCN) for message passing
      4. Score links with RotatEDecoder
    """
    def __init__(
        self,
        bloom_emb: torch.Tensor,
        transE_emb: torch.Tensor,
        num_relations: int,
        global_edge_index: torch.Tensor,
        bloom_proj_dim: int = 256,
        transE_proj_dim: int = 256,
        gnn_hidden_dim: int = 256,
        gnn_out_dim: int = 256,
        conv_type: str = 'sage',
        num_bases: int = None,
        pre_trained_rel_emb: torch.Tensor = None,
    ):
        super().__init__()
        # Node encoder for Bloom + TransE features
        self.hybrid_encoder = HybridNodeEncoder(
            bloom_emb,
            transE_emb,
            bloom_project_dim=bloom_proj_dim,
            transE_proj_dim=transE_proj_dim,
        )
        in_dim = bloom_proj_dim + transE_proj_dim

        # Two-layer GNN: GraphSAGE or R-GCN
        self.gnn = GNN(
            in_dim=in_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            conv_type=conv_type,
            num_relations=num_relations,
            num_bases=num_bases,
        )

        # RotatE decoder for scoring
        self.decoder = RotatEDecoder(
            num_relations=num_relations,
            fuse_dim=gnn_out_dim,
            pre_trained_rel_emb=pre_trained_rel_emb,
        )

        # Store static graph info
        self.global_edge_index = global_edge_index
        self.num_nodes = bloom_emb.size(0)

    def forward(
        self,
        head_idx: torch.Tensor,
        rel_idx: torch.Tensor,
        tail_idx: torch.Tensor,
        edge_type: torch.Tensor = None,
    ) -> torch.Tensor:
        # Combine head and tail indices and get unique nodes
        combined = torch.cat([head_idx, tail_idx])
        unique_nodes, _ = torch.unique(combined, return_inverse=True)

        # Mask to select edges in the induced subgraph
        device = unique_nodes.device
        mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=device)
        mask[unique_nodes] = True
        edge_index = self.global_edge_index.to(device)
        sel = mask[edge_index[0]] & mask[edge_index[1]]
        sub_edge_index = edge_index[:, sel]

        # Remap global node IDs to local indices for the subgraph
        unique_sorted, _ = torch.sort(unique_nodes)
        remap_src = torch.bucketize(sub_edge_index[0], unique_sorted)
        remap_dst = torch.bucketize(sub_edge_index[1], unique_sorted)
        local_edge_index = torch.stack([remap_src, remap_dst], dim=0)

        # Initial embeddings from Bloom+TransE
        x_init = self.hybrid_encoder(unique_sorted)
        # Message passing with chosen GNN
        x = self.gnn(x_init, local_edge_index, edge_type)

        # Gather head/tail embeddings
        head_pos = torch.searchsorted(unique_sorted, head_idx)
        tail_pos = torch.searchsorted(unique_sorted, tail_idx)
        h_emb = x[head_pos]
        t_emb = x[tail_pos]

        # Score triples
        return self.decoder(h_emb, rel_idx, t_emb)

