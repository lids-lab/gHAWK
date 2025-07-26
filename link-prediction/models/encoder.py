
import torch
import torch.nn as nn
############################################
#           HYBRID NODE ENCODER
############################################
class HybridNodeEncoder(nn.Module):
    """
    Computes an initial node feature vector by combining:
      - A Bloom filter representation (local structure)
      - A TransE entity embedding (global structure)
    Each is processed via its own MLP before concatenation.
    """
    def __init__(self, bloom_emb: torch.Tensor, transE_emb: torch.Tensor,
                 bloom_project_dim=256, transE_proj_dim=256):
        super().__init__()
        # Store precomputed arrays as buffers
        self.register_buffer("bloom_data", bloom_emb)   # [num_nodes, bloom_dim]
        self.register_buffer("transE_data", transE_emb)   # [num_nodes, transE_dim]
        self.bloom_dim = bloom_emb.size(1)
        self.transE_dim = transE_emb.size(1)
        # MLP for Bloom features
        self.bloom_mlp = nn.Sequential(
            nn.Linear(self.bloom_dim, bloom_project_dim),
            nn.ReLU(),
            nn.Linear(bloom_project_dim, bloom_project_dim)
        )
        # MLP for TransE features
        self.transe_mlp = nn.Sequential(
            nn.Linear(self.transE_dim, transE_proj_dim),
            nn.ReLU(),
            nn.Linear(transE_proj_dim, transE_proj_dim)
        )

    def forward(self, node_idx: torch.Tensor) -> torch.Tensor:
        device = node_idx.device
        node_idx_cpu = node_idx.cpu()
        bloom_cpu = self.bloom_data[node_idx_cpu]
        transE_cpu = self.transE_data[node_idx_cpu]
        bloom_gpu = bloom_cpu.to(device, non_blocking=True)
        transE_gpu = transE_cpu.to(device, non_blocking=True)
        bloom_rep = self.bloom_mlp(bloom_gpu)
        transE_rep = self.transe_mlp(transE_gpu)
        fused = torch.cat([bloom_rep, transE_rep], dim=1)
        return fused