
import torch
import torch.nn as nn

############################################
#           ROTATE DECODER
############################################
class RotatEDecoder(nn.Module):
    """
    RotatE scoring with margin-based ranking loss.
    Splits node embeddings into real and imaginary parts.
    Relation embeddings are stored as angles (radians).
    """
    def __init__(self, num_relations, fuse_dim, pre_trained_rel_emb=None):
        super().__init__()
        assert fuse_dim % 2 == 0, "fuse_dim must be even for RotatE."
        self.fuse_dim = fuse_dim
        self.half_dim = fuse_dim // 2
        if pre_trained_rel_emb is None:
            self.rel_emb = nn.Embedding(num_relations, self.half_dim)
            nn.init.uniform_(self.rel_emb.weight, a=-3.14159, b=3.14159)
        else:
            orig_dim = pre_trained_rel_emb.shape[1]
            self.rel_proj = nn.Linear(orig_dim, self.half_dim, bias=False)
            projected = self.rel_proj(torch.tensor(pre_trained_rel_emb, dtype=torch.float))
            self.rel_emb = nn.Embedding.from_pretrained(projected, freeze=False)

    def forward(self, h_emb, r_idx, t_emb):
        h_real, h_imag = h_emb[:, :self.half_dim], h_emb[:, self.half_dim:]
        t_real, t_imag = t_emb[:, :self.half_dim], t_emb[:, self.half_dim:]
        r_angles = self.rel_emb(r_idx)  # [batch, half_dim]
        r_real = torch.cos(r_angles)
        r_imag = torch.sin(r_angles)
        h_rot_real = h_real * r_real - h_imag * r_imag
        h_rot_imag = h_real * r_imag + h_imag * r_real
        diff_real = h_rot_real - t_real
        diff_imag = h_rot_imag - t_imag
        dist = torch.sqrt(torch.sum(diff_real**2 + diff_imag**2, dim=1))
        score = -dist
        return score

############################################
#           DistMult+alpha* TransE  DECODER
############################################

class DistMultTransEDecoder(nn.Module):
    def __init__(self, num_relations, fuse_dim, alpha=0.1, pre_trained_rel_emb=None):
        super().__init__()
        self.fuse_dim = fuse_dim
        self.alpha = alpha
        
        if pre_trained_rel_emb is None:
            self.rel_emb = nn.Embedding(num_relations, fuse_dim)
            nn.init.xavier_uniform_(self.rel_emb.weight)
        else:
            orig_dim = pre_trained_rel_emb.shape[1]
            self.rel_proj = nn.Linear(orig_dim, fuse_dim, bias=False)
            self.register_buffer("rel_emb_pre", torch.tensor(pre_trained_rel_emb, dtype=torch.float))
            projected = self.rel_proj(self.rel_emb_pre)
            self.rel_emb = nn.Embedding.from_pretrained(projected, freeze=False)

    def forward(self, h_emb, r_idx, t_emb):
        r = self.rel_emb(r_idx)
        distmult_score = torch.sum(h_emb * r * t_emb, dim=1)
        trans_dist = torch.norm(h_emb + r - t_emb, p=2, dim=1)
        trans_score = -trans_dist
        return distmult_score + self.alpha * trans_score
    
############################################
#           DistMult   DECODER
############################################

class DistMultDecoder(nn.Module):
    """
    Computes DistMult-style scores.
    If fuse_dim = bloom_project_dim + transE_project_dim, then our relation embeddings should have that same dimension.
    If pre-trained relation embeddings are provided (of a different dimension), we project them into the fused space.
    """
    def __init__(self, num_relations, fuse_dim, pre_trained_rel_emb=None):
        super().__init__()
        self.fuse_dim = fuse_dim
        if pre_trained_rel_emb is None:
            # Initialize relation embeddings randomly in the fused space.
            self.rel_emb = nn.Embedding(num_relations, fuse_dim)
            nn.init.xavier_uniform_(self.rel_emb.weight)
        else:
            # If pre-trained relation embeddings are provided, project them to the fused dimension.
            # Assume pre_trained_rel_emb has shape [num_relations, orig_dim]
            orig_dim = pre_trained_rel_emb.shape[1]
            # Create a learnable linear projection layer.
            self.rel_proj = nn.Linear(orig_dim, fuse_dim)
            # Initialize by applying the linear projection on the pre-trained embeddings.
            projected = self.rel_proj(torch.tensor(pre_trained_rel_emb, dtype=torch.float))
            self.rel_emb = nn.Embedding.from_pretrained(projected, freeze=False)

    def forward(self, h_emb, r_idx, t_emb):
        r = self.rel_emb(r_idx)
        score = torch.sum(h_emb * r * t_emb, dim=1)
        return score

