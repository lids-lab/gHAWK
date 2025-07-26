import numpy as np
import torch
from pathlib import Path

from bloom_utils import build_fixed_bloom
from logger import get_logger

logger = get_logger(__name__)

def generate_bloom(
    heads: torch.Tensor,
    rels: torch.Tensor,
    tails: torch.Tensor,
    num_nodes: int,
    bloom_out: str,
    bloom_size: int = 500,
    num_hashes: int = 3,
) -> torch.Tensor:
    """
    Generate Bloom filters from train triples and save to disk.

    Args:
        heads: Tensor of head entity IDs.
        rels:  Tensor of relation IDs.
        tails: Tensor of tail entity IDs.
        num_nodes: Total number of entities (for Bloom array rows).
        bloom_out: Path to save the .npy Bloom arrays.
        bloom_size: Number of bits per Bloom filter.
        num_hashes: Number of hash functions.

    Returns:
        bloom_tensor: A FloatTensor of shape [num_nodes, bloom_size].
    """
    out_path = Path(bloom_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Generating Bloom filters (size={bloom_size}, hashes={num_hashes})"
    )
    # Convert to numpy arrays
    heads_np, rels_np, tails_np = heads.numpy(), rels.numpy(), tails.numpy()
    # Build and save
    bloom_arrays = build_fixed_bloom(
        heads_np, rels_np, tails_np,
        num_nodes=num_nodes,
        bloom_size=bloom_size,
        num_hashes=num_hashes,
    )
    np.save(out_path, bloom_arrays)
    logger.info(f"Saved Bloom arrays to {out_path}")

    # Return as torch Tensor
    bloom_tensor = torch.tensor(bloom_arrays, dtype=torch.float32)
    logger.info(f"Bloom tensor shape: {bloom_tensor.shape}")
    return bloom_tensor
