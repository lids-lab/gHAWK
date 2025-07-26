
import mmh3  # pip install mmh3
import numpy as np
from tqdm import tqdm
from logger import get_logger

logger = get_logger(__name__)

############################################
#           BLOOM FILTER UTILS
############################################
def bloom_hashes(key: str, num_bits=500, num_hashes=3, seed=42):
    positions = []
    for i in range(num_hashes):
        h = mmh3.hash(key, seed + i, signed=False)
        pos = h % num_bits
        positions.append(pos)
    return positions

def build_fixed_bloom(heads, rels, tails, num_nodes, bloom_size=500, num_hashes=3):
    logger.info(f"Building Bloom filters with size={bloom_size}, num_hashes={num_hashes} ...")
    bloom_arrays = np.zeros((num_nodes, bloom_size), dtype=np.bool_)
    for h, r, t in tqdm(zip(heads, rels, tails), total=len(heads), desc="Bloom Construction"):
        key_for_h = f"{r}_{t}"
        positions_h = bloom_hashes(key_for_h, num_bits=bloom_size, num_hashes=num_hashes)
        for pos in positions_h:
            bloom_arrays[h, pos] = True

        key_for_t = f"{r}_{h}"
        positions_t = bloom_hashes(key_for_t, num_bits=bloom_size, num_hashes=num_hashes)
        for pos in positions_t:
            bloom_arrays[t, pos] = True
    return bloom_arrays.astype(np.uint8)
