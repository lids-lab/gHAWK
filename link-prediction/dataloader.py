from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scripts.generate_bloom import generate_bloom
from logger import get_logger
from bloom_utils import build_fixed_bloom
from models.hybrid_model import build_global_edge_index


def _load_triples(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a split (train/valid/test) from a .pt file and return head, relation, tail tensors.
    """
    data = torch.load(path)
    heads = data['head']
    rels  = data['relation']
    tails = data['tail']
    if isinstance(heads, np.ndarray):
        heads, rels, tails = map(torch.from_numpy, (heads, rels, tails))
    return heads, rels, tails


def load_data(args) -> dict:
    """
    Load train triples, build global edge index, generate or load Bloom filters,
    load TransE embeddings, and return relevant objects (including a DataLoader).
    """
    logger = get_logger(__name__)

    # ----- STEP A: Load Train Data -----
    train_pt = Path(args.train_pt)
    assert train_pt.exists(), f"Train file not found at {train_pt}"
    logger.info(f"Loading train data from {train_pt}")
    heads, rels, tails = _load_triples(train_pt)

    num_nodes = max(heads.max().item(), tails.max().item()) + 1
    logger.info(f"Train triples: {len(heads)}, num_nodes={num_nodes}")

    # Build undirected global edge index
    global_edge_index = build_global_edge_index(heads, tails)
    logger.info(f"Global edge index shape: {global_edge_index.shape}")

    # ----- STEP B: Bloom Filter Generation or Loading -----
    bloom_path = Path(args.bloom_out)
    '''if args.generate_bloom:
        logger.info("Generating Bloom filters from scratch...")
        heads_np, rels_np, tails_np = heads.numpy(), rels.numpy(), tails.numpy()
        bloom_arrays = build_fixed_bloom(
            heads_np, rels_np, tails_np,
            num_nodes=num_nodes,
            bloom_size=args.bloom_size,
            num_hashes=args.num_hashes
        )
        bloom_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(bloom_path, bloom_arrays)
        logger.info(f"Saved Bloom arrays to {bloom_path}")
    else:
        assert bloom_path.exists(), f"Bloom file not found at {bloom_path}"
        logger.info(f"Loading Bloom arrays from {bloom_path}")
        bloom_arrays = np.load(bloom_path)'''
    # ----- STEP B: Bloom Filter Generation or Loading -----
    if args.generate_bloom:
        # regenerate and save via helper
        bloom_emb = generate_bloom(
            heads, rels, tails,
            num_nodes   = num_nodes,
            bloom_out   = args.bloom_out,
            bloom_size  = args.bloom_size,
            num_hashes  = args.num_hashes,
        )
        bloom_arrays = bloom_emb.cpu().numpy()
    else:
        bloom_path = Path(args.bloom_out)
        assert bloom_path.exists(), f"Bloom file not found at {bloom_path}"
        logger.info(f"Loading Bloom arrays from {bloom_path}")
        bloom_arrays = np.load(bloom_path)
        bloom_emb = torch.tensor(bloom_arrays, dtype=torch.float32)
        logger.info(f"Bloom embedding shape: {bloom_emb.shape}")

    assert bloom_arrays.shape[0] == num_nodes, \
        f"Bloom array row count {bloom_arrays.shape[0]} does not match num_nodes {num_nodes}"
    logger.info(f"Bloom embedding shape: {bloom_emb.shape}")

    # ----- STEP C: Load TransE Embeddings -----
    transE_path = Path(args.transE_path)
    assert transE_path.exists(), f"TransE entity embedding not found at {transE_path}"
    logger.info(f"Loading TransE entity embeddings from {transE_path}")
    transE_np = np.load(transE_path)
    transE_emb = torch.tensor(transE_np, dtype=torch.float32)
    logger.info(f"TransE entity embedding shape: {transE_emb.shape}")

    transE_rel_path = Path(args.transE_rel_path)
    assert transE_rel_path.exists(), f"TransE relation embedding not found at {transE_rel_path}"
    logger.info(f"Loading TransE relation embeddings from {transE_rel_path}")
    transE_rel_np = np.load(transE_rel_path)
    transE_rel_emb = torch.tensor(transE_rel_np, dtype=torch.float32)
    logger.info(f"TransE relation embedding shape: {transE_rel_emb.shape}")

    # ----- STEP D: Build Train DataLoader -----
    dataset = TensorDataset(heads, rels, tails)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, 'num_workers', 0)
    )
    logger.info(f"Train loader prepared: {len(train_loader)} batches of size {args.batch_size}")

    return {
        'heads': heads,
        'rels': rels,
        'tails': tails,
        'num_nodes': num_nodes,
        'global_edge_index': global_edge_index,
        'bloom_emb': bloom_emb,
        'transE_emb': transE_emb,
        'transE_rel_emb': transE_rel_emb,
        'train_loader': train_loader
    }