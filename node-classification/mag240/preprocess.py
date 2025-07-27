#!/usr/bin/env python3

"""
preprocessing.py — Generate Bloom filters and TransE embeddings for MAG240M
"""

import argparse
import os
from collections import defaultdict

import mmh3
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.format import open_memmap
from ogb.lsc import MAG240MDataset


def run_bloom(root: str, m: int, k: int) -> None:
    """
    Generate Bloom filters for authors and institutions.

    Args:
        root: Path to MAG240M root directory.
        m: Number of bits per filter.
        k: Number of hash functions.
    """
    # Parameters
    print(f"Bloom mode: m={m}, k={k}")

    # Load dataset
    dataset = MAG240MDataset(root=root)
    Na = dataset.num_authors
    Ni = dataset.num_institutions
    print(f"Number of authors: {Na}, institutions: {Ni}")

    # Extract edges
    writes = dataset.edge_index('author', 'writes', 'paper')
    affils = dataset.edge_index('author', 'affiliated_with', 'institution')

    # Build inverted lists
    author_papers = defaultdict(list)
    for a, p in zip(writes[0], writes[1]):
        author_papers[int(a)].append(int(p))

    author_institutions = defaultdict(list)
    for a, i in zip(affils[0], affils[1]):
        author_institutions[int(a)].append(int(i))

    # Bytes per filter
    B = (m + 7) // 8
    print(f"Each filter occupies {B} bytes")

    # Create output directory
    os.makedirs('Bloom', exist_ok=True)

    bloom_author = open_memmap(
        filename='Bloom/bloom_author.npy',
        dtype='uint8',
        mode='w+',
        shape=(Na, B),
    )
    bloom_institution = open_memmap(
        filename='Bloom/bloom_institution.npy',
        dtype='uint8',
        mode='w+',
        shape=(Ni, B),
    )

    # Populate author filters
    for aid in range(Na):
        row = np.zeros(B, dtype='uint8')
        # Papers
        for p in author_papers.get(aid, []):
            key = f"paper:{p}"
            for seed in range(k):
                idx = mmh3.hash(key, seed) % m
                row[idx // 8] |= (1 << (idx % 8))
        # Institutions
        for inst in author_institutions.get(aid, []):
            key = f"institution:{inst}"
            for seed in range(k):
                idx = mmh3.hash(key, seed) % m
                row[idx // 8] |= (1 << (idx % 8))
        bloom_author[aid] = row
        if aid and aid % 1_000_000 == 0:
            print(f"  → Built filters for {aid}/{Na} authors")

    # Populate institution filters (invert relation)
    institution_authors = defaultdict(list)
    for a, i in zip(affils[0], affils[1]):
        institution_authors[int(i)].append(int(a))

    for iid in range(Ni):
        row = np.zeros(B, dtype='uint8')
        for a in institution_authors.get(iid, []):
            key = f"author:{a}"
            for seed in range(k):
                idx = mmh3.hash(key, seed) % m
                row[idx // 8] |= (1 << (idx % 8))
        bloom_institution[iid] = row
        if iid and iid % 1_000_000 == 0:
            print(f"  → Built filters for {iid}/{Ni} institutions")

    bloom_author.flush()
    bloom_institution.flush()
    print(f"Saved bloom_author.npy ({bloom_author.shape})")
    print(f"Saved bloom_institution.npy ({bloom_institution.shape})")


def run_transe(root: str,
               embedding_dim: int,
               batch_size: int,
               lr: float,
               margin: float,
               epochs: int,
               device_id: int) -> None:
    """
    Train TransE embeddings for all nodes in MAG240M.

    Args:
        root: Path to MAG240M root directory.
        embedding_dim: Dimension of embeddings.
        batch_size: Number of triples per batch.
        lr: Learning rate.
        margin: Margin for ranking loss.
        epochs: Number of training epochs.
        device_id: CUDA device index.
    """
    print(f"TransE mode: dim={embedding_dim}, batch={batch_size}, lr={lr},"
          f" margin={margin}, epochs={epochs}")

    dataset = MAG240MDataset(root=root)
    Np = dataset.num_papers
    Na = dataset.num_authors
    Ni = dataset.num_institutions
    N_total = Np + Na + Ni
    print(f"Entities — papers: {Np}, authors: {Na}, inst: {Ni}, total: {N_total}")

    # Relation triples
    cites = dataset.edge_index('paper', 'cites', 'paper')
    writes = dataset.edge_index('author', 'writes', 'paper')
    affils = dataset.edge_index('author', 'affiliated_with', 'institution')

    paper_off = 0
    author_off = Np
    inst_off = Np + Na

    # Build triples
    heads_c = torch.from_numpy(cites[0].astype('int64')) + paper_off
    tails_c = torch.from_numpy(cites[1].astype('int64')) + paper_off
    rels_c = torch.zeros_like(heads_c)

    heads_w = torch.from_numpy(writes[0].astype('int64')) + author_off
    tails_w = torch.from_numpy(writes[1].astype('int64')) + paper_off
    rels_w = torch.ones_like(heads_w)

    heads_a = torch.from_numpy(affils[0].astype('int64')) + author_off
    tails_a = torch.from_numpy(affils[1].astype('int64')) + inst_off
    rels_a = torch.full_like(heads_a, 2)

    heads = torch.cat([heads_c, heads_w, heads_a], dim=0)
    tails = torch.cat([tails_c, tails_w, tails_a], dim=0)
    rels = torch.cat([rels_c, rels_w, rels_a], dim=0)
    triples = torch.stack([heads, tails, rels], dim=0)
    E = triples.size(1)
    print(f"Total triples: {E}")

    # Output memmap
    os.makedirs('TransE', exist_ok=True)
    mmap_path = 'TransE/all_transe.npy'
    emb_mmap = open_memmap(
        filename=mmap_path,
        dtype='float32',
        mode='w+',
        shape=(N_total, embedding_dim),
    )

    # Initialize embeddings
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    ent_emb = nn.Embedding(N_total, embedding_dim, sparse=True).to(device)
    rel_emb = nn.Embedding(3, embedding_dim, sparse=True).to(device)
    nn.init.xavier_uniform_(ent_emb.weight)
    nn.init.xavier_uniform_(rel_emb.weight)
    ent_emb.weight.data = ent_emb.weight.data.half()
    rel_emb.weight.data = rel_emb.weight.data.half()

    optimizer = torch.optim.SGD(
        [{'params': ent_emb.parameters()}, {'params': rel_emb.parameters()}],
        lr=lr
    )

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(E, device='cpu')
        shuf = triples[:, perm]
        total_loss = 0.0
        steps = (E + batch_size - 1) // batch_size
        print(f"Epoch {epoch}/{epochs}, steps={steps}")

        for i in range(0, E, batch_size):
            batch = shuf[:, i:i + batch_size].to(device)
            h_idx, t_idx, r_idx = batch
            neg_t = torch.randint(0, N_total, h_idx.shape, device=device)

            h = ent_emb(h_idx)
            t = ent_emb(t_idx)
            r = rel_emb(r_idx)
            t_neg = ent_emb(neg_t)

            pos = (h + r - t).abs().sum(dim=1)
            neg = (h + r - t_neg).abs().sum(dim=1)
            loss = (pos + margin - neg).clamp(min=0).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"  -> Epoch {epoch} avg loss: {total_loss/steps:.4f}")

    # Write embeddings
    print("Writing embeddings to memmap...")
    chunk = 1_000_000
    weight = ent_emb.weight.data.float().cpu().numpy()
    for start in range(0, N_total, chunk):
        end = min(start + chunk, N_total)
        emb_mmap[start:end] = weight[start:end]
        print(f"  Wrote rows {start}:{end}")
    emb_mmap.flush()
    print(f"Saved TransE embeddings to {mmap_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Bloom filters or TransE embeddings for MAG240M"
    )
    parser.add_argument('--mode', required=True, choices=['bloom', 'transe'],
                        help='Choose preprocessing mode')
    parser.add_argument('--root', type=str, default='dataset',
                        help='MAG240M dataset root directory')

    # Bloom args
    parser.add_argument('--m', type=int, default=500,
                        help='Bits per Bloom filter')
    parser.add_argument('--k', type=int, default=7,
                        help='Number of hash functions')

    # TransE args
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='TransE embedding dimension')
    parser.add_argument('--batch_size', type=int, default=128000,
                        help='TransE training batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for TransE')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for TransE ranking loss')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of TransE epochs')
    parser.add_argument('--device_id', type=int, default=0,
                        help='CUDA device ID')

    args = parser.parse_args()

    if args.mode == 'bloom':
        run_bloom(args.root, args.m, args.k)
    else:
        run_transe(
            args.root,
            args.embedding_dim,
            args.batch_size,
            args.lr,
            args.margin,
            args.epochs,
            args.device_id,
        )


if __name__ == '__main__':
    main()

