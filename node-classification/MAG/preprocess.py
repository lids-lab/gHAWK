#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocessing_ogbn_mag.py — Generate Bloom filters and TransE embeddings for OGBN-MAG
"""

import argparse
import os
from collections import defaultdict

import mmh3
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.format import open_memmap
from torch_geometric.datasets import PygNodePropPredDataset
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected


def run_bloom(root: str, m: int, k: int) -> None:
    """
    Generate Bloom filters for authors and institutions in OGBN-MAG.

    Args:
        root: Path to OGBN-MAG root directory.
        m: Number of bits per filter.
        k: Number of hash functions.
    """
    print(f"[Bloom] m={m}, k={k}")
    dataset = PygNodePropPredDataset(name='ogbn-mag', root=root)
    data = dataset[0]
    # Extract node counts
    num_papers = data.num_nodes_dict['paper']
    num_authors = data.num_nodes_dict['author']
    num_institutions = data.num_nodes_dict['institution']
    print(f"#papers={num_papers}, #authors={num_authors}, #institutions={num_institutions}")

    # Extract edges
    writes = data.edge_index_dict[('author', 'writes', 'paper')]  # [2, E]
    affils = data.edge_index_dict[('author', 'affiliated_with', 'institution')]

    # Build inverted lists
    author_papers = defaultdict(list)
    for a, p in zip(writes[0].tolist(), writes[1].tolist()):
        author_papers[a].append(p)
    author_insts = defaultdict(list)
    for a, i in zip(affils[0].tolist(), affils[1].tolist()):
        author_insts[a].append(i)

    # Bytes per filter
    B = (m + 7) // 8
    os.makedirs('Bloom', exist_ok=True)
    bloom_author = open_memmap('Bloom/bloom_author.npy', mode='w+', dtype='uint8', shape=(num_authors, B))
    bloom_inst   = open_memmap('Bloom/bloom_institution.npy', mode='w+', dtype='uint8', shape=(num_institutions, B))

    # Populate author filters
    for aid in range(num_authors):
        row = np.zeros(B, dtype='uint8')
        for p in author_papers.get(aid, []):
            key = f"paper:{p}"
            for seed in range(k):
                idx = mmh3.hash(key, seed) % m
                row[idx // 8] |= (1 << (idx % 8))
        for inst in author_insts.get(aid, []):
            key = f"institution:{inst}"
            for seed in range(k):
                idx = mmh3.hash(key, seed) % m
                row[idx // 8] |= (1 << (idx % 8))
        bloom_author[aid] = row
        if aid and aid % 100000 == 0:
            print(f"  → {aid}/{num_authors} authors done")

    # Build reverse affiliation for institutions
    inst_auths = defaultdict(list)
    for a, i in zip(affils[0].tolist(), affils[1].tolist()):
        inst_auths[i].append(a)
    for iid in range(num_institutions):
        row = np.zeros(B, dtype='uint8')
        for a in inst_auths.get(iid, []):
            key = f"author:{a}"
            for seed in range(k):
                idx = mmh3.hash(key, seed) % m
                row[idx // 8] |= (1 << (idx % 8))
        bloom_inst[iid] = row
        if iid and iid % 100000 == 0:
            print(f"  → {iid}/{num_institutions} institutions done")

    bloom_author.flush()
    bloom_inst.flush()
    print(f"Saved Bloom filters: bloom_author.npy shape={bloom_author.shape}, bloom_institution.npy shape={bloom_inst.shape}")


def run_transe(root: str,
               embedding_dim: int,
               batch_size: int,
               lr: float,
               margin: float,
               epochs: int,
               device_id: int) -> None:
    """
    Train TransE embeddings for OGBN-MAG.

    Args:
        root: Path to OGBN-MAG root directory.
        embedding_dim: Embedding dimensionality.
        batch_size: Triple batch size.
        lr: Learning rate.
        margin: Ranking loss margin.
        epochs: Number of epochs.
        device_id: CUDA device.
    """
    print(f"[TransE] dim={embedding_dim}, batch={batch_size}, lr={lr}, margin={margin}, epochs={epochs}")
    dataset = PygNodePropPredDataset(name='ogbn-mag', root=root)
    data = dataset[0]
    # Node counts
    np_ = data.num_nodes_dict['paper']
    na = data.num_nodes_dict['author']
    ni = data.num_nodes_dict['institution']
    N = np_ + na + ni
    print(f"#entities total={N} (paper={np_}, author={na}, inst={ni})")

    # Extract and unify edges
    cites = data.edge_index_dict[('paper','cites','paper')]
    writes= data.edge_index_dict[('author','writes','paper')]
    affils= data.edge_index_dict[('author','affiliated_with','institution')]
    # Convert to global IDs
    heads = []
    tails = []
    rels  = []
    # cites rel=0
    heads.append(cites[0]         )
    tails.append(cites[1]         )
    rels .append(torch.zeros_like(cites[0]))
    # writes rel=1
    heads.append(writes[0] + np_)
    tails.append(writes[1]      )
    rels .append(torch.ones_like(writes[0]))
    # affils rel=2
    heads.append(affils[0] + np_)
    tails.append(affils[1] + np_ + na)
    rels .append(torch.full_like(affils[0], 2))

    heads = torch.cat(heads, dim=0)
    tails = torch.cat(tails, dim=0)
    rels  = torch.cat(rels , dim=0)
    triples = torch.stack([heads, tails, rels], dim=0)
    E = triples.size(1)
    print(f"Total triples={E}")

    # Prepare memmap for embeddings
    os.makedirs('TransE', exist_ok=True)
    mmap_path = 'TransE/all_transe.npy'
    emb_mmap = open_memmap(mmap_path, mode='w+', dtype='float32', shape=(N, embedding_dim))

    # Initialize embeddings
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    ent_emb = nn.Embedding(N, embedding_dim, sparse=True).to(device)
    rel_emb = nn.Embedding(3, embedding_dim, sparse=True).to(device)
    nn.init.xavier_uniform_(ent_emb.weight)
    nn.init.xavier_uniform_(rel_emb.weight)
    ent_emb.weight.data = ent_emb.weight.data.half()
    rel_emb.weight.data = rel_emb.weight.data.half()

    optimizer = torch.optim.SGD([
        {'params': ent_emb.parameters()},
        {'params': rel_emb.parameters()}], lr=lr)

    for epoch in range(1, epochs+1):
        perm = torch.randperm(E, device='cpu')
        shuf = triples[:, perm]
        total_loss = 0.0
        steps = (E + batch_size - 1) // batch_size
        print(f"Epoch {epoch}/{epochs}, steps={steps}")
        for i in range(0, E, batch_size):
            batch = shuf[:, i:i+batch_size].to(device)
            h_idx, t_idx, r_idx = batch
            neg_t = torch.randint(0, N, h_idx.shape, device=device)
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
        print(f"  -> Epoch {epoch} avg loss={total_loss/steps:.4f}")

    print("Writing embeddings to memmap...")
    weight = ent_emb.weight.data.float().cpu().numpy()
    for start in range(0, N, 1000000):
        end = min(start+1000000, N)
        emb_mmap[start:end] = weight[start:end]
        print(f"  Wrote rows {start}:{end}")
    emb_mmap.flush()
    print(f"Saved TransE to {mmap_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess OGBN-MAG: bloom or transe")
    parser.add_argument('--mode', required=True, choices=['bloom','transe'])
    parser.add_argument('--root', type=str, default='dataset')
    parser.add_argument('--m', type=int, default=500)
    parser.add_argument('--k', type=int, default=7)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--device_id', type=int, default=0)
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
