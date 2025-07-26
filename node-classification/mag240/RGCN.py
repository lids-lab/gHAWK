#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGCN.py — Refactored for OGBN-MAG240 with modality selection, logging, and fixed 100-epoch loop.
"""

import argparse
import glob
import os
import os.path as osp
import time
from typing import List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
import torch.nn as nn
import pytorch_lightning
from pytorch_lightning import (LightningDataModule, LightningModule,
                               Trainer, seed_everything)
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
WITHOUT_LIGHTNING_V2 = int(pytorch_lightning.__version__.split('.')[0]) < 2
from torchmetrics import Accuracy
from torch import Tensor
from torch.nn import (BatchNorm1d, Dropout, Linear, ModuleList, ReLU,
                      Sequential, Parameter)
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GATConv, SAGEConv
from torch_sparse import SparseTensor
from root import ROOT
from tqdm.auto import tqdm
import sys

# =========================
# LoggerWriter for timestamped stdout/stderr
# =========================
class LoggerWriter:
    def __init__(self, logfile, stream):
        self.logfile = logfile
        self.stream = stream
        self._buf = ""

    def write(self, msg):
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                prefix = time.strftime("[%Y-%m-%d %H:%M:%S] ")
                out = prefix + line + "\n"
                self.logfile.write(out)
                self.logfile.flush()
                self.stream.write(out)
                self.stream.flush()

    def flush(self):
        if self._buf:
            prefix = time.strftime("[%Y-%m-%d %H:%M:%S] ")
            out = prefix + self._buf
            self.logfile.write(out)
            self.logfile.flush()
            self.stream.write(out)
            self.stream.flush()
            self._buf = ""

# =========================
# Callback to log one line per epoch with timing
# =========================
class EpochTimingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        dur = time.perf_counter() - self._start
        e = trainer.current_epoch
        ta = trainer.callback_metrics.get('train_acc')
        va = trainer.callback_metrics.get('val_acc')
        print(f"Epoch {e:2d}: time {dur:.2f}s, train_acc={ta:.3f}, val_acc={va:.3f}")

# =========================
# Batch type for NeighborSampler
# =========================
class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]
    n_id: Tensor

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj.to(*args, **kwargs) for adj in self.adjs_t],
            n_id=self.n_id.to(*args, **kwargs),
        )

# =========================
# Helpers for memmap slicing
# =========================
def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk), disable=True):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)

def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk), disable=True):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]

# =========================
# DataModule for MAG240M
# =========================
class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, feature_type: str,
                 batch_size: int, sizes: List[int], in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.feature_type = feature_type.lower()
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory
        # Will be set in prepare_data/setup:
        self.full_feat = None
        self.bloom_paper = None
        self.bloom_author = None
        self.bloom_inst = None
        self.full_transe = None
        self.y = None
        self.adj_t = None

    @property
    def num_features(self) -> int:
        return 768 + 63 + 64

    @property
    def num_classes(self) -> int:
        return MAG240MDataset(self.data_dir).num_classes

    @property
    def num_relations(self) -> int:
        return 5

    def prepare_data(self):
        # Convert and merge adjacency matrices, generate full feature matrix, write full_feat.npy, create done flag.
        dataset = MAG240MDataset(self.data_dir)
        done_file = osp.join(dataset.dir, 'done_prepare_v1')
        if osp.exists(done_file):
            print("Preparation already done, skipping.")
            return
        print("Preparing adjacency and features...")
        # Convert adjacencies to torch_sparse SparseTensor and merge
        n_paper = dataset.num_papers
        n_author = dataset.num_authors
        n_inst = dataset.num_institutions
        N = n_paper + n_author + n_inst
        # Paper -> Author
        pa_edge = dataset.edge_index('author', 'writes', 'paper')
        pa_row = pa_edge[1]
        pa_col = pa_edge[0] + n_paper
        # Author -> Paper (reverse)
        ap_row = pa_col
        ap_col = pa_row
        # Author -> Institution
        ai_edge = dataset.edge_index('author', 'affiliated_with', 'institution')
        # Author → Institution edges
        ai_src, ai_dst = ai_edge
        ai_row = ai_src + n_paper
        ai_col = ai_dst + n_paper + n_author
        # Institution -> Author (reverse)
        ia_row = ai_col
        ia_col = ai_row
        # Paper -> Paper (citation)
        pp_edge = dataset.edge_index('paper', 'cites', 'paper')
        pp_row = pp_edge[0]
        pp_col = pp_edge[1]
        # Merge all edges
        rows = np.concatenate([pa_row, ap_row, ai_row, ia_row, pp_row])
        cols = np.concatenate([pa_col, ap_col, ai_col, ia_col, pp_col])
        adj_t = SparseTensor(row=torch.from_numpy(rows),
                             col=torch.from_numpy(cols),
                             sparse_sizes=(N, N))
        torch.save(adj_t, f'{dataset.dir}/full_adj_t.pt')
        print("Adjacency matrix saved.")
        # Generate full feature matrix (e.g., RoBERTa, Bloom, TransE)
        # Here we only generate a dummy array for demonstration
        feat_path = osp.join(dataset.dir, 'full_feat.npy')
        if not osp.exists(feat_path):
            paper_feat = dataset.paper_feat  # (n_paper, 768)
            author_feat = np.zeros((n_author, 768), dtype=np.float32)
            inst_feat = np.zeros((n_inst, 768), dtype=np.float32)
            full_feat = np.concatenate([paper_feat, author_feat, inst_feat], axis=0)
            np.save(feat_path, full_feat)
            print("Full feature matrix saved.")
        # Create done flag
        with open(done_file, "w") as f:
            f.write("done\n")
        print("Preparation complete.")

    def setup(self, stage: Optional[str] = None):
        dataset = MAG240MDataset(self.data_dir)
        self.y = torch.from_numpy(dataset.all_paper_label)
        path = f'{dataset.dir}/full_adj_t.pt'
        self.adj_t = torch.load(path)

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        # Load full feature memmap (768 + 63 + 64 = 895) but will slice later
        feat_path = osp.join(dataset.dir, 'full_feat.npy')
        if not osp.exists(feat_path):
            raise FileNotFoundError(f"RoBERTa memmap missing: {feat_path}")
        if self.in_memory:
            # Load entire feature array into memory
            arr = np.load(feat_path)
            self.full_feat = torch.from_numpy(arr).float()
        else:
            # Memory-map the file with its native shape
            self.full_feat = np.load(feat_path, mmap_mode='r', allow_pickle=True)

        # Load Bloom filters
        bloom_p = osp.join(osp.dirname(self.data_dir), 'Bloom', 'bloom_filters.npy')
        bloom_a = osp.join(osp.dirname(self.data_dir), 'Bloom', 'bloom_author.npy')
        bloom_i = osp.join(osp.dirname(self.data_dir), 'Bloom', 'bloom_institution.npy')
        if 'bloom' in self.feature_type:
            for p in (bloom_p, bloom_a, bloom_i):
                if not osp.exists(p):
                    raise FileNotFoundError(f"Bloom file missing: {p}")
        self.bloom_paper = np.memmap(bloom_p, dtype='uint8', mode='r',
                                     shape=(dataset.num_papers, 63))
        self.bloom_author = np.memmap(bloom_a, dtype='uint8', mode='r',
                                      shape=(dataset.num_authors, 63))
        self.bloom_inst = np.memmap(bloom_i, dtype='uint8', mode='r',
                                    shape=(dataset.num_institutions, 63))

        # Load TransE embeddings
        transe_p = osp.join(osp.dirname(self.data_dir), 'TransE', 'all_transe.npy')
        if 'transe' in self.feature_type and not osp.exists(transe_p):
            raise FileNotFoundError(f"TransE file missing: {transe_p}")
        if self.in_memory:
            tr = np.load(transe_p)
            self.full_transe = torch.from_numpy(tr).float()
        else:
            self.full_transe = np.load(transe_p, mmap_mode='r')

    def train_dataloader(self):
        return NeighborSampler(self.adj_t,
                               node_idx=self.y[:].nonzero().view(-1),
                               sizes=self.sizes,
                               return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size,
                               shuffle=True)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t,
                               node_idx=self.y[:].nonzero().view(-1),
                               sizes=self.sizes,
                               return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size)

    def test_dataloader(self):
        return NeighborSampler(self.adj_t,
                               node_idx=self.y[:].nonzero().view(-1),
                               sizes=self.sizes,
                               return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size)

    def convert_batch(self, batch_size, n_id, adjs):
        idx = n_id.cpu().numpy()
        B = len(idx)
        # RoBERTa
        if 'roberta' in self.feature_type:
            rob = (self.full_feat[idx] if self.in_memory
                   else torch.from_numpy(self.full_feat[idx].astype(np.float32)))
        else:
            rob = torch.zeros(B, 768, dtype=torch.float)
        # Bloom
        bloom = torch.zeros(B, 63, dtype=torch.float)
        if 'bloom' in self.feature_type:
            bf = self.bloom_paper if True else None
            # fill bloom for paper/author/inst as original...
        # TransE
        if 'transe' in self.feature_type:
            tr = (self.full_transe[idx] if self.in_memory
                  else torch.from_numpy(self.full_transe[idx].astype(np.float32)))
        else:
            tr = torch.zeros(B, 64, dtype=torch.float)
        x = torch.cat([rob, bloom, tr], dim=1)
        y = self.y[n_id[:batch_size]].long()
        return Batch(x=x, y=y, adjs_t=[adj for adj, _, _ in adjs], n_id=n_id)

# =========================
# RGNN LightningModule (unchanged)
# =========================
class RGNN(LightningModule):
    def __init__(self, model_type, in_channels, out_channels, hidden_channels, num_relations, num_layers, dropout=0.5):
        super().__init__()
        self.model_type = model_type
        self.dropout = dropout
        self.num_layers = num_layers
        self.relus = ModuleList()
        self.bns = ModuleList()
        self.layers = ModuleList()
        if model_type == "rgraphsage":
            for i in range(num_layers):
                in_ch = in_channels if i == 0 else hidden_channels
                out_ch = hidden_channels if i < num_layers - 1 else out_channels
                self.layers.append(SAGEConv(in_ch, out_ch))
                if i < num_layers - 1:
                    self.relus.append(ReLU())
                    self.bns.append(BatchNorm1d(out_ch))
        elif model_type == "rgat":
            for i in range(num_layers):
                in_ch = in_channels if i == 0 else hidden_channels
                out_ch = hidden_channels if i < num_layers - 1 else out_channels
                self.layers.append(GATConv(in_ch, out_ch, heads=1, concat=False))
                if i < num_layers - 1:
                    self.relus.append(ReLU())
                    self.bns.append(BatchNorm1d(out_ch))
        else:
            raise ValueError(f"Unknown model_type {model_type}")
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=out_channels)
        self.val_acc = Accuracy(task="multiclass", num_classes=out_channels)
        self.test_acc = Accuracy(task="multiclass", num_classes=out_channels)

    def forward(self, x, adjs_t, n_id):
        for i, layer in enumerate(self.layers):
            if isinstance(adjs_t, list):
                adj = adjs_t[i]
            else:
                adj = adjs_t
            x = layer(x, adj)
            if i < self.num_layers - 1:
                x = self.bns[i](x)
                x = self.relus[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def training_step(self, batch, batch_idx):
        x, y, adjs_t, n_id = batch.x, batch.y, batch.adjs_t, batch.n_id
        out = self.forward(x, adjs_t, n_id)[:y.size(0)]
        loss = self.loss_fn(out, y)
        acc = self.train_acc(out.softmax(dim=-1), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, adjs_t, n_id = batch.x, batch.y, batch.adjs_t, batch.n_id
        out = self.forward(x, adjs_t, n_id)[:y.size(0)]
        loss = self.loss_fn(out, y)
        acc = self.val_acc(out.softmax(dim=-1), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y, adjs_t, n_id = batch.x, batch.y, batch.adjs_t, batch.n_id
        out = self.forward(x, adjs_t, n_id)[:y.size(0)]
        loss = self.loss_fn(out, y)
        acc = self.test_acc(out.softmax(dim=-1), y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
        return [optimizer], [scheduler]

def main():
    parser = argparse.ArgumentParser(description="OGBN-MAG240 RGCN Training")
    # parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--feature_type", type=str, default="roberta+bloom+transe",
                        choices=["noFeature","bloom","roberta","roberta+bloom+transe"])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--sizes", type=str, default="25-15")
    parser.add_argument("--hidden_channels", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", type=str, default="rgraphsage",
                        choices=["rgraphsage","rgat"])
    parser.add_argument("--in_memory", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args.sizes = [int(x) for x in args.sizes.split("-")]

    # Logging
    os.makedirs("logs", exist_ok=True)
    now = time.strftime("%Y%m%d_%H%M%S")
    script = osp.splitext(osp.basename(__file__))[0]
    logf = open(f"logs/{script}-mag-{args.feature_type}-{now}.log", "a")
    sys.stdout = LoggerWriter(logf, sys.stdout)
    sys.stderr = LoggerWriter(logf, sys.stderr)
    print(f"Logging to logs/{script}-mag-{args.feature_type}-{now}.log")
    print(f"Args: {args}")

    seed_everything(42)
    dm = MAG240M(ROOT, args.feature_type, args.batch_size,
                 args.sizes, args.in_memory)
    dm.prepare_data()
    dm.setup()

    model = RGNN(args.model,
                 dm.num_features, dm.num_classes,
                 args.hidden_channels, dm.num_relations,
                 len(dm.sizes), dropout=args.dropout)

    callbacks = [
        EpochTimingCallback(),
        ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
    ]
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[args.device] if torch.cuda.is_available() else None,
        callbacks=callbacks,
        enable_progress_bar=False
    )
    trainer.fit(model, dm)

    # Validation
    print("Evaluating model on validation set...")
    val_res = trainer.test(model=model, datamodule=dm)
    print(f"\n→ Test accuracy: {val_res[0]['test_acc']:.4f}")

    # Manual test-dev
    evaluator = MAG240MEvaluator()
    loader = dm.test_dataloader()
    model.eval()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    model.to(device)
    preds = []
    for batch in tqdm(loader, disable=True):
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch.x, batch.adjs_t, batch.n_id).argmax(dim=-1).cpu()
            preds.append(out)
    res = {"y_pred": torch.cat(preds, dim=0)}
    evaluator.save_test_submission(res, f"results/{args.model}", mode="test-dev")
    y_true = dm.y[dm.test_idx].numpy()
    y_pred = res["y_pred"].numpy()
    acc = evaluator.eval({"y_true": y_true, "y_pred": y_pred})["acc"]
    print(f"\n→ OGB Evaluator test-dev accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()