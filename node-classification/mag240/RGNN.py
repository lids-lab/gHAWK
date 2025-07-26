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
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
WITHOUT_LIGHTNING_V2 = int(pytorch_lightning.__version__.split('.')[0]) < 2
from torchmetrics import Accuracy
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.nn import Parameter
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GATConv, SAGEConv
from torch_sparse import SparseTensor

from tqdm.auto import tqdm
import sys
from root import ROOT
LOG_DIR = os.path.join('logs', 'rgnn-features3')
# ——— Set up logging to file ———
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, 'rgnn-features3.log')
sys.stdout = open(log_path, 'w')
sys.stderr = sys.stdout

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


# ————————————————————————
# ===============================
# Callback to log one line per epoch with timing
# ===============================
class EpochTimingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Record start time of the epoch
        self.epoch_start_time = time.perf_counter()
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Compute epoch duration
        epoch_time = time.perf_counter() - self.epoch_start_time
        current_epoch = trainer.current_epoch
        train_acc = trainer.callback_metrics.get('train_acc', None)
        val_acc = trainer.callback_metrics.get('val_acc', None)
        # Print a single summary line per epoch
        print(f"Epoch {current_epoch:2d}: time {epoch_time:.2f}s, "
              f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
        
class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]
    n_id: Tensor

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
            n_id = self.n_id.to(*args, **kwargs),
        )


def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)


def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk)):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]


class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int],
                 in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory
        # Load metadata for node counts
        dataset_tmp = MAG240MDataset(self.data_dir)
        self.num_papers = dataset_tmp.num_papers
        self.num_authors = dataset_tmp.num_authors
        self.num_institutions = dataset_tmp.num_institutions

    @property
    def num_features(self) -> int:
        return 768 + 63 + 64

    @property
    def num_classes(self) -> int:
        return 153

    @property
    def num_relations(self) -> int:
        return 5

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):  # Will take approximately 5 minutes...
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_adj_t.pt'
        if not osp.exists(path):  # Will take approximately 16 minutes...
            t = time.perf_counter()
            print('Merging adjacency matrices...', end=' ', flush=True)

            row, col, _ = torch.load(
                f'{dataset.dir}/paper_to_paper_symmetric.pt').coo()
            rows, cols = [row], [col]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            rows += [row, col]
            cols += [col, row]

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors
            rows += [row, col]
            cols += [col, row]

            edge_types = [
                torch.full(x.size(), i, dtype=torch.int8)
                for i, x in enumerate(rows)
            ]

            row = torch.cat(rows, dim=0)
            del rows
            col = torch.cat(cols, dim=0)
            del cols

            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            perm = (N * row).add_(col).numpy().argsort()
            perm = torch.from_numpy(perm)
            row = row[perm]
            col = col[perm]

            edge_type = torch.cat(edge_types, dim=0)[perm]
            del edge_types

            full_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                                      sparse_sizes=(N, N), is_sorted=True)

            torch.save(full_adj_t, path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_feat.npy'
        done_flag_path = f'{dataset.dir}/full_feat_done.txt'
        if not osp.exists(done_flag_path):  # Will take ~3 hours...
            t = time.perf_counter()
            print('Generating full feature matrix...')

            node_chunk_size = 100000
            dim_chunk_size = 64
            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            paper_feat = dataset.paper_feat
            x = np.memmap(path, dtype=np.float16, mode='w+',
                          shape=(N, self.num_features))

            print('Copying paper features...')
            for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):
                j = min(i + node_chunk_size, dataset.num_papers)
                x[i:j] = paper_feat[i:j]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=row, col=col,
                sparse_sizes=(dataset.num_authors, dataset.num_papers),
                is_sorted=True)

            # Processing 64-dim subfeatures at a time for memory efficiency.
            print('Generating author features...')
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(paper_feat, start_row_idx=0,
                                       end_row_idx=dataset.num_papers,
                                       start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                del outputs

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=col, col=row,
                sparse_sizes=(dataset.num_institutions, dataset.num_authors),
                is_sorted=False)

            print('Generating institution features...')
            # Processing 64-dim subfeatures at a time for memory efficiency.
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(
                    x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x,
                    start_row_idx=dataset.num_papers + dataset.num_authors,
                    end_row_idx=N, start_col_idx=i, end_col_idx=j)
                del outputs

            x.flush()
            del x
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            with open(done_flag_path, 'w') as f:
                f.write('done')

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        # Store train/val/test indices
        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
        self.test_idx.share_memory_()

        # Keep track of node counts
        self.num_papers = dataset.num_papers
        self.num_authors = dataset.num_authors
        self.num_institutions = dataset.num_institutions


        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        # === Load just the 768‐dim RoBERTa memmap ===
        feat_path = osp.join(dataset.dir, 'full_feat.npy')
        if self.in_memory:
            roberta_full = np.load(feat_path)                                 # shape (N,768)
            self.full_feat = torch.from_numpy(roberta_full).share_memory_()    # tensor (N,768)
        else:
            # Force memmap to exactly 768 cols, not 931
            self.full_feat = np.memmap(feat_path, dtype=np.float16, mode='r',
                                    shape=(N, 768))                           # memmap (N,768)


        # Load Bloom filter bytes for each node type
        # Paper
        bloom_paper_path = osp.join(osp.dirname(self.data_dir), 'Bloom', 'bloom_filters.npy')
        self.bloom_paper = np.memmap(bloom_paper_path, dtype='uint8', mode='r',
                                     shape=(self.num_papers, 63))
        # Author
        bloom_author_path = osp.join(osp.dirname(self.data_dir), 'Bloom', 'bloom_author.npy')
        self.bloom_author = np.memmap(bloom_author_path, dtype='uint8', mode='r',
                                      shape=(self.num_authors, 63))
        # Institution
        bloom_inst_path = osp.join(osp.dirname(self.data_dir), 'Bloom', 'bloom_institution.npy')
        self.bloom_inst = np.memmap(bloom_inst_path, dtype='uint8', mode='r',
                                    shape=(self.num_institutions, 63))

        # Load TransE embeddings for all nodes
        all_transe_path = osp.join(osp.dirname(self.data_dir), 'TransE', 'all_transe.npy')
        if self.in_memory:
            transe_full = np.load(all_transe_path)
            self.full_transe = torch.from_numpy(transe_full).share_memory_()
        else:
            self.full_transe = np.load(all_transe_path, mmap_mode='r')

        # Load labels and adjacency
        self.y = torch.from_numpy(dataset.all_paper_label)
        path = f'{dataset.dir}/full_adj_t.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=4)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def hidden_test_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=3)

    def convert_batch(self, batch_size, n_id, adjs):
        idx_arr = n_id.cpu().numpy()
        B = len(idx_arr)

        # (1) RoBERTa ← first 768 dims
        if self.in_memory:
            rob = self.full_feat[n_id].float()                   # [B,768]
        else:
            rob = torch.from_numpy(self.full_feat[idx_arr].astype(np.float32))

        # (2) Bloom & (3) TransE: exactly as before
        bloom = torch.zeros(B, 63, dtype=torch.float)
        mask_p = idx_arr < self.num_papers
        mask_a = (idx_arr >= self.num_papers) & (idx_arr < self.num_papers + self.num_authors)
        mask_i = ~mask_p & ~mask_a

        if mask_p.any():
            paper_idx = idx_arr[mask_p]
            bf_p = self.bloom_paper[paper_idx]                  # (papers_in_batch,63)
            bloom_vals = torch.from_numpy(bf_p.astype(np.float32))
            bloom[mask_p] = bloom_vals
        if mask_a.any():
            author_idx = idx_arr[mask_a] - self.num_papers
            bf_a = self.bloom_author[author_idx]
            bloom_vals = torch.from_numpy(bf_a.astype(np.float32))
            bloom[mask_a] = bloom_vals
        if mask_i.any():
            inst_idx = idx_arr[mask_i] - self.num_papers - self.num_authors
            bf_i = self.bloom_inst[inst_idx]
            bloom_vals = torch.from_numpy(bf_i.astype(np.float32))
            bloom[mask_i] = bloom_vals

        if self.in_memory:
            tr = self.full_transe[n_id].float()                  # [B,64]
        else:
            tr = torch.from_numpy(self.full_transe[idx_arr].astype(np.float32))

        # (4) Concatenate → [B,931]
        x = torch.cat([rob, bloom, tr], dim=1)
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs], n_id=n_id)


class RGNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_relations: int, num_layers: int,
                 num_papers: int,
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout
        # Per-modality projection and normalization
        # RoBERTa: 768 → 384
        self.rob_proj_ml = Sequential(
            Linear(768, 384),
            BatchNorm1d(384),
            ReLU(inplace=True),
        )
        # Bloom: 63 → 63 (just normalize)
        self.bloom_norm = BatchNorm1d(63)
        # TransE: 64 → 53
        self.transe_proj_ml = Sequential(
            Linear(64, 53),
            BatchNorm1d(53),
            ReLU(inplace=True),
        )
        # New fused input dimension
        fused_in = 384 + 63 + 53  # = 500

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'rgat':
            self.convs.append(
                ModuleList([
                    GATConv(fused_in, hidden_channels // heads, heads,
                            add_self_loops=False) for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        GATConv(hidden_channels, hidden_channels // heads,
                                heads, add_self_loops=False)
                        for _ in range(num_relations)
                    ]))

        elif self.model == 'rgraphsage':
            self.convs.append(
                ModuleList([
                    SAGEConv(fused_in, hidden_channels, root_weight=False)
                    for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        SAGEConv(hidden_channels, hidden_channels,
                                 root_weight=False)
                        for _ in range(num_relations)
                    ]))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.skips.append(Linear(fused_in, hidden_channels))
        for _ in range(num_layers - 1):
            self.skips.append(Linear(hidden_channels, hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.train_acc = Accuracy(task="multiclass", num_classes=out_channels)
        self.val_acc   = Accuracy(task="multiclass", num_classes=out_channels)
        self.test_acc  = Accuracy(task="multiclass", num_classes=out_channels)


    def forward(self, x: Tensor, adjs_t: List[SparseTensor], n_id: Tensor) -> Tensor:
        # Split raw features into modalities
        rob = x[:, :768]
        bloom = x[:, 768:768+63]
        transe = x[:, 768+63:]
        # Project and normalize each modality
        rob = self.rob_proj_ml(rob)          # → [B,384]
        bloom = self.bloom_norm(bloom)       # → [B,63]
        transe = self.transe_proj_ml(transe) # → [B,53]
        # Concatenate fused features
        x = torch.cat([rob, bloom, transe], dim=1)  # → [B,500]

        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]

            out = self.skips[i](x_target)
            for j in range(self.num_relations):
                val = adj_t.storage.value()
                # Skip if no edge_type values are stored
                if val is None:
                    continue
                mask = val == j
                # mask must be a tensor for masked_select_nnz
                subadj_t = adj_t.masked_select_nnz(mask, layout='coo')
                subadj_t = subadj_t.set_value(None, layout=None)
                if subadj_t.nnz() > 0:
                    out += self.convs[i][j]((x, x_target), subadj_t)

            x = self.norms[i](out)
            x = F.elu(x) if self.model == 'rgat' else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t, batch.n_id)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t, batch.n_id)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t, batch.n_id)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--in-memory', action='store_true')
    parser.add_argument('--device', type=str, default='1')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    # Parse GPU device IDs
    args.device_ids = [int(x) for x in args.device.split(',')]
    print(args)

    seed_everything(42)
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.in_memory)

    model = RGNN(
        args.model,
        datamodule.num_features,
        datamodule.num_classes,
        args.hidden_channels,
        datamodule.num_relations,
        len(args.sizes),
        datamodule.num_papers,
        dropout=args.dropout
    )
    print(f'#Params {sum([p.numel() for p in model.parameters()])}')
    print(model)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',
                                          save_top_k=1)
    if WITHOUT_LIGHTNING_V2:
        trainer = Trainer(accelerator='gpu', gpus=args.device_ids, max_epochs=args.epochs,
                          callbacks=[checkpoint_callback, EpochTimingCallback()],
                          default_root_dir=LOG_DIR, 
                          progress_bar_refresh_rate=0)
    else:
        trainer = Trainer(accelerator='gpu', devices=args.device_ids, max_epochs=args.epochs,
                          callbacks=[checkpoint_callback, EpochTimingCallback()],
                          enable_progress_bar=False,
                          default_root_dir=LOG_DIR)
    trainer.fit(model, datamodule=datamodule)

    # === Begin evaluation after training ===
    # Test with Lightning
    print("Evaluating model on validation set...")
    results = trainer.test(model=model, datamodule=datamodule)
    test_acc = results[0]['test_acc']
    print(f"\n→ Final test accuracy: {test_acc:.5f}")

    # Manual test-dev predictions
    evaluator = MAG240MEvaluator()
    loader = datamodule.hidden_test_dataloader()

    model.eval()
    device = f'cuda:{args.device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    y_preds = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch.x, batch.adjs_t, batch.n_id).argmax(dim=-1).cpu()
            y_preds.append(out)
    res = {'y_pred': torch.cat(y_preds, dim=0)}
    evaluator.save_test_submission(res, f'results/{args.model}',
                                   mode='test-dev')

    # Compute and print test-dev accuracy
    y_pred = res['y_pred'].numpy() if isinstance(res['y_pred'], torch.Tensor) else res['y_pred']
    # True labels for test-dev nodes
    y_true = datamodule.y[datamodule.test_idx].numpy()
    acc = evaluator.eval({'y_pred': y_pred, 'y_true': y_true})['acc']
    print(f"\n→ OGB Evaluator test-dev accuracy: {acc:.4f}")
    # === End evaluation ===