import os
import sys
import time
import glob
import argparse
import os.path as osp
import numpy as np
from tqdm.auto import tqdm
import datetime
from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT

# ————————————————————————
# LoggerWriter for redirecting stdout/stderr to a file and console
class LoggerWriter:
    def __init__(self, file):
        self.terminal = sys.__stdout__
        self.file = file
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
    def flush(self):
        self.terminal.flush()
        self.file.flush()


class EpochTimeCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._start_time = time.time()
    def on_train_epoch_end(self, trainer, pl_module):
        duration = time.time() - self._start_time
        epoch = trainer.current_epoch
        print(f"Epoch {epoch} duration: {duration:.2f}s")

WITHOUT_LIGHTNING_V2 = int(pytorch_lightning.__version__.split('.')[0]) < 2
print(f'Using PyTorch Lightning v{pytorch_lightning.__version__} '
      f'→ {"without" if WITHOUT_LIGHTNING_V2 else "with"} v2 features.')

class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )


class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, feature_type: str, batch_size: int, sizes: List[int], in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.feature_type = feature_type.lower()
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory

    @property
    def num_features(self) -> int:
        # Raw dims: RoBERTa (768) + Bloom (63) + TransE (100)
        return 768 + 63 + 100

    @property
    def num_classes(self) -> int:
        return 153

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)
        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):
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

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx = self.train_idx
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
        self.test_idx.share_memory_()

        # feature loading based on feature type
        num_papers = dataset.num_papers
        bloom_path = osp.join(osp.dirname(self.data_dir), 'Bloom', 'bloom_filters.npy')


        # Load TransE embeddings for structural positions
        transe_path = osp.join(osp.dirname(self.data_dir), 'TransE', 'paper_transe.npy')
        
        roberta_exists = hasattr(dataset, 'all_paper_feat') or hasattr(dataset, 'paper_feat')
        bloom_exists = os.path.exists(bloom_path)
        transe_exists = os.path.exists(transe_path)
        self.paper_feat = None
        self.bf_bytes = None
        self.transe = None

        ft = self.feature_type
        if ft == 'nofeature':
            # No features loaded
            self.paper_feat = None
            self.bf_bytes = None
            self.transe = None
        elif ft == 'bloom':
            if not bloom_exists:
                raise FileNotFoundError(f"Bloom filter file not found at {bloom_path}")
            self.bf_bytes = np.memmap(bloom_path, dtype='uint8', mode='r', shape=(num_papers, 63))
        elif ft == 'roberta':
            if not roberta_exists:
                raise FileNotFoundError("RoBERTa features not found in dataset.")
            if self.in_memory:
                self.paper_feat = torch.from_numpy(dataset.all_paper_feat).share_memory_()
            else:
                self.paper_feat = dataset.paper_feat
        elif ft == 'roberta+bloom+transe':
            if not roberta_exists:
                raise FileNotFoundError("RoBERTa features not found in dataset.")
            if not bloom_exists:
                raise FileNotFoundError(f"Bloom filter file not found at {bloom_path}")
            if not transe_exists:
                raise FileNotFoundError(f"TransE file not found at {transe_path}")
            if self.in_memory:
                self.paper_feat = torch.from_numpy(dataset.all_paper_feat).share_memory_()
            else:
                self.paper_feat = dataset.paper_feat
            self.bf_bytes = np.memmap(bloom_path, dtype='uint8', mode='r', shape=(num_papers, 63))
            if self.in_memory:
                transe_full = np.load(transe_path)
                self.transe = torch.from_numpy(transe_full).share_memory_()
            else:
                self.transe = np.load(transe_path, mmap_mode='r')
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        self.y = torch.from_numpy(dataset.all_paper_label)

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
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
        ft = self.feature_type
        # RoBERTa
        if ft in ('roberta', 'roberta+bloom+transe'):
            if self.in_memory:
                rob_tensor = self.paper_feat[n_id].to(torch.float)
            else:
                rob_tensor = torch.from_numpy(self.paper_feat[idx_arr]).float()
        else:
            rob_tensor = torch.zeros((n_id.shape[0], 768), dtype=torch.float32)
        # Bloom
        if ft in ('bloom', 'roberta+bloom+transe'):
            bf_bytes_batch = self.bf_bytes[idx_arr]  # shape [batch, 63]
            bloom_tensor = torch.from_numpy(bf_bytes_batch.astype(np.float32)).float()
        else:
            bloom_tensor = torch.zeros((n_id.shape[0], 63), dtype=torch.float32)
        # TransE
        if ft == 'roberta+bloom+transe':
            if self.in_memory:
                transe_tensor = self.transe[n_id].to(torch.float)
            else:
                transe_tensor = torch.from_numpy(self.transe[idx_arr]).float()
        else:
            transe_tensor = torch.zeros((n_id.shape[0], 100), dtype=torch.float32)
        # Concatenate raw features
        x = torch.cat([rob_tensor, bloom_tensor, transe_tensor], dim=1)
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])


class GNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_layers: int, heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.dropout = dropout

        # Modality projection MLPs
        self.rob_proj_ml = Sequential(
            Linear(768, 384),
            BatchNorm1d(384),
            ReLU(inplace=True),
        )
        self.bloom_proj_ml = Sequential(
            Linear(63, 116),
            BatchNorm1d(116),
            ReLU(inplace=True),
        )
        # Normalize TransE embeddings only, do not project
        self.transe_norm = BatchNorm1d(100)
        # Fused input dimension: 384 + 116 + 100 = 600
        fused_in = 384 + 116 + 100

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'gat':
            self.convs.append(
                GATConv(fused_in, hidden_channels // heads, heads))
            self.skips.append(Linear(fused_in, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels // heads, heads))
                self.skips.append(Linear(hidden_channels, hidden_channels))

        elif self.model == 'graphsage':
            self.convs.append(
                SAGEConv(fused_in, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

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

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        # Split raw features
        rob = x[:, :768]
        bloom = x[:, 768:768+63]
        transe = x[:, 768+63:]
        # Project and normalize each modality
        rob = self.rob_proj_ml(rob)          # [batch,384]
        bloom = self.bloom_proj_ml(bloom)    # [batch,116]
        transe = self.transe_norm(transe)    # [batch,100]
        # Concatenate fused features
        x = torch.cat([rob, bloom, transe], dim=1)  # [batch,600]
        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]
            x = self.convs[i]((x, x_target), adj_t)
            if self.model == 'gat':
                x = x + self.skips[i](x_target)
                x = F.elu(self.norms[i](x))
            elif self.model == 'graphsage':
                x = F.relu(self.norms[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default='gat',
                        choices=['gat', 'graphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--in-memory', action='store_true')
    parser.add_argument('--device', type=str, default='3')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--feature_type', type=str, default='roberta+bloom+transe',
                        choices=['noFeature','bloom','roberta','roberta+bloom+transe'])
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    # Parse GPU device IDs from args.device
    args.device_ids = [int(x) for x in args.device.split(',')]
    print(args)

    # Logging setup
    LOG_DIR = os.path.join('logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    scriptname = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_fname = f"{scriptname}_{args.feature_type}_{timestamp}.log"
    log_path = os.path.join(LOG_DIR, log_fname)
    log_file = open(log_path, 'w')
    sys.stdout = LoggerWriter(log_file)
    sys.stderr = sys.stdout

    seed_everything(42)
    datamodule = MAG240M(ROOT, args.feature_type, args.batch_size, args.sizes, args.in_memory)

    if not args.evaluate:
        model = GNN(args.model, datamodule.num_features,
                    datamodule.num_classes, args.hidden_channels,
                    num_layers=len(args.sizes), dropout=args.dropout)
        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        print(model)
        print(f'Using {args.model.upper()} model with {args.hidden_channels} hidden channels, '
              f'{len(args.sizes)} layers, and dropout={args.dropout:.2f}.')
        print(f'Using {args.sizes} neighbor sampling sizes.')
        print(f'Using {args.batch_size} batch size.')
        print(f'Using {args.device} device.')
        print(f'Using {"in-memory" if args.in_memory else "on-disk"} data loading.')
        print(f'Using {torch.cuda.device_count()} GPUs: {args.device_ids}')
        print(f'Using GPUs: {args.device_ids}')
        print(f'Using device name: {torch.cuda.get_device_name(args.device_ids[0])} GPU.')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode = 'max', save_top_k=1)
        if WITHOUT_LIGHTNING_V2:
          trainer = Trainer(accelerator='gpu', gpus=args.device_ids, max_epochs=args.epochs,
                            callbacks=[checkpoint_callback, EpochTimeCallback()],
                            default_root_dir=LOG_DIR,
                            progress_bar_refresh_rate=0)
          
        else:
          # up to date usage
          trainer = Trainer(accelerator='gpu', devices=args.device_ids, max_epochs=args.epochs,
                            callbacks=[checkpoint_callback, EpochTimeCallback()],
                            default_root_dir=LOG_DIR,
                            enable_progress_bar=False)
        trainer.fit(model, datamodule=datamodule)

        # Evaluate immediately after training
        print("Evaluating trained model...")
        t_lit_start = time.time()
        results = trainer.test(model=model, datamodule=datamodule)
        t_lit_end = time.time()
        print(f"Lightning test completed in {t_lit_end - t_lit_start:.2f}s")
        test_acc = results[0]['test_acc']
        print(f"\n→ Final test accuracy: {test_acc:.5f}")

        evaluator = MAG240MEvaluator()
        loader = datamodule.hidden_test_dataloader()
        model.eval()
        device = f'cuda:{args.device_ids[0]}' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        y_preds = []
        t_manual_start = time.time()
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        t_manual_end = time.time()
        print(f"Manual OGB evaluation loop completed in {t_manual_end - t_manual_start:.2f}s")
        y_pred = torch.cat(y_preds, dim=0).cpu().numpy()
        y_true = datamodule.y[datamodule.test_idx].cpu().numpy()
        acc = evaluator.eval({'y_pred': y_pred, 'y_true': y_true})['acc']
        print(f"\n→ OGB Evaluator test-dev accuracy: {acc:.5f}")

    if args.evaluate:
        dirs = glob.glob(os.path.join(LOG_DIR, 'lightning_logs', '*'))
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = os.path.join(LOG_DIR, 'lightning_logs', f'version_{version}')
        print(f'Evaluating saved model in {logdir}...')
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]
        print(dirs, version, logdir, ckpt)
        if WITHOUT_LIGHTNING_V2:
          trainer = Trainer(accelerator='gpu', gpus=args.device_ids, resume_from_checkpoint=ckpt, progress_bar_refresh_rate=0)
        else:
          trainer = Trainer(accelerator='gpu', devices=args.device_ids, enable_progress_bar=False)
        model = GNN.load_from_checkpoint(checkpoint_path=ckpt,
                                         hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16
        datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

        # … after loading your model …
        print("Evaluating saved model...")
        t_lit_start = time.time()
        results = trainer.test(model=model, datamodule=datamodule)
        t_lit_end = time.time()
        print(f"Lightning test completed in {t_lit_end - t_lit_start:.2f}s")
        # Lightning returns something like: [{'test_acc': 0.7123}]
        test_acc = results[0]['test_acc']
        print(f"\n→ Final test accuracy: {test_acc:.5f}")

        evaluator = MAG240MEvaluator()
        loader = datamodule.hidden_test_dataloader()

        model.eval()
        device = f'cuda:{args.device_ids[0]}' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        y_preds = []
        t_manual_start = time.time()
        for batch in tqdm(loader):
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        t_manual_end = time.time()
        print(f"Manual OGB evaluation loop completed in {t_manual_end - t_manual_start:.2f}s")
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        evaluator.save_test_submission(res, os.path.join(LOG_DIR, 'results'), mode='test-dev')

        y_pred = torch.cat(y_preds, dim=0).cpu().numpy()
        # get the true labels for the test-dev nodes
        y_true = datamodule.y[datamodule.test_idx].cpu().numpy()

        acc = evaluator.eval({'y_pred': y_pred, 'y_true': y_true})['acc']
        print(f"\n→ OGB Evaluator test-dev accuracy: {acc:.9f}")