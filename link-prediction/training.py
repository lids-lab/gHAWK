import torch
import os
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from evaluation import run_evaluation
from logger import get_logger
from dataloader import load_data
from models.hybrid_model import HybridLinkPredictorWithGNN_PyG, build_global_edge_index
from evaluation import evaluate_ogb
from scripts.generate_bloom import generate_bloom

import numpy as np
logger = get_logger(__name__)

def train_one_epoch(model, train_loader, optimizer, num_entities, device, margin=1.0):
    model.train()
    total_loss = 0.0
    for batch_heads, batch_rels, batch_tails in train_loader:
        batch_heads = batch_heads.to(device)
        batch_rels = batch_rels.to(device)
        batch_tails = batch_tails.to(device)
        pos_score = model(batch_heads, batch_rels, batch_tails)
        neg_heads, neg_rels, neg_tails = negative_sampling(batch_heads, batch_rels, batch_tails, num_entities, device)
        neg_score = model(neg_heads, neg_rels, neg_tails)
        loss = margin_ranking_loss(pos_score, neg_score, margin=margin)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Finished epoch. Avg loss = {avg_loss:.4f}")
    return avg_loss


def negative_sampling(heads, rels, tails, num_entities, device):
    batch_size = heads.size(0)
    rand_ent = torch.randint(0, num_entities, (batch_size,), device=device)
    mask = torch.rand(batch_size, device=device) < 0.5
    neg_heads = torch.where(mask, rand_ent, heads)
    neg_tails = torch.where(~mask, rand_ent, tails)
    return neg_heads, rels, neg_tails


def margin_ranking_loss(pos_score, neg_score, margin=1.0):
    loss = F.relu(margin + neg_score - pos_score)
    return loss.mean()


def run_training(args):
    """
    Top-level training loop: loads data, builds model, runs epochs, evaluates and checkpoints.
    Expects args to have attributes:
      - device, lr, weight_decay, epochs, margin
      - bloom and TransE paths, data splits (train_pt, valid_pt, test_pt)
      - batch_size, eval_every, checkpoint_dir
    """
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data & loader
    data = load_data(args)
    train_loader      = data['train_loader']
    num_entities      = data['num_nodes']



    # Build model
    model = HybridLinkPredictorWithGNN_PyG(
        bloom_emb=data['bloom_emb'],
        transE_emb=data['transE_emb'],
        num_relations=args.num_relations,
        global_edge_index=data['global_edge_index'],
        bloom_proj_dim=args.proj_dim_bloom,
        transE_proj_dim=args.proj_dim_transe,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_out_dim=args.gnn_out_dim,
        conv_type=args.conv_type,
        num_bases=getattr(args, 'num_bases', None),
        pre_trained_rel_emb=data['transE_rel_emb'],
    ).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, 'weight_decay', 0))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training epochs

    # Training epochs
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, num_entities, device, margin=args.margin
        )
        logger.info(f"[Epoch {epoch}/{args.epochs}] Loss = {avg_loss:.4f}")

        # Validation every eval_every epochs
        if epoch % args.eval_every == 0:
            run_evaluation(model, args, split='valid')

        # Checkpoint
        ckpt_path = os.path.join(
            args.checkpoint_dir, f"epoch_{epoch}.pt"
        )
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

    # Final test evaluation
    run_evaluation(model, args, split='test')





