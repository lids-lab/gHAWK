from ogb.linkproppred import Evaluator  # for OGB-based evaluation
import torch
from logger import get_logger
import numpy as np
logger = get_logger(__name__)


############################################
#           OGB EVALUATION UTILS
############################################
def evaluate_ogb(model, triple_dict, device, batch_size=100):
    logger.info("Starting OGB evaluation ...")
    evaluator = Evaluator(name='ogbl-wikikg2')
    model.eval()
    heads = torch.as_tensor(triple_dict['head'], device=device)
    rels = torch.as_tensor(triple_dict['relation'], device=device)
    tails = torch.as_tensor(triple_dict['tail'], device=device)
    head_neg = torch.as_tensor(triple_dict['head_neg'], device=device)
    tail_neg = torch.as_tensor(triple_dict['tail_neg'], device=device)
    N = heads.size(0)
    y_pred_pos_head = []
    y_pred_neg_head = []
    y_pred_pos_tail = []
    y_pred_neg_tail = []
    # Head corruption
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bh = heads[start:end]
        br = rels[start:end]
        bt = tails[start:end]
        pos_scores = model(bh, br, bt)
        neg_list = []
        for i in range(bh.size(0)):
            neg_heads = head_neg[start + i].to(device)
            r_rep = br[i].repeat(neg_heads.size(0))
            t_rep = bt[i].repeat(neg_heads.size(0))
            score_neg = model(neg_heads, r_rep, t_rep)
            neg_list.append(score_neg.unsqueeze(0))
        neg_scores = torch.cat(neg_list, dim=0)
        y_pred_pos_head.append(pos_scores.detach().cpu().numpy())
        y_pred_neg_head.append(neg_scores.detach().cpu().numpy())
    # Tail corruption
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bh = heads[start:end]
        br = rels[start:end]
        bt = tails[start:end]
        pos_scores = model(bh, br, bt)
        neg_list = []
        for i in range(bh.size(0)):
            neg_tails = tail_neg[start + i].to(device)
            h_rep = bh[i].repeat(neg_tails.size(0))
            r_rep = br[i].repeat(neg_tails.size(0))
            score_neg = model(h_rep, r_rep, neg_tails)
            neg_list.append(score_neg.unsqueeze(0))
        neg_scores = torch.cat(neg_list, dim=0)
        y_pred_pos_tail.append(pos_scores.detach().cpu().numpy())
        y_pred_neg_tail.append(neg_scores.detach().cpu().numpy())
    y_pred_pos_head = np.concatenate(y_pred_pos_head, axis=0)
    y_pred_neg_head = np.concatenate(y_pred_neg_head, axis=0)
    y_pred_pos_tail = np.concatenate(y_pred_pos_tail, axis=0)
    y_pred_neg_tail = np.concatenate(y_pred_neg_tail, axis=0)
    y_pred_pos_head = torch.from_numpy(y_pred_pos_head)
    y_pred_neg_head = torch.from_numpy(y_pred_neg_head)
    y_pred_pos_tail = torch.from_numpy(y_pred_pos_tail)
    y_pred_neg_tail = torch.from_numpy(y_pred_neg_tail)
    result_head = evaluator.eval({'y_pred_pos': y_pred_pos_head, 'y_pred_neg': y_pred_neg_head})
    result_tail = evaluator.eval({'y_pred_pos': y_pred_pos_tail, 'y_pred_neg': y_pred_neg_tail})
    final_mrr = (result_head['mrr_list'].mean().item() + result_tail['mrr_list'].mean().item()) / 2.0
    final_h3 = (result_head['hits@3_list'].mean().item() + result_tail['hits@3_list'].mean().item()) / 2.0
    final_h10 = (result_head['hits@10_list'].mean().item() + result_tail['hits@10_list'].mean().item()) / 2.0
    logger.info(f"Evaluation done: MRR={final_mrr:.4f}, H@3={final_h3:.4f}, H@10={final_h10:.4f}")
    return final_mrr, final_h3, final_h10

def run_evaluation(model: torch.nn.Module, args, split: str = 'valid') -> tuple[float, float, float]:
    """
    Load a split (valid or test), run OGB evaluation, and log results.
    split must be 'valid' or 'test'.
    """
    assert split in ('valid', 'test'), "split must be 'valid' or 'test'"
    logger.info(f"Running evaluation on {split} split")

    pt_path = args.valid_pt if split == 'valid' else args.test_pt
    logger.info(f"Loading {split} data from {pt_path}")
    data = torch.load(pt_path)
    triple_dict = {
        'head':     data['head'],
        'relation': data['relation'],
        'tail':     data['tail'],
        'head_neg': data['head_neg'],
        'tail_neg': data['tail_neg'],
    }

    batch_size = getattr(args, f"{split}_batch_size", args.batch_size)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    mrr, h3, h10 = evaluate_ogb(model, triple_dict, device, batch_size=batch_size)
    logger.info(f"[{split.capitalize()}] MRR={mrr:.4f}, Hits@3={h3:.4f}, Hits@10={h10:.4f}")
    return mrr, h3, h10
