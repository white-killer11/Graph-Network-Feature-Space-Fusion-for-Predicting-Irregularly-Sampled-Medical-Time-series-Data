"""训练脚本 (论文 Model Training)

分类损失: BCE = -(1/N) sum [ y log p + (1-y) log(1-p) ]   (死亡 + 多表型)
回归损失: L1  = (1/N) sum |y - f(y)|                     (住院时长, min-max 归一化)

消融选项: --ablate point 仅点特征空间 / --ablate edge 仅边权重空间
可解释性: 保存各时间点注意力权重 w_fp (对应论文 interpretability analysis)
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import Config
from dataset import MedicalGraphDataset, SyntheticGenerator, collate_graphs
from graph_construction import GraphBuilder
from models import GNFFusionModel


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_bce(logits, targets, mask):
    """多标签 BCE, mask 标记缺失标签 (-100 位置忽略)"""
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def train_one_epoch(model, loader, opt, cfg, ablate, device):
    model.train()
    tot = 0.0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        cls_logits, reg_out, _ = model(batch['node_feat'], batch['adj'],
                                       batch['we'], batch['we_mask'],
                                       batch['n_time'], batch['last_idx'],
                                       ablate=None if ablate == 'none' else ablate)
        # 分类目标: [death, pheno...]
        targets = torch.cat([batch['death'].unsqueeze(1), batch['pheno']], dim=1)
        cls_loss = masked_bce(cls_logits, targets, torch.ones_like(targets))
        reg_loss = nn.functional.l1_loss(reg_out.squeeze(-1), batch['stay'])
        loss = cls_loss + reg_loss
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    return tot / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """返回 acc / AUC (死亡与表型均值) / MAE (住院时长)"""
    from sklearn.metrics import roc_auc_score, accuracy_score
    model.eval()
    all_logits, all_reg, all_t, all_stay, all_w = [], [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        cls_logits, reg_out, w_fp = model(batch['node_feat'], batch['adj'],
                                          batch['we'], batch['we_mask'],
                                          batch['n_time'], batch['last_idx'],
                                          ablate=getattr(model, '_ablate', None))
        all_logits.append(cls_logits.cpu()); all_reg.append(reg_out.cpu())
        all_t.append(torch.cat([batch['death'].unsqueeze(1), batch['pheno']], dim=1).cpu())
        all_stay.append(batch['stay'].cpu()); all_w.append(w_fp.cpu())
    logits = torch.cat(all_logits); reg = torch.cat(all_reg).squeeze(-1)
    targets = torch.cat(all_t); stay = torch.cat(all_stay)

    probs = torch.sigmoid(logits)
    accs, aucs = [], []
    for j in range(probs.shape[1]):
        y = targets[:, j]
        if y.sum() in (0, len(y)):            # 全一类 -> 指标无意义
            continue
        accs.append(accuracy_score(y, (probs[:, j] > 0.5).float()))
        aucs.append(roc_auc_score(y, probs[:, j]))
    mae = nn.functional.l1_loss(reg, stay).item()
    return dict(acc=float(np.mean(accs)), auc=float(np.mean(aucs)), mae=mae,
                time_weights=torch.cat(all_w).numpy().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--ablate', choices=['none', 'point', 'edge'], default='none')
    ap.add_argument('--synthetic', action='store_true', help='使用合成数据冒烟测试')
    args = ap.parse_args()

    cfg = Config()
    if args.epochs: cfg.epochs = args.epochs
    set_seed(cfg.seed)
    device = cfg.device

    # ---------------- 数据 ----------------
    gen = SyntheticGenerator(n_patients=400 if args.synthetic else 2000,
                             n_features=cfg.n_features, seed=cfg.seed)
    builder = GraphBuilder(n_features=cfg.n_features, n_kl_bins=cfg.n_kl_bins,
                           value_range=(-4.0, 8.0),
                           max_time_nodes=cfg.max_time_nodes)
    ds = MedicalGraphDataset(gen.records_list, gen.labels_list, builder,
                             pred_time=cfg.time_window_hours)
    n_val = max(1, int(0.15 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val])
    tr_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                           collate_fn=collate_graphs)
    va_loader = DataLoader(va, batch_size=cfg.batch_size, shuffle=False,
                           collate_fn=collate_graphs)

    # ---------------- 模型 ----------------
    sample = ds[0]
    node_dim = sample['node_feat'].shape[1]
    n_cls = 1 + cfg.n_phenotypes
    model = GNFFusionModel(cfg, node_dim, n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    # ---------------- 训练 ----------------
    best = 0.0
    for ep in range(cfg.epochs):
        loss = train_one_epoch(model, tr_loader, opt, cfg, args.ablate, device)
        metrics = evaluate(model, va_loader, device)
        score = metrics['auc']
        marker = ''
        if score > best:
            best = score
            torch.save(model.state_dict(), 'best_model.pt')
            json.dump(metrics, open('interpretability_weights.json', 'w'))
            marker = ' *'
        print(f"epoch {ep:03d} loss={loss:.4f} "
              f"acc={metrics['acc']:.3f} auc={metrics['auc']:.3f} "
              f"mae={metrics['mae']:.4f}{marker}")

    print(f"\nBest AUC={best:.4f}  "
          f"(消融: {args.ablate}; 时间点权重已存 interpretability_weights.json)")


if __name__ == '__main__':
    main()
