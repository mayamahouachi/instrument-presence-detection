# src/metrics.py
import numpy as np
import torch
from tqdm import tqdm

@torch.no_grad()
def metrics(logits: torch.Tensor,targets: torch.Tensor,threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold)
    t = (targets > 0.5)

    tp = (preds & t).sum(dim=0).float()
    fp = (preds & ~t).sum(dim=0).float()
    fn = ((~preds) & t).sum(dim=0).float()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy(),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
    }



@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, desc: str | None = None):
    import torch.nn as nn
    bce = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss = 0.0
    all_logits, all_targets = [], []

    it = loader
    if desc is not None:
        it = tqdm(loader, desc=desc, leave=False)

    for X, Y in it:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True).float()

        logits = model(X)
        loss = bce(logits, Y)

        total_loss += loss.item() * X.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(Y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metrics_result = metrics(logits, targets, threshold=threshold)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics_result