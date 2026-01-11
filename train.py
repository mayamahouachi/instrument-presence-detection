# train.py
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.data import SegmentDataset, list_npz, compute_train_stats
from src.model import InstrumentCNN
from src.evaluate import evaluate
from src.utils import set_seed, get_device


def train_one_epoch(model, loader, device, optimizer, desc="Train"):
    bce = nn.BCEWithLogitsLoss()
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc=desc, leave=False)
    for X, Y in pbar:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True).float()
        logits = model(X)
        loss = bce(logits, Y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item() * X.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="decompressed_musdb18/train")
    p.add_argument("--stems", type=str, default="vocals,drums,bass")

    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--normalize", action="store_true")
    p.add_argument("--stats-max-segments", type=int, default=2000)

    p.add_argument("--out", type=str, default="runs/cnn_baseline")
    args = p.parse_args()

    stems = [s.strip() for s in args.stems.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    files = list_npz(args.data_dir)
    train_files, val_files = train_test_split(files, test_size=args.val_split, random_state=args.seed)
    print(f"Tracks: {len(files)} | Train: {len(train_files)} | Val: {len(val_files)}")

    stats = None
    if args.normalize:
        print("Computing train normalization stats...")
        mean, std = compute_train_stats(train_files, max_segments_per_track=args.stats_max_segments, seed=args.seed)
        stats = {"mean": mean, "std": std}
        torch.save({"mean": mean, "std": std}, out_dir / "train_stats.pt")
        print("Saved stats to:", out_dir / "train_stats.pt")

    train_ds = SegmentDataset(train_files, normalize=args.normalize, stats=stats)
    val_ds = SegmentDataset(val_files, normalize=args.normalize, stats=stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True,persistent_workers=True)

    device = get_device()
    print("Device:", device)

    model = InstrumentCNN(n_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, desc=f"Train epoch {epoch}")
        val_loss, val_metrics = evaluate(model, val_loader, device, threshold=args.threshold, desc=f"Val epoch {epoch}")

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val   loss: {val_loss:.4f}")
        print(f"  Val macro F1: {val_metrics['macro_f1']:.3f} | "
              f"P: {val_metrics['macro_precision']:.3f} | R: {val_metrics['macro_recall']:.3f}")

        for i, name in enumerate(stems):
            print(f"   - {name:<7} F1={val_metrics['f1'][i]:.3f} "
                  f"P={val_metrics['precision'][i]:.3f} R={val_metrics['recall'][i]:.3f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stems": stems,
                    "threshold": args.threshold,
                    "normalize": args.normalize,
                },
                best_path )
            print(f"Saved best model: {best_path} (macro F1={best_f1:.3f})")

    print("Done : Best macro F1 =", best_f1)
    print("Best model:", best_path)


if __name__ == "__main__":
    main()
