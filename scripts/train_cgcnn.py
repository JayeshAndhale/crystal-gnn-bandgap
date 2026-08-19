"""
Train CGCNN on the crystal graphs: 80/10/10 split, target normalization
(train-set stats only), multi-task MSE loss, best-checkpoint saving.

Run locally with a small --epochs value first (a few epochs) just to confirm
loss actually decreases and nothing crashes over a full epoch — the real,
many-epoch run happens on Kaggle GPU.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from crystal_gnn.models.cgcnn import CGCNN

#GRAPHS_DIR = Path("data/processed/graphs")
CHECKPOINT_DIR = Path("checkpoints")
SEED = 42


def load_all_graphs(graphs_dir):
    files = sorted(Path(graphs_dir).glob("*.pt"))
    return [torch.load(f, weights_only=False) for f in files]


def split_graphs(graphs, train_frac=0.8, val_frac=0.1):
    random.seed(SEED)
    shuffled = graphs.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def compute_norm_stats(train_graphs):
    """Mean/std for both targets, from TRAINING data only — never val/test,
    to avoid leaking information about held-out data into training."""
    fe = torch.tensor([g.y_formation_energy.item() for g in train_graphs])
    bg = torch.tensor([g.y_band_gap.item() for g in train_graphs])
    return {
        "fe_mean": fe.mean().item(), "fe_std": fe.std().item(),
        "bg_mean": bg.mean().item(), "bg_std": bg.std().item(),
    }


def normalize(value, mean, std):
    return (value - mean) / std


def run_epoch(model, loader, stats, optimizer=None, device="cpu"):
    """One pass through the data. If optimizer is given, trains; otherwise, eval-only."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, n_batches = 0.0, 0
    loss_fn = nn.MSELoss()

    for batch in loader:
        batch = batch.to(device)

        fe_true_norm = normalize(batch.y_formation_energy.view(-1, 1), stats["fe_mean"], stats["fe_std"])
        bg_true_norm = normalize(batch.y_band_gap.view(-1, 1), stats["bg_mean"], stats["bg_std"])

        with torch.set_grad_enabled(is_training):
            fe_pred, bg_pred = model(batch)
            loss = loss_fn(fe_pred, fe_true_norm) + loss_fn(bg_pred, bg_true_norm)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_dir", type=str, default="data/processed/graphs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading graphs...")
    graphs = load_all_graphs(args.graphs_dir)
    train_graphs, val_graphs, test_graphs = split_graphs(graphs)
    print(f"Split: {len(train_graphs)} train / {len(val_graphs)} val / {len(test_graphs)} test")

    stats = compute_norm_stats(train_graphs)
    print(f"Normalization stats (from train set): {stats}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    model = CGCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, stats, optimizer, device)
        val_loss = run_epoch(model, val_loader, stats, optimizer=None, device=device)

        print(f"Epoch {epoch}/{args.epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_stats": stats,
                "epoch": epoch,
                "val_loss": val_loss,
            }, CHECKPOINT_DIR / "cgcnn_best.pt")
            print(f"  -> new best, checkpoint saved")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()