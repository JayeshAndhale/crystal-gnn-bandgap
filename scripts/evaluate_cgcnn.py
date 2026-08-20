"""
Evaluate the best CGCNN checkpoint on the held-out test set.
Un-normalizes predictions back to real units (eV/atom, eV) and reports MAE —
the same metric CGCNN's published benchmarks use, for an honest comparison.
"""

import argparse
import random
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from crystal_gnn.models.cgcnn import CGCNN

SEED = 42  # must match train_cgcnn.py exactly, so the test split is identical


def load_all_graphs(graphs_dir):
    files = sorted(Path(graphs_dir).glob("*.pt"))
    return [torch.load(f, weights_only=False) for f in files]


def split_graphs(graphs, train_frac=0.8, val_frac=0.1):
    """Identical logic to train_cgcnn.py — same seed, same shuffle, same
    split fractions — so this reconstructs the exact same test set that
    was never touched during training or checkpoint selection."""
    random.seed(SEED)
    shuffled = graphs.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def unnormalize(value, mean, std):
    return value * std + mean


def evaluate(model, loader, stats, device):
    model.eval()
    fe_abs_errors, bg_abs_errors = [], []

    with torch.no_grad():  # no gradients needed — we're not training
        for batch in loader:
            batch = batch.to(device)
            fe_pred_norm, bg_pred_norm = model(batch)

            fe_pred = unnormalize(fe_pred_norm, stats["fe_mean"], stats["fe_std"])
            bg_pred = unnormalize(bg_pred_norm, stats["bg_mean"], stats["bg_std"])

            fe_true = batch.y_formation_energy.view(-1, 1)
            bg_true = batch.y_band_gap.view(-1, 1)

            fe_abs_errors.append((fe_pred - fe_true).abs())
            bg_abs_errors.append((bg_pred - bg_true).abs())

    fe_mae = torch.cat(fe_abs_errors).mean().item()
    bg_mae = torch.cat(bg_abs_errors).mean().item()
    return fe_mae, bg_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_dir", type=str, default="data/processed/graphs")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cgcnn_best.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    graphs = load_all_graphs(args.graphs_dir)
    _, _, test_graphs = split_graphs(graphs)
    print(f"Test set: {len(test_graphs)} graphs (never seen during training or checkpoint selection)")

    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CGCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val loss: {checkpoint['val_loss']:.4f})")

    fe_mae, bg_mae = evaluate(model, test_loader, checkpoint["norm_stats"], device)

    print(f"\n--- Test set results ---")
    print(f"Formation energy MAE: {fe_mae:.4f} eV/atom  (CGCNN paper reports ~0.039 eV/atom)")
    print(f"Band gap MAE:         {bg_mae:.4f} eV        (CGCNN paper reports ~0.29-0.39 eV)")


if __name__ == "__main__":
    main()