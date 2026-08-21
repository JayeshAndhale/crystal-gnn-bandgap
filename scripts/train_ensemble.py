"""
Train a deep ensemble (N models, same architecture, different seeds) for
uncertainty quantification. Works for either CGCNN or ALIGNN via --model.

Design, matching standard deep-ensemble practice:
- The train/val/test SPLIT is identical across all members (fixed SPLIT_SEED,
  same logic as train_cgcnn.py / train_alignn.py) -- only model init +
  DataLoader shuffling differ per member (via torch.manual_seed(member_seed)).
  Varying the split too would confound "model disagreement" with "different
  members literally trained on different data", which is not what ensemble
  uncertainty is supposed to measure.
- Normalization stats are computed once (from the shared train split) and
  reused for every member, so their predictions are on the same scale and
  directly comparable/averageable.
- Resumable: an existing member checkpoint is skipped unless --force is
  passed -- Kaggle sessions in this project have died mid-run before (see
  PROJECT_HANDOFF.md), and a 5-member ensemble is 5x the compute of the
  single-model runs, so losing partial progress is a real risk worth
  designing around, not a hypothetical one.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from build_graphs_alignn import ALIGNNData  # noqa: F401 -- must be a module-level (not function-local)
# name for unpickling to find it: the .pt files were built by running
# build_graphs_alignn.py directly, so ALIGNNData was pickled under module
# "__main__" -- this import has to bind it as an attribute of *this* script's
# own __main__ namespace when run as a script, which only works at module scope.

SPLIT_SEED = 42  # must match train_cgcnn.py / train_alignn.py -- identical split for every member


def load_all_graphs(graphs_dir):
    """graphs_dir may be a single path or multiple, comma-separated (ALIGNN's
    25k-material set is split across several Kaggle Datasets)."""
    all_files = []
    for d in str(graphs_dir).split(","):
        all_files.extend(sorted(Path(d.strip()).glob("*.pt")))
    return [torch.load(f, weights_only=False) for f in all_files]


def split_graphs(graphs, train_frac=0.8, val_frac=0.1):
    random.seed(SPLIT_SEED)
    shuffled = graphs.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def compute_norm_stats(train_graphs):
    fe = torch.tensor([g.y_formation_energy.item() for g in train_graphs])
    bg = torch.tensor([g.y_band_gap.item() for g in train_graphs])
    return {
        "fe_mean": fe.mean().item(), "fe_std": fe.std().item(),
        "bg_mean": bg.mean().item(), "bg_std": bg.std().item(),
    }


def normalize(value, mean, std):
    return (value - mean) / std


def run_epoch(model, loader, stats, optimizer=None, device="cpu"):
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


def build_model(model_name):
    if model_name == "cgcnn":
        from crystal_gnn.models.cgcnn import CGCNN
        return CGCNN()
    elif model_name == "alignn":
        from crystal_gnn.models.alignn import ALIGNN
        return ALIGNN()
    raise ValueError(f"Unknown model: {model_name}")


def train_one_member(model_name, seed, train_graphs, val_graphs, stats, args, device, checkpoint_path):
    torch.manual_seed(seed)  # only source of difference between ensemble members

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    model = build_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, stats, optimizer, device)
        val_loss = run_epoch(model, val_loader, stats, optimizer=None, device=device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  [seed {seed}] epoch {epoch}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} | lr {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_stats": stats,
                "epoch": epoch,
                "val_loss": val_loss,
                "seed": seed,
                "model_name": model_name,
            }, checkpoint_path)

    return best_epoch, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["cgcnn", "alignn"], default="cgcnn")
    parser.add_argument("--graphs_dir", type=str, default=None,
                         help="Default: data/processed/graphs (cgcnn) or data/processed/graphs_alignn (alignn)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5",
                         help="Comma-separated ensemble member seeds. Deliberately distinct from "
                              "SPLIT_SEED=42 to avoid confusing 'the split seed' with 'a member seed'.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                         help="Default: checkpoints/ensemble_<model>")
    parser.add_argument("--force", action="store_true", help="Retrain members even if their checkpoint already exists")
    args = parser.parse_args()

    graphs_dir = args.graphs_dir or (
        "data/processed/graphs" if args.model == "cgcnn" else "data/processed/graphs_alignn"
    )
    checkpoint_dir = Path(args.checkpoint_dir or f"checkpoints/ensemble_{args.model}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model} | ensemble seeds: {seeds}")

    print("Loading graphs...")
    graphs = load_all_graphs(graphs_dir)
    train_graphs, val_graphs, test_graphs = split_graphs(graphs)
    print(f"Split (SPLIT_SEED={SPLIT_SEED}, shared by every member): "
          f"{len(train_graphs)} train / {len(val_graphs)} val / {len(test_graphs)} test")

    stats = compute_norm_stats(train_graphs)
    print(f"Normalization stats (from train set, shared by every member): {stats}")

    results = []
    for seed in seeds:
        checkpoint_path = checkpoint_dir / f"member_seed{seed}_best.pt"
        if checkpoint_path.exists() and not args.force:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            print(f"\nSeed {seed}: checkpoint already exists (epoch {checkpoint['epoch']}, "
                  f"val loss {checkpoint['val_loss']:.4f}) -- skipping. Use --force to retrain.")
            results.append((seed, checkpoint["epoch"], checkpoint["val_loss"]))
            continue

        print(f"\nTraining member seed={seed} -> {checkpoint_path}")
        best_epoch, best_val_loss = train_one_member(
            args.model, seed, train_graphs, val_graphs, stats, args, device, checkpoint_path
        )
        print(f"  -> best epoch {best_epoch}, val loss {best_val_loss:.4f}")
        results.append((seed, best_epoch, best_val_loss))

    print("\n--- Ensemble training summary ---")
    for seed, epoch, val_loss in results:
        print(f"  seed {seed}: best epoch {epoch}, val loss {val_loss:.4f}")
    print(f"\nAll member checkpoints in {checkpoint_dir}/")
    print("Push this directory to a permanent Kaggle Dataset before ending the session "
          "(see PROJECT_HANDOFF.md's infrastructure lessons) -- then run evaluate_ensemble.py.")


if __name__ == "__main__":
    main()
