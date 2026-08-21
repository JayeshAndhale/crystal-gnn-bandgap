"""
Active learning simulation: does picking which materials to label next by
model uncertainty (rather than randomly) reach a given accuracy using fewer
labels? Depends on Tier 3.2's ensemble UQ -- uncertainty here is the same
ensemble-std signal validated there, and deliberately defaults to band gap
(--acquisition_target bg) because Tier 3.2 found BG's ensemble std ranks
materials by difficulty more reliably (r=0.49) than FE's (r=0.23), even
though BG's uncertainty is underconfident in absolute magnitude -- for
acquisition we only need the *ranking* to be right, not the magnitude.

Protocol (standard pool-based active learning):
- The held-out TEST set (same SPLIT_SEED=42 split as every other script
  here) is fixed throughout and never touched by training or acquisition --
  used only to measure "how good is the model" at each round.
- The train+val split is combined into one POOL. A small random SEED_SIZE
  subset starts "labeled"; the rest starts "unlabeled" (labels exist in the
  data but are hidden from the acquisition logic until revealed).
- Each round: train a small fresh ensemble (ENSEMBLE_SIZE members, from
  scratch -- not warm-started, to avoid confounding round comparisons) on
  the current labeled set, evaluate its ensemble-mean MAE on the fixed test
  set, then reveal ACQUISITION_SIZE more materials from the pool -- either
  the most-uncertain ones ("uncertainty" strategy) or a random subset
  ("random" strategy, the baseline). Both strategies share the exact same
  initial labeled set and round schedule for a fair comparison.
- Resumable per round: round state (which materials are labeled, this
  round's MAE) is written to round_state.json after each round completes,
  and each round's member checkpoints are skipped if already trained --
  this is a long, multi-hour, multi-round job and Kaggle sessions in this
  project have died mid-run before (see PROJECT_HANDOFF.md).
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from crystal_gnn.models.cgcnn import CGCNN

SPLIT_SEED = 42  # must match every other script's train/val/test split
AL_SEED = 7      # separate seed: initial labeled set + random-strategy draws


def load_all_graphs_with_ids(graphs_dir):
    files = sorted(Path(graphs_dir).glob("*.pt"))
    ids = [f.stem for f in files]
    graphs = [torch.load(f, weights_only=False) for f in files]
    return dict(zip(ids, graphs))


def split_pool_and_test(id_to_graph, train_frac=0.8, val_frac=0.1):
    """Same split logic as every other script (SPLIT_SEED=42), but here
    train+val are merged into one 'pool' -- val's only role elsewhere is
    checkpoint selection, which this script doesn't do (fixed epoch budget
    per round instead, see module docstring)."""
    ids = list(id_to_graph.keys())
    random.seed(SPLIT_SEED)
    shuffled = ids.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    pool_ids = shuffled[:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]
    return pool_ids, test_ids


def normalize(value, mean, std):
    return (value - mean) / std


def unnormalize(value, mean, std):
    return value * std + mean


def compute_norm_stats(graphs):
    fe = torch.tensor([g.y_formation_energy.item() for g in graphs])
    bg = torch.tensor([g.y_band_gap.item() for g in graphs])
    return {
        "fe_mean": fe.mean().item(), "fe_std": fe.std().item(),
        "bg_mean": bg.mean().item(), "bg_std": bg.std().item(),
    }


def train_one_member(seed, train_graphs, stats, epochs, batch_size, lr, device, checkpoint_path):
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = CGCNN().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    torch.manual_seed(seed)
    loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    model = CGCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            fe_true = normalize(batch.y_formation_energy.view(-1, 1), stats["fe_mean"], stats["fe_std"])
            bg_true = normalize(batch.y_band_gap.view(-1, 1), stats["bg_mean"], stats["bg_std"])
            fe_pred, bg_pred = model(batch)
            loss = loss_fn(fe_pred, fe_true) + loss_fn(bg_pred, bg_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "norm_stats": stats, "seed": seed}, checkpoint_path)
    return model


def predict_ensemble(models, graphs, stats, batch_size, device):
    """Returns (fe_mean, fe_std, bg_mean, bg_std, fe_true, bg_true), each a
    flat numpy array in `graphs` order (loader shuffle=False)."""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    fe_all, bg_all, fe_true, bg_true = [], [], [], []

    for model in models:
        model.eval()
        fe_preds, bg_preds = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                fe_pred_norm, bg_pred_norm = model(batch)
                fe_preds.append(unnormalize(fe_pred_norm, stats["fe_mean"], stats["fe_std"]).cpu())
                bg_preds.append(unnormalize(bg_pred_norm, stats["bg_mean"], stats["bg_std"]).cpu())
        fe_all.append(torch.cat(fe_preds).flatten())
        bg_all.append(torch.cat(bg_preds).flatten())

    # true values only need to be read once (identical every member)
    with torch.no_grad():
        for batch in loader:
            fe_true.append(batch.y_formation_energy.view(-1))
            bg_true.append(batch.y_band_gap.view(-1))

    fe_all = torch.stack(fe_all)  # (n_members, n_graphs)
    bg_all = torch.stack(bg_all)
    return (
        fe_all.mean(0).numpy(), fe_all.std(0).numpy(),
        bg_all.mean(0).numpy(), bg_all.std(0).numpy(),
        torch.cat(fe_true).numpy(), torch.cat(bg_true).numpy(),
    )


def run_strategy(strategy, pool_ids, id_to_graph, test_graphs, args, device):
    strategy_dir = Path(args.checkpoint_dir) / strategy
    state_path = strategy_dir / "round_state.json"
    strategy_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(AL_SEED)
    shuffled_pool = pool_ids.copy()
    rng.shuffle(shuffled_pool)
    initial_labeled = set(shuffled_pool[:args.seed_size])
    full_unlabeled_order = shuffled_pool[args.seed_size:]  # only used if pool runs out

    if state_path.exists():
        rounds = json.loads(state_path.read_text())
        print(f"[{strategy}] resuming from round {len(rounds) - 1} ({rounds[-1]['n_labels']} labels)")
        labeled_ids = set(rounds[-1]["labeled_ids"])
        unlabeled_ids = [i for i in pool_ids if i not in labeled_ids]
    else:
        rounds = []
        labeled_ids = initial_labeled
        unlabeled_ids = [i for i in pool_ids if i not in labeled_ids]

    start_round = len(rounds)
    for r in range(start_round, args.n_rounds + 1):
        print(f"\n[{strategy}] round {r}: training on {len(labeled_ids)} labeled materials")
        labeled_graphs = [id_to_graph[i] for i in labeled_ids]
        stats = compute_norm_stats(labeled_graphs)

        models = []
        for seed in range(args.ensemble_size):
            ckpt_path = strategy_dir / f"round{r}" / f"member{seed}.pt"
            model = train_one_member(seed, labeled_graphs, stats, args.epochs_per_round,
                                      args.batch_size, args.lr, device, ckpt_path)
            models.append(model)

        fe_mean, fe_std, bg_mean, bg_std, fe_true, bg_true = predict_ensemble(
            models, test_graphs, stats, args.batch_size, device
        )
        fe_mae = float(abs(fe_mean - fe_true).mean())
        bg_mae = float(abs(bg_mean - bg_true).mean())
        print(f"[{strategy}] round {r}: test FE MAE {fe_mae:.4f}, BG MAE {bg_mae:.4f}")

        acquired = []
        if r < args.n_rounds and unlabeled_ids:
            k = min(args.acquisition_size, len(unlabeled_ids))
            if strategy == "uncertainty":
                unlabeled_graphs = [id_to_graph[i] for i in unlabeled_ids]
                _, u_fe_std, _, u_bg_std, _, _ = predict_ensemble(
                    models, unlabeled_graphs, stats, args.batch_size, device
                )
                score = u_bg_std if args.acquisition_target == "bg" else u_fe_std
                ranked = sorted(zip(unlabeled_ids, score), key=lambda x: -x[1])
                acquired = [i for i, _ in ranked[:k]]
            elif strategy == "random":
                round_rng = random.Random(AL_SEED + r)
                acquired = round_rng.sample(unlabeled_ids, k)

            labeled_ids = labeled_ids | set(acquired)
            unlabeled_ids = [i for i in unlabeled_ids if i not in set(acquired)]

        rounds.append({
            "round": r, "n_labels": len(labeled_ids) - len(acquired) if r < args.n_rounds else len(labeled_ids),
            "fe_mae": fe_mae, "bg_mae": bg_mae,
            "labeled_ids": sorted(labeled_ids - set(acquired)) if r < args.n_rounds else sorted(labeled_ids),
            "acquired_ids": acquired,
        })
        # n_labels/labeled_ids above reflect the set THIS round trained on (pre-acquisition),
        # which is what the MAE was actually measured with.
        state_path.write_text(json.dumps(rounds, indent=2))

    return rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_dir", type=str, default="data/processed/graphs")
    parser.add_argument("--seed_size", type=int, default=2000)
    parser.add_argument("--acquisition_size", type=int, default=3000)
    parser.add_argument("--n_rounds", type=int, default=6)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--epochs_per_round", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--acquisition_target", type=str, choices=["fe", "bg"], default="bg")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/active_learning")
    parser.add_argument("--output_dir", type=str, default="results/active_learning")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading graphs...")
    id_to_graph = load_all_graphs_with_ids(args.graphs_dir)
    pool_ids, test_ids = split_pool_and_test(id_to_graph)
    test_graphs = [id_to_graph[i] for i in test_ids]
    print(f"Pool: {len(pool_ids)} materials (train+val combined) | Test: {len(test_graphs)} (fixed, held out)")
    print(f"Seed size {args.seed_size}, +{args.acquisition_size}/round x {args.n_rounds} rounds "
          f"-> final label budget {min(args.seed_size + args.n_rounds * args.acquisition_size, len(pool_ids))}")

    all_results = {}
    for strategy in ["uncertainty", "random"]:
        all_results[strategy] = run_strategy(strategy, pool_ids, id_to_graph, test_graphs, args, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "active_learning_results.json").write_text(json.dumps(all_results, indent=2))

    print("\n--- Summary (test MAE by label budget) ---")
    print(f"{'n_labels':>10} | {'uncertainty FE':>14} | {'random FE':>10} | {'uncertainty BG':>14} | {'random BG':>10}")
    for u, r in zip(all_results["uncertainty"], all_results["random"]):
        print(f"{u['n_labels']:>10} | {u['fe_mae']:>14.4f} | {r['fe_mae']:>10.4f} | "
              f"{u['bg_mae']:>14.4f} | {r['bg_mae']:>10.4f}")

    print(f"\nDone. Results in {output_dir}/active_learning_results.json")
    print("Run scripts/plot_active_learning.py next for the sample-efficiency plot + writeup.")


if __name__ == "__main__":
    main()
