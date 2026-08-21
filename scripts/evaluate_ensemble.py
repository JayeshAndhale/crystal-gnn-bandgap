"""
Evaluate a trained deep ensemble on the held-out test set: ensemble-mean
accuracy (does averaging N models beat a single model?) and calibration
(does higher ensemble disagreement/std actually predict higher error?).

Calibration is the point of this script -- a model that "knows what it
doesn't know" is a materially different, stronger claim than a model that
just outputs a number. Standard deep-ensemble diagnostics:
  - Pearson correlation between predicted std and actual |error| (positive
    and significant = the uncertainty estimate carries real signal).
  - Spread-skill plot: bin materials by predicted std, compare each bin's
    mean predicted std against its mean actual |error|. A well-calibrated
    ensemble tracks close to the y=x line.
  - Miscalibration ratio: mean(|error|) / mean(std). Under a well-calibrated
    Gaussian-residual assumption this should be close to sqrt(2/pi) ~= 0.80.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from build_graphs_alignn import ALIGNNData  # noqa: F401 -- must be module-level, see train_ensemble.py

SPLIT_SEED = 42  # must match train_ensemble.py exactly


def load_all_graphs_with_ids(graphs_dir):
    all_files = []
    for d in str(graphs_dir).split(","):
        all_files.extend(sorted(Path(d.strip()).glob("*.pt")))
    ids = [f.stem for f in all_files]
    graphs = [torch.load(f, weights_only=False) for f in all_files]
    return ids, graphs


def split_with_ids(ids, graphs, train_frac=0.8, val_frac=0.1):
    import random
    random.seed(SPLIT_SEED)
    paired = list(zip(ids, graphs))
    random.shuffle(paired)

    n = len(paired)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    test_paired = paired[n_train + n_val:]
    return [mid for mid, _ in test_paired], [g for _, g in test_paired]


def unnormalize(value, mean, std):
    return value * std + mean


def build_model(model_name):
    if model_name == "cgcnn":
        from crystal_gnn.models.cgcnn import CGCNN
        return CGCNN()
    elif model_name == "alignn":
        from crystal_gnn.models.alignn import ALIGNN
        return ALIGNN()
    raise ValueError(f"Unknown model: {model_name}")


def predict_all(model, loader, stats, device):
    """Returns (fe_pred, bg_pred, fe_true, bg_true) as flat numpy arrays, in
    loader order (shuffle=False, so this order is identical across members
    and matches test_ids)."""
    model.eval()
    fe_preds, bg_preds, fe_trues, bg_trues = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            fe_pred_norm, bg_pred_norm = model(batch)
            fe_preds.append(unnormalize(fe_pred_norm, stats["fe_mean"], stats["fe_std"]).cpu())
            bg_preds.append(unnormalize(bg_pred_norm, stats["bg_mean"], stats["bg_std"]).cpu())
            fe_trues.append(batch.y_formation_energy.view(-1, 1).cpu())
            bg_trues.append(batch.y_band_gap.view(-1, 1).cpu())
    return (torch.cat(fe_preds).numpy().flatten(), torch.cat(bg_preds).numpy().flatten(),
            torch.cat(fe_trues).numpy().flatten(), torch.cat(bg_trues).numpy().flatten())


def spread_skill(pred_std, abs_error, n_bins=10):
    """Bin materials by predicted std into n_bins quantile bins; return each
    bin's mean predicted std and mean actual abs error."""
    order = np.argsort(pred_std)
    std_sorted = pred_std[order]
    err_sorted = abs_error[order]
    bins_std, bins_err = [], []
    for chunk_std, chunk_err in zip(np.array_split(std_sorted, n_bins), np.array_split(err_sorted, n_bins)):
        if len(chunk_std) == 0:
            continue
        bins_std.append(chunk_std.mean())
        bins_err.append(chunk_err.mean())
    return np.array(bins_std), np.array(bins_err)


def make_plots(df, output_dir, model_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, prefix, label, unit in [
        (axes[0], "fe", "Formation energy", "eV/atom"),
        (axes[1], "bg", "Band gap", "eV"),
    ]:
        std = df[f"{prefix}_ens_std"].values
        err = df[f"{prefix}_abs_error"].values
        bin_std, bin_err = spread_skill(std, err)

        lim = max(bin_std.max(), bin_err.max()) * 1.15
        ax.plot([0, lim], [0, lim], "--", color="gray", label="perfect calibration (y=x)")
        ax.plot(bin_std, bin_err, "o-", color="tab:red", label="observed (10 bins)")
        ax.set_xlabel(f"Mean predicted std ({unit})")
        ax.set_ylabel(f"Mean actual |error| ({unit})")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_name.upper()} ensemble calibration (spread-skill, held-out test set)")
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, prefix, label, unit in [
        (axes[0], "fe", "Formation energy", "eV/atom"),
        (axes[1], "bg", "Band gap", "eV"),
    ]:
        ax.scatter(df[f"{prefix}_ens_std"], df[f"{prefix}_abs_error"], s=6, alpha=0.25, color="tab:blue")
        ax.set_xlabel(f"Predicted std ({unit})")
        ax.set_ylabel(f"Actual |error| ({unit})")
        ax.set_title(label)
    fig.suptitle(f"{model_name.upper()} ensemble: predicted uncertainty vs. actual error (per material)")
    fig.tight_layout()
    fig.savefig(output_dir / "uncertainty_vs_error_scatter.png", dpi=150)
    plt.close(fig)


def write_report(df, output_dir, model_name, n_members, member_maes, ensemble_maes):
    lines = [
        f"# {model_name.upper()} Ensemble Uncertainty Quantification",
        "",
        f"{n_members}-member deep ensemble, identical architecture, identical train/val/test "
        "split (SPLIT_SEED=42) for every member -- only model init + minibatch shuffling "
        "differ (per-member torch seed). Evaluated on the held-out test set "
        f"({len(df)} materials), never seen during training or checkpoint selection by any member.",
        "",
        "## Ensembling vs. a single model",
        "",
        f"Mean individual-member test MAE: FE {member_maes['fe']:.4f} eV/atom, "
        f"BG {member_maes['bg']:.4f} eV.",
        f"Ensemble-mean test MAE: FE {ensemble_maes['fe']:.4f} eV/atom, "
        f"BG {ensemble_maes['bg']:.4f} eV.",
    ]
    fe_delta = member_maes["fe"] - ensemble_maes["fe"]
    bg_delta = member_maes["bg"] - ensemble_maes["bg"]
    lines.append(
        f"Averaging the ensemble {'improved' if fe_delta > 0 else 'did not improve'} formation "
        f"energy MAE by {abs(fe_delta):.4f} and {'improved' if bg_delta > 0 else 'did not improve'} "
        f"band gap MAE by {abs(bg_delta):.4f}, relative to the average single member -- "
        "the expected direction if members make partially independent errors."
    )

    lines += ["", "## Calibration", ""]
    for prefix, label, unit in [("fe", "Formation energy", "eV/atom"), ("bg", "Band gap", "eV")]:
        std = df[f"{prefix}_ens_std"].values
        err = df[f"{prefix}_abs_error"].values
        corr, pvalue = scipy_stats.pearsonr(std, err)
        ratio = err.mean() / std.mean()
        lines.append(
            f"**{label}:** corr(predicted std, actual |error|) = {corr:.3f} (p = {pvalue:.2e}); "
            f"mean predicted std = {std.mean():.4f} {unit}; mean actual |error| = {err.mean():.4f} {unit}; "
            f"miscalibration ratio (mean|error| / mean std) = {ratio:.3f} "
            "(well-calibrated Gaussian residuals -> ~0.80)."
        )
        if corr > 0.15 and pvalue < 0.05:
            lines.append(
                "Positive, significant correlation -- the ensemble's disagreement is real signal: "
                "materials it's more uncertain about tend to actually be the ones it gets more wrong."
            )
        else:
            lines.append(
                "Weak/no correlation -- the ensemble's spread is not a reliable error predictor for "
                "this target; worth stating as a limitation rather than a working UQ claim."
            )
        lines.append("")

    lines += [
        "## Files",
        "",
        "`calibration.png` -- spread-skill plot (predicted std vs. actual error, binned).",
        "`uncertainty_vs_error_scatter.png` -- same relationship, per-material, unbinned.",
        "`ensemble_predictions.csv` -- raw per-material ensemble mean/std/error for both targets.",
        "",
        "## Caveat",
        "",
        "Ensemble size (N members) sets the resolution of the std estimate; with a small N, "
        "individual per-material std values are noisy even if the aggregate calibration trend "
        "above is real. Read the binned spread-skill plot, not single points on the per-material "
        "scatter, as the calibration evidence.",
    ]

    (output_dir / "UNCERTAINTY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["cgcnn", "alignn"], default="cgcnn")
    parser.add_argument("--graphs_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                         help="Default: checkpoints/ensemble_<model>")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Default: results/uncertainty_<model>")
    args = parser.parse_args()

    graphs_dir = args.graphs_dir or (
        "data/processed/graphs" if args.model == "cgcnn" else "data/processed/graphs_alignn"
    )
    checkpoint_dir = Path(args.checkpoint_dir or f"checkpoints/ensemble_{args.model}")
    output_dir = Path(args.output_dir or f"results/uncertainty_{args.model}")

    member_paths = sorted(checkpoint_dir.glob("member_seed*_best.pt"))
    if not member_paths:
        raise FileNotFoundError(
            f"No ensemble checkpoints found in {checkpoint_dir}/ -- run train_ensemble.py first "
            f"(and pull the trained checkpoints back from Kaggle if they were trained there)."
        )
    print(f"Found {len(member_paths)} ensemble members in {checkpoint_dir}/")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading graphs and reconstructing the held-out test split...")
    ids, graphs = load_all_graphs_with_ids(graphs_dir)
    test_ids, test_graphs = split_with_ids(ids, graphs)
    print(f"Test set: {len(test_graphs)} materials")
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    fe_preds_all, bg_preds_all = [], []
    fe_true, bg_true = None, None
    member_fe_maes, member_bg_maes = [], []

    for path in member_paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = build_model(args.model).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  member seed={checkpoint.get('seed', '?')}: epoch {checkpoint['epoch']}, "
              f"val loss {checkpoint['val_loss']:.4f}")

        fe_pred, bg_pred, fe_t, bg_t = predict_all(model, test_loader, checkpoint["norm_stats"], device)
        fe_preds_all.append(fe_pred)
        bg_preds_all.append(bg_pred)
        fe_true, bg_true = fe_t, bg_t  # identical every member (shuffle=False, shared split)

        member_fe_maes.append(np.abs(fe_pred - fe_t).mean())
        member_bg_maes.append(np.abs(bg_pred - bg_t).mean())

    fe_preds_all = np.stack(fe_preds_all)  # (n_members, n_test)
    bg_preds_all = np.stack(bg_preds_all)

    fe_ens_mean = fe_preds_all.mean(axis=0)
    fe_ens_std = fe_preds_all.std(axis=0)
    bg_ens_mean = bg_preds_all.mean(axis=0)
    bg_ens_std = bg_preds_all.std(axis=0)

    df = pd.DataFrame({
        "material_id": test_ids,
        "fe_true": fe_true, "fe_ens_mean": fe_ens_mean, "fe_ens_std": fe_ens_std,
        "fe_abs_error": np.abs(fe_ens_mean - fe_true),
        "bg_true": bg_true, "bg_ens_mean": bg_ens_mean, "bg_ens_std": bg_ens_std,
        "bg_abs_error": np.abs(bg_ens_mean - bg_true),
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "ensemble_predictions.csv", index=False)

    member_maes = {"fe": float(np.mean(member_fe_maes)), "bg": float(np.mean(member_bg_maes))}
    ensemble_maes = {"fe": float(df["fe_abs_error"].mean()), "bg": float(df["bg_abs_error"].mean())}

    print("\n--- Results ---")
    print(f"Mean individual-member MAE: FE {member_maes['fe']:.4f} eV/atom, BG {member_maes['bg']:.4f} eV")
    print(f"Ensemble-mean MAE:          FE {ensemble_maes['fe']:.4f} eV/atom, BG {ensemble_maes['bg']:.4f} eV")

    make_plots(df, output_dir, args.model)
    write_report(df, output_dir, args.model, len(member_paths), member_maes, ensemble_maes)

    print(f"\nDone. Report + plots + raw data written to {output_dir}/")


if __name__ == "__main__":
    main()
