"""
Turn active_learning.py's raw results.json into the sample-efficiency plot
and a write-up: does uncertainty-based acquisition reach a given accuracy
using fewer labels than random sampling?
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def labels_to_reach(rounds, target_mae, metric):
    """First n_labels at which `metric` <= target_mae, via linear
    interpolation between the two bracketing rounds. None if never reached."""
    xs = [r["n_labels"] for r in rounds]
    ys = [r[metric] for r in rounds]
    for i in range(len(xs)):
        if ys[i] <= target_mae:
            if i == 0:
                return xs[0]
            x0, y0, x1, y1 = xs[i - 1], ys[i - 1], xs[i], ys[i]
            if y0 == y1:
                return x1
            frac = (y0 - target_mae) / (y0 - y1)
            return x0 + frac * (x1 - x0)
    return None


def make_plot(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, label, unit in [
        (axes[0], "fe_mae", "Formation energy", "eV/atom"),
        (axes[1], "bg_mae", "Band gap", "eV"),
    ]:
        for strategy, color in [("uncertainty", "tab:red"), ("random", "tab:gray")]:
            xs = [r["n_labels"] for r in results[strategy]]
            ys = [r[metric] for r in results[strategy]]
            ax.plot(xs, ys, "o-", color=color, label=strategy)
        ax.set_xlabel("Number of labeled materials")
        ax.set_ylabel(f"Test MAE ({unit})")
        ax.set_title(label)
        ax.legend()

    fig.suptitle("Active learning: uncertainty-based vs. random acquisition")
    fig.tight_layout()
    fig.savefig(output_dir / "active_learning_curves.png", dpi=150)
    plt.close(fig)


def write_report(results, output_dir, acquisition_target):
    lines = [
        "# Active Learning Simulation: Uncertainty vs. Random Acquisition",
        "",
        "Pool-based active learning simulation on the same fixed held-out test set used "
        "throughout this project. Both strategies share an identical initial labeled set "
        "and round schedule -- the only difference is which materials get revealed each "
        f"round: highest ensemble-std ('uncertainty', acquiring by {acquisition_target.upper()} "
        "uncertainty) vs. uniformly random ('random', the baseline). A small ensemble is "
        "retrained from scratch each round (not warm-started) on the current labeled set.",
        "",
        "## Results by label budget",
        "",
        "| n_labels | uncertainty FE MAE | random FE MAE | uncertainty BG MAE | random BG MAE |",
        "|---|---|---|---|---|",
    ]
    for u, r in zip(results["uncertainty"], results["random"]):
        lines.append(
            f"| {u['n_labels']} | {u['fe_mae']:.4f} | {r['fe_mae']:.4f} | "
            f"{u['bg_mae']:.4f} | {r['bg_mae']:.4f} |"
        )

    lines += ["", "## Label efficiency", ""]
    for metric, label, unit in [("fe_mae", "Formation energy", "eV/atom"), ("bg_mae", "Band gap", "eV")]:
        random_final_mae = results["random"][-1][metric]
        # a couple of reference thresholds: the random strategy's own final MAE,
        # and its second-to-last round's MAE (an earlier point on its own curve).
        for ref_round_idx in [-1, -2]:
            if len(results["random"]) < abs(ref_round_idx):
                continue
            target = results["random"][ref_round_idx][metric]
            u_labels = labels_to_reach(results["uncertainty"], target, metric)
            r_labels = results["random"][ref_round_idx]["n_labels"]
            if u_labels is None:
                lines.append(
                    f"**{label}**, target MAE {target:.4f} {unit} (random's round "
                    f"{ref_round_idx}): uncertainty-based acquisition never reached this "
                    f"MAE within the labels tried."
                )
            else:
                saved = r_labels - u_labels
                pct = 100 * saved / r_labels if r_labels else 0
                lines.append(
                    f"**{label}**, target MAE {target:.4f} {unit}: uncertainty-based "
                    f"acquisition reached it at ~{u_labels:.0f} labels vs. random's "
                    f"{r_labels} labels "
                    + (f"({pct:.0f}% fewer labels)." if saved > 0 else "(no fewer labels -- no efficiency gain here).")
                )
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        f"Acquisition targets {acquisition_target.upper()} uncertainty specifically because "
        "Tier 3.2's calibration analysis found band gap's ensemble std ranks materials by "
        "difficulty more reliably (r=0.49) than formation energy's (r=0.23), even though its "
        "absolute magnitude is underconfident -- acquisition only needs correct relative "
        "ranking, not calibrated magnitude, so that mismatch doesn't invalidate using it here. "
        "Whether this actually beats random sampling, and by how much, is exactly what the "
        "table and plot above show for this run -- not asserted in advance.",
        "",
        "## Caveats",
        "",
        "- Each round trains a smaller, faster ensemble (not the full Tier 3.2 setup) to keep "
        "the whole multi-round simulation tractable, so absolute MAE at any given round is not "
        "directly comparable to the Tier 2/3.2 headline numbers -- only the *relative* gap "
        "between the two curves at equal label budgets is the actual finding here.",
        "- A single simulation run has no error bars on the acquisition-strategy comparison "
        "itself (unlike Tier 3.2's ensemble members, which do have across-seed variance) -- "
        "a repeat run with a different AL_SEED would be needed to know how much of any gap "
        "is signal vs. run-to-run noise.",
    ]

    (output_dir / "ACTIVE_LEARNING.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/active_learning")
    parser.add_argument("--acquisition_target", type=str, default="bg")
    args = parser.parse_args()

    output_dir = Path(args.results_dir)
    results = json.loads((output_dir / "active_learning_results.json").read_text())

    make_plot(results, output_dir)
    write_report(results, output_dir, args.acquisition_target)
    print(f"Done. Plot + report written to {output_dir}/")


if __name__ == "__main__":
    main()
