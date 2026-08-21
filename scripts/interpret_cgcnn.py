"""
Interpretability pass on the trained CGCNN checkpoint: extract the per-bond
gate values CGCNNConv already computes during its forward pass (how much
each neighbor's message is "let through" per layer) and check whether the
model is weighting chemically sensible bonds more heavily. No retraining —
hooks on gate_net's Linear layer inside each existing CGCNNConv.

Runs entirely on the held-out test split (same SEED=42 split used for
reported metrics), so the materials examined here were never seen in
training or checkpoint selection.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monty.json import MontyDecoder
from scipy import stats as scipy_stats
from torch_geometric.loader import DataLoader

from crystal_gnn.models.cgcnn import CGCNN

SEED = 42                 # must match train_cgcnn.py's split exactly
SAMPLE_SEED = 123          # separate seed: which test materials to inspect
CUTOFF_RADIUS = 8.0        # must match build_graphs.py exactly
MAX_NEIGHBORS = 12         # must match build_graphs.py exactly


def load_all_graphs_with_ids(graphs_dir):
    """Like evaluate_cgcnn.py's load_all_graphs, but keeps each graph's
    material_id (from its filename) paired with it, so gate values can be
    traced back to a real structure afterwards."""
    files = sorted(Path(graphs_dir).glob("*.pt"))
    ids = [f.stem for f in files]
    graphs = [torch.load(f, weights_only=False) for f in files]
    return list(zip(ids, graphs))


def split_paired(paired, train_frac=0.8, val_frac=0.1):
    """Identical shuffle logic to train_cgcnn.py / evaluate_cgcnn.py. Shuffling
    (id, graph) pairs together, from the same starting order, with the same
    seed, reproduces the exact same test-set membership as shuffling graphs
    alone — the permutation doesn't depend on the element's contents."""
    random.seed(SEED)
    shuffled = paired.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return shuffled[n_train + n_val:]  # only the test split is needed here


def load_structure(material_id, structures_path):
    """Scan structures.jsonl for one material_id's pymatgen Structure.
    Structures aren't indexed by id on disk, so this does a linear scan --
    fine for a one-off interpretability run over a handful of materials."""
    with open(structures_path) as f:
        for line in f:
            if f'"material_id": "{material_id}"' not in line:
                continue
            record = json.loads(line, cls=MontyDecoder)
            if record["material_id"] == material_id:
                return record["structure"]
    raise KeyError(f"{material_id} not found in {structures_path}")


def recompute_edges(structure):
    """Exact same neighbor-finding logic as build_graphs.py's
    structure_to_graph: periodic neighbors within CUTOFF_RADIUS, nearest
    MAX_NEIGHBORS per atom, sorted by distance. Reproduces the graph's
    edge_index order exactly, so gate values line up with real elements
    and real distances (rather than trying to invert the Gaussian-expanded
    edge_attr stored in the .pt file)."""
    all_neighbors = structure.get_all_neighbors(CUTOFF_RADIUS)

    edge_src, edge_dst, edge_dist, edge_elems = [], [], [], []
    for i, neighbors in enumerate(all_neighbors):
        nearest = sorted(neighbors, key=lambda n: n.nn_distance)[:MAX_NEIGHBORS]
        for n in nearest:
            edge_src.append(i)
            edge_dst.append(n.index)
            edge_dist.append(n.nn_distance)
            edge_elems.append((str(structure[i].specie), str(n.specie)))
    return edge_src, edge_dst, edge_dist, edge_elems


def register_gate_hooks(model):
    """One forward hook per conv layer's gate_net, capturing the sigmoid'd
    gate matrix (num_edges, atom_dim) each time propagate() calls message().
    gate = sigmoid(gate_net(combined)) exactly mirrors CGCNNConv.message."""
    captured = []

    def make_hook():
        def hook(module, inp, output):
            captured.append(torch.sigmoid(output).detach())
        return hook

    handles = [conv.gate_net.register_forward_hook(make_hook()) for conv in model.conv_layers]
    return captured, handles


def analyze_material(material_id, graph, model, structures_path):
    structure = load_structure(material_id, structures_path)
    edge_src, edge_dst, edge_dist, edge_elems = recompute_edges(structure)

    assert len(edge_src) == graph.edge_index.shape[1], (
        f"{material_id}: recomputed edge count {len(edge_src)} != stored graph "
        f"edge count {graph.edge_index.shape[1]} -- neighbor logic drifted from "
        f"build_graphs.py, gate values would not line up with the right bonds."
    )

    captured, handles = register_gate_hooks(model)
    try:
        with torch.no_grad():
            batch = graph.clone()
            batch.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
            model(batch)
    finally:
        for h in handles:
            h.remove()

    # captured[layer] has shape (num_edges, atom_dim); mean over channels ->
    # one scalar "how open was this bond's gate" per edge, per layer.
    per_layer_scalar = [g.mean(dim=-1).numpy() for g in captured]
    gate_final_layer = per_layer_scalar[-1]
    gate_mean_all_layers = np.mean(per_layer_scalar, axis=0)

    formula = structure.composition.reduced_formula
    rows = []
    for idx in range(len(edge_src)):
        elem_i, elem_j = edge_elems[idx]
        rows.append({
            "material_id": material_id,
            "formula": formula,
            "src_elem": elem_i,
            "dst_elem": elem_j,
            "hetero": elem_i != elem_j,
            "distance": edge_dist[idx],
            "gate_layer1": per_layer_scalar[0][idx],
            "gate_layer2": per_layer_scalar[1][idx],
            "gate_layer3": per_layer_scalar[2][idx],
            "gate_final_layer": gate_final_layer[idx],
            "gate_mean_all_layers": gate_mean_all_layers[idx],
        })
    return pd.DataFrame(rows)


def make_plots(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gate strength vs bond distance, hetero- vs homo-nuclear bonds.
    fig, ax = plt.subplots(figsize=(7, 5))
    for hetero, color, label in [(True, "tab:blue", "hetero-nuclear (e.g. cation-anion)"),
                                  (False, "tab:orange", "homo-nuclear (same element)")]:
        subset = df[df["hetero"] == hetero]
        ax.scatter(subset["distance"], subset["gate_final_layer"], s=10, alpha=0.4, color=color, label=label)
    ax.set_xlabel("Bond distance (Å)")
    ax.set_ylabel("Final-layer gate strength (mean over channels)")
    ax.set_title("CGCNN gate strength vs. bond distance\n(held-out test materials)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gate_vs_distance.png", dpi=150)
    plt.close(fig)

    # 2. Per-material bar charts of *distinct bond types* (crystal symmetry
    # means most individual bonds duplicate one of a handful of true bond
    # environments -- e.g. many symmetry-equivalent Zr-Zr bonds at the same
    # distance -- so group by (element pair, distance) before ranking, or
    # every material's "top-8" is just one bond type repeated 8 times).
    df["bond_type"] = (
        df[["src_elem", "dst_elem"]].apply(lambda r: "-".join(sorted(r)), axis=1)
        + " @ " + df["distance"].round(2).astype(str) + "Å"
    )
    material_ids = df["material_id"].unique()
    n = len(material_ids)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, mid in zip(axes, material_ids):
        sub = (
            df[df["material_id"] == mid]
            .groupby("bond_type", as_index=False)
            .agg(gate_final_layer=("gate_final_layer", "mean"), formula=("formula", "first"))
            .sort_values("gate_final_layer", ascending=False)
            .head(8)
        )
        ax.bar(range(len(sub)), sub["gate_final_layer"], color="tab:green")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["bond_type"], fontsize=7, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"{mid} ({sub['formula'].iloc[0]})", fontsize=9)
        ax.set_ylabel("gate")

    for ax in axes[len(material_ids):]:
        ax.axis("off")

    fig.suptitle("Highest-gated distinct bond types per material (final conv layer)")
    fig.tight_layout()
    fig.savefig(output_dir / "top_bonds_per_material.png", dpi=150)
    plt.close(fig)


def write_report(df, output_dir):
    hetero = df[df["hetero"]]["gate_final_layer"]
    homo = df[~df["hetero"]]["gate_final_layer"]
    corr, pvalue = scipy_stats.pearsonr(df["distance"], df["gate_final_layer"])

    t_stat, t_pvalue = (None, None)
    if len(homo) > 1 and len(hetero) > 1:
        t_stat, t_pvalue = scipy_stats.ttest_ind(hetero, homo, equal_var=False)

    lines = [
        "# CGCNN Interpretability: Bond Gate Analysis",
        "",
        "Extracted from `checkpoints/cgcnn_best.pt` via forward hooks on each ",
        "`CGCNNConv.gate_net` -- no retraining. Gate = `sigmoid(gate_net([x_i, x_j, edge_attr]))`, ",
        "a 64-dim vector per bond per layer; reported as its mean over channels, i.e. ",
        "how strongly that bond's message was let through, roughly \"attention weight\".",
        "",
        f"Materials analyzed: {df['material_id'].nunique()} (all from the held-out test split, ",
        "SEED=42 -- never seen in training or checkpoint selection).",
        f"Total bonds analyzed: {len(df)}",
        "",
        "## Findings",
        "",
        f"**Gate strength vs. bond distance:** Pearson r = {corr:.3f} (p = {pvalue:.2e}).",
    ]
    if corr < 0:
        lines.append(
            "Negative correlation -- shorter bonds tend to get stronger gates, consistent "
            "with shorter bonds generally being the chemically stronger/more relevant "
            "interaction."
        )
    else:
        lines.append(
            "Positive/no correlation -- gate strength is not simply distance-driven; the "
            "model appears to be using more than raw bond length to weight neighbors."
        )

    lines += [
        "",
        f"**Hetero-nuclear vs. homo-nuclear bonds:** mean gate {hetero.mean():.3f} "
        f"(n={len(hetero)}) vs. {homo.mean():.3f} (n={len(homo)}).",
    ]
    if t_pvalue is not None:
        lines.append(f"Welch's t-test: t = {t_stat:.2f}, p = {t_pvalue:.2e}.")
        if hetero.mean() > homo.mean() and t_pvalue < 0.05:
            lines.append(
                "Hetero-nuclear bonds (different elements -- typically the real "
                "cation-anion/covalent bonding interaction in a crystal) are gated "
                "significantly higher than homo-nuclear bonds. This is chemically "
                "sensible: the model is not treating all short contacts equally, it is "
                "preferentially weighting the bonds that carry the actual bonding "
                "chemistry."
            )
        elif homo.mean() > hetero.mean() and t_pvalue < 0.05:
            lines.append(
                "Homo-nuclear bonds are gated higher, which is the less chemically "
                "expected direction and worth a caveat rather than a clean claim in an "
                "interview -- possibly driven by metallic materials in this sample "
                "(homo-nuclear metal-metal bonding is real, just a different mechanism "
                "than ionic/covalent cation-anion bonding)."
            )
        else:
            lines.append("Difference is not statistically significant at p < 0.05.")

    lines += [
        "",
        "## Per-material detail",
        "",
        "See `top_bonds_per_material.png` for the 8 highest-gated bonds in each "
        "material examined, and `gate_vs_distance.png` for the full distance/gate "
        "scatter across all bonds. Raw per-bond values are in `gate_values.csv`.",
        "",
        "## Caveat",
        "",
        "This is a qualitative/statistical read of gate values on a handful of test "
        "materials, not a claim about global model behavior -- worth stating plainly if "
        "asked in an interview. A stronger next step (not done here) would be a full-test-set "
        "aggregate broken out by bond type, or ablating specific bonds and measuring the "
        "prediction shift directly.",
    ]

    (output_dir / "INTERPRETABILITY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_dir", type=str, default="data/processed/graphs")
    parser.add_argument("--structures_path", type=str, default="data/raw/structures.jsonl")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cgcnn_best.pt")
    parser.add_argument("--output_dir", type=str, default="results/interpretability")
    parser.add_argument("--n_materials", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("Loading graphs and reconstructing the held-out test split...")
    paired = load_all_graphs_with_ids(args.graphs_dir)
    test_paired = split_paired(paired)
    print(f"Test set: {len(test_paired)} materials")

    rng = random.Random(SAMPLE_SEED)
    sample = rng.sample(test_paired, min(args.n_materials, len(test_paired)))
    print(f"Inspecting {len(sample)} test materials: {[mid for mid, _ in sample]}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CGCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val loss: {checkpoint['val_loss']:.4f})")

    all_dfs = []
    for material_id, graph in sample:
        print(f"  analyzing {material_id}...")
        df = analyze_material(material_id, graph, model, args.structures_path)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_dir / "gate_values.csv", index=False)

    make_plots(full_df, output_dir)
    write_report(full_df, output_dir)

    print(f"\nDone. Report + plots + raw data written to {output_dir}/")


if __name__ == "__main__":
    main()
