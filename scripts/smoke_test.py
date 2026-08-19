"""
CPU smoke test: load a few real graphs, batch them, run one forward pass.
Checks the model's plumbing works end-to-end — not checking accuracy,
since nothing has been trained yet.
"""

from pathlib import Path

import torch
from torch_geometric.data import Batch

from crystal_gnn.models.cgcnn import CGCNN

GRAPHS_DIR = Path("data/processed/graphs")
NUM_TEST_GRAPHS = 3


def main():
    graph_files = sorted(GRAPHS_DIR.glob("*.pt"))[:NUM_TEST_GRAPHS]
    graphs = [torch.load(f, weights_only=False) for f in graph_files]

    print(f"Loaded {len(graphs)} graphs:")
    for f, g in zip(graph_files, graphs):
        print(f"  {f.name}: {g.x.shape[0]} atoms, {g.edge_index.shape[1]} edges")

    batch = Batch.from_data_list(graphs)
    print(f"\nBatched: {batch.x.shape[0]} total atoms across {batch.num_graphs} crystals")
    print(f"batch.batch (per-atom crystal index): {batch.batch.tolist()}")

    model = CGCNN()
    pred_formation_energy, pred_band_gap = model(batch)

    print(f"\nOutput shapes:")
    print(f"  formation energy predictions: {pred_formation_energy.shape}")
    print(f"  band gap predictions: {pred_band_gap.shape}")
    print(f"  (should both be [{len(graphs)}, 1] — one prediction per crystal, not per atom)")


if __name__ == "__main__":
    main()