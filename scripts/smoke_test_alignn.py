"""
CPU smoke test for ALIGNN: load a few real graphs, batch them, run one
forward pass. Checks plumbing, not accuracy — nothing's trained yet.
"""

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).parent))
from build_graphs_alignn import ALIGNNData  # noqa: F401 -- needed for torch.load to unpickle correctly

from crystal_gnn.models.alignn import ALIGNN

GRAPHS_DIR = Path("data/processed/graphs_alignn")
NUM_TEST_GRAPHS = 3


def main():
    graph_files = sorted(GRAPHS_DIR.glob("*.pt"))[:NUM_TEST_GRAPHS]
    graphs = [torch.load(f, weights_only=False) for f in graph_files]

    print(f"Loaded {len(graphs)} graphs:")
    for f, g in zip(graph_files, graphs):
        print(f"  {f.name}: {g.x.shape[0]} atoms, {g.edge_index.shape[1]} bonds, "
              f"{g.line_graph_edge_index.shape[1]} bond-pairs")

    batch = Batch.from_data_list(graphs)
    print(f"\nBatched: {batch.x.shape[0]} total atoms, "
          f"{batch.edge_index.shape[1]} total bonds, "
          f"{batch.line_graph_edge_index.shape[1]} total bond-pairs, "
          f"across {batch.num_graphs} crystals")

    model = ALIGNN()
    pred_formation_energy, pred_band_gap = model(batch)

    print(f"\nOutput shapes:")
    print(f"  formation energy predictions: {pred_formation_energy.shape}")
    print(f"  band gap predictions: {pred_band_gap.shape}")
    print(f"  (should both be [{len(graphs)}, 1] — one prediction per crystal, not per atom)")


if __name__ == "__main__":
    main()