"""
Build ALIGNN-style graphs: atom graph (same as CGCNN) + a derived line graph
encoding bond-bond angles at each atom (three-body information CGCNN's
distance-only edges can't represent). Reuses the same raw structures already
pulled — no new MP API calls needed.
"""

import json
import math
from pathlib import Path

import argparse

import torch
from monty.json import MontyDecoder
from torch_geometric.data import Data

STRUCTURES_PATH = Path("data/raw/structures.jsonl")
OUTPUT_DIR = Path("data/processed/graphs_alignn")

CUTOFF_RADIUS = 8.0
MAX_NEIGHBORS = 12
DIST_GAUSSIAN_CENTERS = torch.linspace(0, CUTOFF_RADIUS, 41)
DIST_GAUSSIAN_WIDTH = 0.2
ANGLE_GAUSSIAN_CENTERS = torch.linspace(0, math.pi, 41)
ANGLE_GAUSSIAN_WIDTH = 0.2

ATOM_INIT_PATH = Path(__file__).parent.parent / "src" / "crystal_gnn" / "models" / "atom_init.json"
with open(ATOM_INIT_PATH) as f:
    ATOM_FEATURES = {int(k): v for k, v in json.load(f).items()}


class ALIGNNData(Data):
    """Custom batching: line_graph_edge_index indexes into atom-graph EDGES,
    not atoms — PyG's default batching offset would be wrong without this."""
    def __inc__(self, key, value, *args, **kwargs):
        if key == "line_graph_edge_index":
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)


def gaussian_expand(values: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    diff = values.unsqueeze(-1) - centers
    return torch.exp(-(diff ** 2) / (width ** 2))


def structure_to_alignn_graph(structure, band_gap: float, formation_energy: float) -> ALIGNNData:
    node_features = torch.tensor(
        [ATOM_FEATURES[site.specie.Z] for site in structure], dtype=torch.float
    )

    all_neighbors = structure.get_all_neighbors(CUTOFF_RADIUS)

    edge_src, edge_dst, edge_dist = [], [], []
    # per-atom bookkeeping needed for the angle/line-graph step below
    atom_edge_ids = [[] for _ in range(len(structure))]
    atom_bond_vectors = [[] for _ in range(len(structure))]

    for i, neighbors in enumerate(all_neighbors):
        nearest = sorted(neighbors, key=lambda n: n.nn_distance)[:MAX_NEIGHBORS]
        for n in nearest:
            edge_id = len(edge_src)
            edge_src.append(i)
            edge_dst.append(n.index)
            edge_dist.append(n.nn_distance)

            vector = torch.tensor(n.coords - structure[i].coords, dtype=torch.float)
            atom_edge_ids[i].append(edge_id)
            atom_bond_vectors[i].append(vector)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = gaussian_expand(
        torch.tensor(edge_dist, dtype=torch.float), DIST_GAUSSIAN_CENTERS, DIST_GAUSSIAN_WIDTH
    )

    # Line graph: for each atom, every pair of its outgoing bonds becomes
    # a line-graph edge, with the angle between those bonds as the feature.
    lg_src, lg_dst, lg_angles = [], [], []
    for edge_ids, vectors in zip(atom_edge_ids, atom_bond_vectors):
        num_bonds = len(edge_ids)
        for a in range(num_bonds):
            for b in range(num_bonds):
                if a == b:
                    continue
                v_a, v_b = vectors[a], vectors[b]
                cos_angle = torch.dot(v_a, v_b) / (v_a.norm() * v_b.norm())
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # guard against float rounding
                angle = torch.acos(cos_angle)

                lg_src.append(edge_ids[a])
                lg_dst.append(edge_ids[b])
                lg_angles.append(angle.item())

    line_graph_edge_index = torch.tensor([lg_src, lg_dst], dtype=torch.long)
    line_graph_edge_attr = gaussian_expand(
        torch.tensor(lg_angles, dtype=torch.float), ANGLE_GAUSSIAN_CENTERS, ANGLE_GAUSSIAN_WIDTH
    )

    return ALIGNNData(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        line_graph_edge_index=line_graph_edge_index,
        line_graph_edge_attr=line_graph_edge_attr,
        y_band_gap=torch.tensor([band_gap], dtype=torch.float),
        y_formation_energy=torch.tensor([formation_energy], dtype=torch.float),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max structures to process (for testing/subsetting)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(STRUCTURES_PATH) as f:
        lines = f.readlines()

    if args.limit:
        lines = lines[:args.limit]

    print(f"Total structures to process: {len(lines)}")

    built, skipped, failed = 0, 0, 0
    for line in lines:
        record = json.loads(line, cls=MontyDecoder)
        material_id = record["material_id"]
        out_path = OUTPUT_DIR / f"{material_id}.pt"

        if out_path.exists():
            skipped += 1
            continue

        try:
            graph = structure_to_alignn_graph(
                record["structure"],
                record["band_gap"],
                record["formation_energy_per_atom"],
            )
            torch.save(graph, out_path)
            built += 1
        except Exception as e:
            print(f"  FAILED {material_id}: {e}")
            failed += 1

        if (built + skipped + failed) % 500 == 0:
            print(f"  progress: {built} built, {skipped} skipped, {failed} failed")

    print(f"\nDone. {built} built, {skipped} already existed, {failed} failed.")


if __name__ == "__main__":
    main()