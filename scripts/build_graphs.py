"""
Build crystal graphs from raw structures: atoms -> nodes, periodic neighbors
within a cutoff -> edges, distances -> Gaussian-expanded edge features.
One .pt file per material, resumable — skips anything already built.
"""

import json
from pathlib import Path

import torch
from monty.json import MontyDecoder
from torch_geometric.data import Data

ATOM_INIT_PATH = Path(__file__).parent.parent / "src" / "crystal_gnn" / "models" / "atom_init.json"
with open(ATOM_INIT_PATH) as f:
    ATOM_FEATURES = {int(k): v for k, v in json.load(f).items()}

STRUCTURES_PATH = Path("data/raw/structures.jsonl")
OUTPUT_DIR = Path("data/processed/graphs")

CUTOFF_RADIUS = 8.0       # angstroms — standard CGCNN-style cutoff
MAX_NEIGHBORS = 12        # cap per atom: keeps graphs sparse, sizes consistent
GAUSSIAN_CENTERS = torch.linspace(0, CUTOFF_RADIUS, 41)
GAUSSIAN_WIDTH = 0.2


def gaussian_expand(distances: torch.Tensor) -> torch.Tensor:
    """Each raw distance -> a vector of Gaussian bumps across the cutoff range,
    instead of a single bare float. Richer signal for the network to learn from."""
    diff = distances.unsqueeze(-1) - GAUSSIAN_CENTERS
    return torch.exp(-(diff ** 2) / (GAUSSIAN_WIDTH ** 2))


def structure_to_graph(structure, band_gap: float, formation_energy: float) -> Data:
    
    node_features = torch.tensor([ATOM_FEATURES[site.specie.Z] for site in structure], dtype=torch.float)

    all_neighbors = structure.get_all_neighbors(CUTOFF_RADIUS)  # handles periodic images

    edge_src, edge_dst, edge_dist = [], [], []
    for i, neighbors in enumerate(all_neighbors):
        nearest = sorted(neighbors, key=lambda n: n.nn_distance)[:MAX_NEIGHBORS]
        for n in nearest:
            edge_src.append(i)
            edge_dst.append(n.index)
            edge_dist.append(n.nn_distance)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = gaussian_expand(torch.tensor(edge_dist, dtype=torch.float))

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_band_gap=torch.tensor([band_gap], dtype=torch.float),
        y_formation_energy=torch.tensor([formation_energy], dtype=torch.float),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(STRUCTURES_PATH) as f:
        lines = f.readlines()

    print(f"Total structures to process: {len(lines)}")

    built, skipped, failed = 0, 0, 0
    for line in lines:
        record = json.loads(line, cls=MontyDecoder)  # decodes the pymatgen Structure inline
        material_id = record["material_id"]
        out_path = OUTPUT_DIR / f"{material_id}.pt"

        if out_path.exists():
            skipped += 1
            continue

        try:
            graph = structure_to_graph(
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