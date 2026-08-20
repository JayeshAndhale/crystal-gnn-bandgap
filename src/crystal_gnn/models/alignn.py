"""
ALIGNN: atom graph + line graph interleaved convolution. Adds three-body
angular information (bond-bond angles) on top of CGCNN's distance-only
convolution, by running gated message passing on both graphs each layer,
each refining the other. Reuses CGCNNConv unchanged — it's graph-agnostic,
so the same gated message-passing block works for both the line graph
(bonds updating bonds via angles) and the atom graph (atoms updating atoms
via the now angle-informed bond features).
"""

import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from crystal_gnn.models.cgcnn import CGCNNConv

ATOM_INPUT_DIM = 92   # matches atom_init.json's fixed elemental descriptor size
DIST_EDGE_DIM = 41    # matches build_graphs_alignn.py's distance Gaussian bins
ANGLE_EDGE_DIM = 41   # matches build_graphs_alignn.py's angle Gaussian bins


class ALIGNNLayer(nn.Module):
    """One ALIGNN layer: line-graph conv (bonds updated by neighboring
    bonds + the angle between them), then atom-graph conv (atoms updated
    using the just-refined bond features as edge input, instead of raw
    distances the way CGCNN does it)."""

    def __init__(self, atom_dim: int):
        super().__init__()
        self.line_graph_conv = CGCNNConv(atom_dim, ANGLE_EDGE_DIM)
        self.atom_conv = CGCNNConv(atom_dim, atom_dim)  # edge_dim = atom_dim now, not DIST_EDGE_DIM

    def forward(self, atom_features, bond_features, edge_index,
                line_graph_edge_index, line_graph_edge_attr):
        bond_features = self.line_graph_conv(
            bond_features, line_graph_edge_index, line_graph_edge_attr
        )
        atom_features = self.atom_conv(atom_features, edge_index, bond_features)
        return atom_features, bond_features


class ALIGNN(nn.Module):
    def __init__(self, atom_dim=64, num_layers=3, hidden_dim=128):
        super().__init__()
        self.atom_embedding = nn.Linear(ATOM_INPUT_DIM, atom_dim)
        self.bond_embedding = nn.Linear(DIST_EDGE_DIM, atom_dim)  # project raw distances into atom_dim so bonds can act as "nodes" in the line graph step

        self.layers = nn.ModuleList(
            [ALIGNNLayer(atom_dim) for _ in range(num_layers)]
        )

        self.mlp = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.Softplus(),
        )
        self.head_formation_energy = nn.Linear(hidden_dim, 1)
        self.head_band_gap = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        atom_features = self.atom_embedding(data.x)
        bond_features = self.bond_embedding(data.edge_attr)

        for layer in self.layers:
            atom_features, bond_features = layer(
                atom_features, bond_features, data.edge_index,
                data.line_graph_edge_index, data.line_graph_edge_attr,
            )

        pooled = global_mean_pool(atom_features, data.batch)  # same per-crystal pooling as CGCNN
        features = self.mlp(pooled)

        return self.head_formation_energy(features), self.head_band_gap(features)