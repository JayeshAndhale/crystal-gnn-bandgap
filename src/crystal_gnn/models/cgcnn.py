"""
CGCNN: crystal graph convolutional network. Gate-and-message convolution
over periodic-neighbor graphs, multi-task output (formation energy + band gap).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


class CGCNNConv(MessagePassing):
    """One convolution layer: gated message passing over bonds."""

    def __init__(self, atom_dim: int, edge_dim: int):
        super().__init__(aggr="add")  # sum messages onto each destination atom
        combined_dim = 2 * atom_dim + edge_dim
        self.gate_net = nn.Linear(combined_dim, atom_dim)
        self.core_net = nn.Linear(combined_dim, atom_dim)
        self.bn = nn.BatchNorm1d(atom_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i = destination atom vector, x_j = source (neighbor) atom vector
        combined = torch.cat([x_i, x_j, edge_attr], dim=-1)
        gate = torch.sigmoid(self.gate_net(combined))
        core = nn.functional.softplus(self.core_net(combined))
        return gate * core

    def update(self, aggregated, x):
        return self.bn(x + aggregated)  # residual: add messages onto current vector


class CGCNN(nn.Module):
    def __init__(self, atom_dim=64, edge_dim=41, num_conv_layers=3, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Linear(92, atom_dim)
        self.conv_layers = nn.ModuleList(
            [CGCNNConv(atom_dim, edge_dim) for _ in range(num_conv_layers)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.Softplus(),
        )
        self.head_formation_energy = nn.Linear(hidden_dim, 1)
        self.head_band_gap = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = self.embedding(data.x)
        for conv in self.conv_layers:
            x = conv(x, data.edge_index, data.edge_attr)

        pooled = global_mean_pool(x, data.batch)  # per-crystal average, not global
        features = self.mlp(pooled)

        return self.head_formation_energy(features), self.head_band_gap(features)