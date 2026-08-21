# CGCNN Interpretability: Bond Gate Analysis

Extracted from `checkpoints/cgcnn_best.pt` via forward hooks on each 
`CGCNNConv.gate_net` -- no retraining. Gate = `sigmoid(gate_net([x_i, x_j, edge_attr]))`, 
a 64-dim vector per bond per layer; reported as its mean over channels, i.e. 
how strongly that bond's message was let through, roughly "attention weight".

Materials analyzed: 8 (all from the held-out test split, 
SEED=42 -- never seen in training or checkpoint selection).
Total bonds analyzed: 2724

## Findings

**Gate strength vs. bond distance:** Pearson r = -0.379 (p = 1.16e-93).
Negative correlation -- shorter bonds tend to get stronger gates, consistent with shorter bonds generally being the chemically stronger/more relevant interaction.

**Hetero-nuclear vs. homo-nuclear bonds:** mean gate 0.449 (n=1391) vs. 0.417 (n=1333).
Welch's t-test: t = 13.04, p = 1.00e-37.
Hetero-nuclear bonds (different elements -- typically the real cation-anion/covalent bonding interaction in a crystal) are gated significantly higher than homo-nuclear bonds. This is chemically sensible: the model is not treating all short contacts equally, it is preferentially weighting the bonds that carry the actual bonding chemistry.

## Per-material detail

See `top_bonds_per_material.png` for the 8 highest-gated bonds in each material examined, and `gate_vs_distance.png` for the full distance/gate scatter across all bonds. Raw per-bond values are in `gate_values.csv`.

## Caveat

This is a qualitative/statistical read of gate values on a handful of test materials, not a claim about global model behavior -- worth stating plainly if asked in an interview. A stronger next step (not done here) would be a full-test-set aggregate broken out by bond type, or ablating specific bonds and measuring the prediction shift directly.
