# Crystal Graph Neural Networks for Materials Property Prediction

Predicting **formation energy** and **band gap** from crystal structure, with a
direct, controlled comparison of two graph neural network architectures —
**CGCNN** (Xie & Grossman, 2018) and **ALIGNN** — on identical data, followed
by ensembling, uncertainty quantification, interpretability, and an active
learning simulation built on top of the winning setup.

25,125 stability-filtered materials from the [Materials Project](https://materialsproject.org/),
pulled via the official API. Full technical write-up, including every
ablation, infrastructure decision, and exact reproduction commands, is in
[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md).

## Why two architectures

CGCNN represents a crystal as a graph where atoms are nodes and bonds are
edges, with interatomic **distance** as the only edge feature. That's a real,
identifiable limitation: two chemically distinct coordination geometries
(e.g. tetrahedral vs. square-planar) with similar bond lengths look identical
to it, because it has no way to represent bond **angle**. ALIGNN fixes this
with a line-graph construction — each bond becomes a node in a second graph,
connected to bonds it shares an atom with, with the angle between them as the
edge feature — adding the three-body information CGCNN structurally cannot
see.

## Headline results

Test-set MAE, identical 20,100 / 2,512 / 2,513 train/val/test split for every
model below (never touched during training or checkpoint selection):

| Model | Formation energy (eV/atom) | Band gap (eV) |
|---|---|---|
| CGCNN | 0.0639 | 0.3712 |
| ALIGNN | 0.0689 | 0.3572 |
| **5-model CGCNN ensemble** | **0.0509** | **0.3558** |
| Published CGCNN benchmark | ~0.039 | ~0.29–0.39 |
| Published ALIGNN benchmark | ~0.022 | ~0.218 |

## Findings

**1. CGCNN baseline — ablation, not a single black-box run.** Enriching atom
features from bare atomic number to a 92-dim chemical descriptor *hurt*
accuracy at 4,000 training materials — a real overfitting problem, confirmed
by scaling to 20,100 materials (where the same richer features then helped
substantially). An LR scheduler squeezed out a further improvement,
visible as literal inflection points in the loss curve at each LR drop.

**2. CGCNN vs. ALIGNN — a mixed result, deliberately not oversold.** Band gap
improved ~3.8% with ALIGNN (physically sensible: band gap depends on local
coordination geometry, exactly what bond angles capture) while formation
energy regressed ~7.8% (ALIGNN has more capacity per layer but was run with
the same untuned layer count as CGCNN, and formation energy is a more
"global" property that benefits less from local angular precision). The
architectural change helped the property it should mechanistically help, and
didn't automatically help the one it has a weaker link to.

**3. A 5-model ensemble beat the architecture change.** Simple ensembling
closed more of the formation-energy gap to the published benchmark (1.6x →
1.3x) than either the LR scheduler fix or switching to ALIGNN did, and its
band gap MAE edges out the single ALIGNN run — with zero architecture
changes.

**4. The ensemble's uncertainty is trustworthy for one target, not the other.**
Ensemble disagreement correlates with actual error for both properties, but
only formation energy's predicted spread is well-*calibrated in magnitude*
(predicted-std-to-actual-error ratio ≈0.95, close to the ~0.80 Gaussian
ideal). Band gap's spread is real signal for *ranking* which materials are
harder (r=0.49) but is ~2x underconfident in absolute magnitude — a
meaningful distinction if the uncertainty estimate is ever used for
anything beyond ranking.

**5. Interpretability: CGCNN's learned attention is chemically sensible.**
Extracting the model's per-bond gate values (no retraining, just forward
hooks) shows it weights shorter bonds and real hetero-atomic bonding
interactions (cation–anion, covalent) significantly higher than same-element
contacts (p≈1e-37) — it isn't a black box weighting all short contacts
equally.

**6. Active learning: a calibration prediction, confirmed by two symmetric
experiments.** Acquiring the next batch of labeled materials by ensemble
uncertainty (rather than randomly) was run twice — once targeting band gap
uncertainty, once targeting formation energy's. Band-gap-targeted acquisition
consistently beat random sampling from 5,000 labels onward, reaching random's
final-round accuracy using **31% fewer labels**. Formation-energy-targeted
acquisition did *not* cleanly beat random (only ~3% savings, never reaching
the final target) — exactly what the calibration finding above predicts,
since FE's ensemble std ranks material difficulty far less reliably (r=0.23)
than BG's (r=0.49). This isn't "active learning worked" — it's a mechanism
predicted in advance and confirmed in both directions.

## Repo structure

```
scripts/
  pull_mp_data.py, pull_structures.py     # Materials Project data pipeline
  build_graphs.py, build_graphs_alignn.py # graph construction (CGCNN / ALIGNN)
  train_cgcnn.py, train_alignn.py         # single-model training
  evaluate_cgcnn.py, evaluate_alignn.py   # test-set evaluation
  train_ensemble.py, evaluate_ensemble.py # deep ensemble + calibration
  interpret_cgcnn.py                      # bond-gate interpretability
  active_learning.py, plot_active_learning.py
src/crystal_gnn/models/                   # CGCNN and ALIGNN implementations
results/                                  # plots, CSVs, write-ups per tier
```

## Reproducing

```bash
pip install -e .
python scripts/train_cgcnn.py --graphs_dir <path-to-graphs> --epochs 100
python scripts/evaluate_cgcnn.py --graphs_dir <path-to-graphs>
```

Exact commands for every result above (including the ALIGNN, ensemble, and
active-learning runs) are in [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md).

## Stack

PyTorch, PyTorch Geometric, pymatgen, the Materials Project API. Training on
Kaggle T4 GPUs.
