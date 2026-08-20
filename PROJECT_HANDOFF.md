# crystal-gnn-bandgap — Project Handoff

## Overview
Crystal graph neural network for formation energy + band gap prediction,
replicating and extending CGCNN (Xie & Grossman, 2018) on Materials Project
data. Portfolio project — see repo README for full framing.

**Repo:** github.com/JayeshAndhale/crystal-gnn-bandgap
**Local dev:** VS Code, Python 3.12, venv (`.venv/`)
**GPU training:** Kaggle Notebooks (T4 — NOT P100, see Known Issues)
**Compute split:** all data pipeline steps (MP API pull, structure fetch,
graph construction) run locally, CPU-only. Model training runs on Kaggle GPU.
Repo is the source of truth; Kaggle notebooks are disposable execution
environments and should not be relied on to persist anything.

---

## Tier 1: Data Acquisition — COMPLETE

### Pipeline
1. **`scripts/pull_mp_data.py`** — queries Materials Project (`mp-api`) for
   lightweight fields only (`material_id`, `formula_pretty`, `band_gap`,
   `formation_energy_per_atom`, `energy_above_hull`). Filters:
   `energy_above_hull` in (0, 0.05) (standard stability cutoff — structures
   above this are unlikely to be experimentally realizable), `nelements >= 2`
   (excludes elemental phases — trivial ~0 formation energy, no interesting
   bonding chemistry). Randomly samples down to a target size (currently
   25,000; was 5,000 in earlier runs — see Tier 2 for why this changed).
   Output: `data/raw/candidate_pool.json`.
2. **`scripts/pull_structures.py`** — fetches full `pymatgen.Structure`
   objects for the sampled material IDs. Chunked (200/batch) and resumable —
   checks what's already in the output file and only fetches what's missing,
   so a dropped connection or a later re-sample only costs the incremental
   difference. Uses `MontyEncoder`/`MontyDecoder` to serialize pymatgen
   objects to JSON. Output: `data/raw/structures.jsonl` (JSON Lines — one
   record per line, so a partial write is still readable up to the last
   complete line).
3. **`scripts/build_graphs.py`** — converts each structure into a
   `torch_geometric.data.Data` graph object:
   - Nodes = atoms. Node feature (`x`) = CGCNN's standard 92-dim fixed
     elemental descriptor vector (`src/crystal_gnn/models/atom_init.json`,
     sourced from the original CGCNN repo — group, period, electronegativity,
     covalent radius, valence electrons, ionization energy, electron
     affinity, block, atomic volume, each binned + one-hot encoded).
   - Edges = periodic-neighbor bonds within an 8Å cutoff (via
     `pymatgen`'s `get_all_neighbors`, which correctly handles periodic
     images — an atom's nearest neighbor may be in an adjacent unit cell
     copy, not just the atoms listed in the base cell). Capped at 12
     neighbors/atom.
   - Edge feature (`edge_attr`) = interatomic distance, Gaussian-expanded
     into a 41-bin vector (smoother signal than a bare scalar distance).
   - Resumable (skip-if-file-exists per material ID).
   - Output: one `.pt` file per material in `data/processed/graphs/`.

### Current dataset state
- 25,125 materials (25,000 sampled + 125 incidental leftover from an earlier
  5,000-sample pull — see note below)
- Band gap range 0–9.29 eV, 44.0% metals (`band_gap == 0`)
- All graphs built with 92-dim enriched atom features (see Tier 2 ablation —
  an earlier version used bare atomic number as a `nn.Embedding` lookup;
  fully superseded, do not use)
- Uploaded to Kaggle as Dataset `crystal-gnn-graphs`. **Correct folder:
  `graphs_newfeatures`** (92-dim format). Other folders in that dataset
  (`graphs`, `graphs.zip`, `graphs_newfeatures.zip`) are stale/bare-format —
  ignore or delete.

**Note on 25,125 vs 25,000:** `random.sample()` with a fixed seed doesn't
produce a superset when the sample size increases — resampling at 25,000
drew a different set than the original 5,000-sample draw, so 125 materials
from the original pull aren't in the current `candidate_pool.json` but are
still present in `structures.jsonl`/`graphs/` from before (Pass 2/3's
skip-if-exists logic doesn't prune stale entries). Harmless — all are valid,
stability-filtered materials — but worth knowing the exact count doesn't
match the nominal sample size.

### Known issues resolved
- **`.gitignore` was missing a `data/` line for most of the project's
  history.** ~25k+ files and multiple large zips (600MB+ total) got
  committed to git before this was caught (discovered when a `git push`
  timed out). Fixed by adding `data/` to `.gitignore`, then rewriting git
  history with `git filter-repo --path data/ --invert-paths` (run on a
  `git clone --no-local` fresh copy — filter-repo refuses to run in-place
  or on hardlinked local clones), then force-pushing the cleaned history.
  `.git` folder went from 600MB+ to 3.9MB.
  **Lesson: run `git status` before every `git add`, especially right after
  creating any zip or large output file — don't reflexively `git add .`.**
- Homebrew Python 3.12 on the local dev machine has a `.pth`-file processing
  bug that silently breaks editable installs (`pip install -e .` reports
  success, `pip show` confirms correct metadata, but `import crystal_gnn`
  still fails with `ModuleNotFoundError`). Root cause never fully identified;
  workaround is exporting `PYTHONPATH` directly in `.venv/bin/activate`
  (machine-local fix, not committed to git, doesn't affect Kaggle — Kaggle's
  standard Linux Python environment doesn't have this issue).

---

## Tier 2: CGCNN Baseline — COMPLETE

### Model
`src/crystal_gnn/models/cgcnn.py` — gated message-passing convolution
(`CGCNNConv`, subclassing `torch_geometric.nn.MessagePassing`), 3 stacked
conv layers, atom_dim=64, edge_dim=41. Node input: 92-dim descriptor →
learned `nn.Linear(92, 64)` projection. Multi-task output: separate
`formation_energy` and `band_gap` prediction heads sharing the same
convolutional backbone, pooled per-crystal via `global_mean_pool` (batching
handled via PyG's `Batch.from_data_list` / `DataLoader`, which stacks
variable-atom-count graphs into one batch with a `batch` index tensor).

### Training setup
- `scripts/train_cgcnn.py`: 80/10/10 train/val/test split, `SEED=42` used
  for BOTH `random.seed()` (controls the data split) AND
  `torch.manual_seed()` (controls model weight init + DataLoader shuffling)
  — both are required for full run-to-run reproducibility; only seeding the
  former (an earlier bug) meant reruns produced different models despite
  identical data splits.
- Target normalization (mean/std, from train split only) applied before
  loss computation, saved inside the checkpoint dict, reversed at eval time.
- Adam optimizer, `weight_decay=1e-5`, `ReduceLROnPlateau` scheduler
  (`factor=0.5`, `patience=8`) — added after observing early runs plateau
  with noisy/spiky val loss, consistent with a fixed LR overshooting near
  convergence.
- `scripts/evaluate_cgcnn.py`: loads best checkpoint, runs on the held-out
  test set only (never touched during training or checkpoint selection),
  reports real-unit MAE (eV/atom, eV) — directly comparable to published
  benchmarks, unlike the normalized training-loss numbers.

### Results (4 runs, chronological — see git log for exact commits)

| Run | Train size | Atom features | Scheduler | Best val loss (epoch) | Test FE MAE | Test BG MAE |
|---|---|---|---|---|---|---|
| 1 | 4,000 | bare atomic number | no | 0.2058 (49) | 0.1066 eV/atom | 0.4664 eV |
| 2 | 4,000 | 92-dim enriched | no | 0.2279 (60) | 0.1234 eV/atom | 0.5175 eV |
| 3 | 20,100 | 92-dim enriched | no | 0.1480 (70) | 0.0689 eV/atom | 0.3831 eV |
| **4 (final)** | 20,100 | 92-dim enriched | **yes** | **0.1362 (89)** | **0.0639 eV/atom** | **0.3712 eV** |

**Published CGCNN benchmark** (Xie & Grossman 2018, ~28k–46k structures):
FE MAE ~0.039 eV/atom, BG MAE ~0.29–0.39 eV. Run 4's band gap MAE falls
within the published range; formation energy MAE is ~1.6x published, on
roughly half the training data — a defensible, directionally-honest result
given the data-scale gap.

### Key findings (ablation logic, in order — worth walking through end to end)
1. **Run 1→2 (4k data, bare→enriched atom features): regressed** (FE MAE
   0.107→0.123, BG MAE 0.466→0.518). Enriching node features with real
   elemental chemistry made things *worse* at small data scale.
2. **Diagnosis:** both runs 1 and 2 show a widening train/val loss gap
   throughout training (visible directly in the per-epoch logs) — classic
   data-limited overfitting. The feature representation wasn't the
   bottleneck; dataset size was. Richer input features had no data volume
   to actually pay off on.
3. **Run 2→3 (4k→20.1k data, features held constant): FE MAE 0.123→0.069,
   BG MAE 0.518→0.383.** Confirms the diagnosis — once the data bottleneck
   was addressed, the enriched features combined with more examples to
   substantially close the gap toward published numbers.
4. **Run 3→4 (added LR scheduler + weight decay): FE MAE 0.069→0.064,
   BG MAE 0.383→0.371.** Directly visible in the training log: each LR drop
   (epochs 62, 87, 98) immediately reduces val-loss noise and unlocks a new
   best, confirming the causal mechanism rather than coincidental
   improvement — training was overshooting near convergence at the fixed
   initial LR.

### Checkpoint
Run 4's checkpoint (`checkpoints/cgcnn_best.pt`, epoch 89, val loss 0.1362)
is **not committed to git** — binary weights, deliberately gitignored (see
`checkpoints/` in `.gitignore`). Kept locally only, downloaded from Kaggle
after training. To reproduce from scratch: