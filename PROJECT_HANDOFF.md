# crystal-gnn-bandgap — Project Handoff

## Overview
Crystal graph neural network project for formation energy + band gap
prediction on Materials Project data. Two architectures compared on
identical data: CGCNN (Xie & Grossman, 2018 — distance-only edges) and
ALIGNN (line-graph representation — adds bond-angle/three-body information).
Portfolio project — see repo README for full framing.

**Repo:** github.com/JayeshAndhale/crystal-gnn-bandgap
**Local dev:** VS Code, Python 3.12, venv (`.venv/`)
**GPU training:** Kaggle Notebooks (T4 — NOT P100, see Known Issues)
**Compute split:** data pipeline steps (MP API pull, structure fetch, graph
construction) run locally CPU-only where disk allows; large graph builds
(ALIGNN's full 25k set) run on Kaggle directly — see Tier 3. Repo is the
source of truth; Kaggle notebooks/sessions are disposable and NOT reliably
persistent (see Known Issues) — nothing should be trusted to survive in
`/kaggle/tmp` or an interactive session without being explicitly pushed
somewhere permanent (a Kaggle Dataset, or downloaded).

---

## Tier 1: Data Acquisition — COMPLETE

### Pipeline
1. **`scripts/pull_mp_data.py`** — queries Materials Project (`mp-api`) for
   lightweight fields (`material_id`, `formula_pretty`, `band_gap`,
   `formation_energy_per_atom`, `energy_above_hull`). Filters:
   `energy_above_hull` in (0, 0.05) (standard stability cutoff), `nelements
   >= 2` (excludes elemental phases — trivial ~0 formation energy). Randomly
   samples down to `SAMPLE_SIZE` (currently 25,000). Output:
   `data/raw/candidate_pool.json`.
2. **`scripts/pull_structures.py`** — fetches full `pymatgen.Structure`
   objects for sampled IDs. Chunked (200/batch), resumable (skips IDs
   already in the output file). Uses `MontyEncoder`/`MontyDecoder` to
   serialize pymatgen objects to JSON. Output: `data/raw/structures.jsonl`
   (JSON Lines — partial writes stay readable up to the last complete line).
3. **`scripts/build_graphs.py`** (CGCNN graphs) — converts each structure
   into a `torch_geometric.data.Data` object:
   - Nodes = atoms. Node feature (`x`) = CGCNN's standard 92-dim fixed
     elemental descriptor vector (`src/crystal_gnn/models/atom_init.json`,
     from the original CGCNN repo).
   - Edges = periodic-neighbor bonds within an 8Å cutoff (`pymatgen`'s
     `get_all_neighbors`, correctly handles periodic images). Capped at 12
     neighbors/atom.
   - Edge feature = interatomic distance, Gaussian-expanded (41 bins).
   - Resumable (skip-if-file-exists per material ID).
   - Output: one `.pt` file per material in `data/processed/graphs/`.

### Current dataset state
- 25,125 materials total (25,000 sampled + 125 incidental leftover from an
  earlier 5,000-sample pull — `random.sample()` at a different size draws a
  different set, not a superset; harmless, all valid stability-filtered
  materials, just don't expect the count to match the nominal sample size
  exactly)
- Band gap range 0–9.29 eV, 44.0% metals (`band_gap == 0`)
- CGCNN graphs (25,125, 92-dim atom features) uploaded to Kaggle Dataset
  `crystal-gnn-graphs`, folder **`graphs_newfeatures`** (correct format —
  other folders in that dataset are stale bare-atomic-number format, ignore)

### Known issues resolved
- **`.gitignore` was missing a `data/` line for most of the project's early
  history.** ~25k+ files and multiple large zips (600MB+) got committed to
  git before this was caught (a `git push` timed out). Fixed by adding
  `data/` to `.gitignore`, then `git filter-repo --path data/
  --invert-paths` on a `git clone --no-local` fresh copy (filter-repo
  refuses to run in-place or on hardlinked local clones), then force-pushed
  the cleaned history. `.git` went from 600MB+ → 3.9MB.
  **Lesson: run `git status` before every `git add`, especially right after
  creating any zip or large output file.**
- Homebrew Python 3.12 on the local dev machine has a `.pth`-file processing
  bug that silently breaks editable installs — `pip install -e .` reports
  success, `pip show` confirms correct metadata, but `import crystal_gnn`
  still fails with `ModuleNotFoundError`. Root cause not fully identified;
  workaround is `PYTHONPATH` exported directly in `.venv/bin/activate`
  (machine-local, not committed to git — Kaggle's standard Linux Python
  environment doesn't have this bug).

---

## Tier 2: CGCNN Baseline — COMPLETE

### Model
`src/crystal_gnn/models/cgcnn.py` — gated message-passing convolution
(`CGCNNConv`, subclasses `torch_geometric.nn.MessagePassing`), 3 stacked
layers, atom_dim=64, edge_dim=41. Node input: 92-dim descriptor → learned
`nn.Linear(92, 64)`. Multi-task: separate `formation_energy`/`band_gap`
heads sharing the conv backbone, pooled per-crystal via `global_mean_pool`.

### Training setup
- `scripts/train_cgcnn.py`: 80/10/10 split, `SEED=42` used for BOTH
  `random.seed()` (data split) AND `torch.manual_seed()` (model init +
  DataLoader shuffling) — both required for reproducibility; an early bug
  only seeded the former, so reruns produced different models despite
  identical splits.
- Target normalization (train-split mean/std only), saved in checkpoint,
  reversed at eval time.
- Adam, `weight_decay=1e-5`, `ReduceLROnPlateau` (`factor=0.5`,
  `patience=8`) — added after early runs showed noisy/spiky val loss near
  convergence, consistent with a fixed LR overshooting.
- `scripts/evaluate_cgcnn.py`: test-set-only real-unit MAE (eV/atom, eV).

### Results (4 runs, chronological)

| Run | Train size | Atom features | Scheduler | Best val loss (epoch) | Test FE MAE | Test BG MAE |
|---|---|---|---|---|---|---|
| 1 | 4,000 | bare atomic number | no | 0.2058 (49) | 0.1066 eV/atom | 0.4664 eV |
| 2 | 4,000 | 92-dim enriched | no | 0.2279 (60) | 0.1234 eV/atom | 0.5175 eV |
| 3 | 20,100 | 92-dim enriched | no | 0.1480 (70) | 0.0689 eV/atom | 0.3831 eV |
| **4 (final)** | 20,100 | 92-dim enriched | **yes** | **0.1362 (89)** | **0.0639 eV/atom** | **0.3712 eV** |

**Published CGCNN** (~28-46k structures): FE MAE ~0.039 eV/atom, BG MAE
~0.29-0.39 eV. Run 4's BG MAE falls within the published range; FE MAE is
~1.6x published, on roughly half the training data.

### Key findings (ablation, in order — walk through end to end in interviews)
1. **Run 1→2 (4k data, bare→enriched atom features): regressed** (FE MAE
   0.107→0.123, BG MAE 0.466→0.518).
2. **Diagnosis:** both runs show a widening train/val loss gap — data-limited
   overfitting, not a feature-quality problem. Richer features had no data
   volume to pay off on yet.
3. **Run 2→3 (4k→20.1k data, features held constant): FE MAE 0.123→0.069,
   BG MAE 0.518→0.383.** Confirms the diagnosis.
4. **Run 3→4 (added scheduler): FE MAE 0.069→0.064, BG MAE 0.383→0.371.**
   Directly visible in the log: each LR drop (epochs 62, 87, 98) immediately
   reduces val-loss noise and unlocks a new best.

### Checkpoint
`checkpoints/cgcnn_best.pt` (epoch 89, val loss 0.1362) — **not committed to
git** (gitignored binary weights). Kept locally, downloaded from Kaggle. To
reproduce:
python scripts/train_cgcnn.py
--graphs_dir <path to graphs_newfeatures>
--epochs 100 --batch_size 64



Deterministic given `SEED=42` — confirmed via matching checkpoint
epoch/loss across independent reruns.

---

## Tier 3: ALIGNN — COMPLETE

### Concept
CGCNN's edges only encode bond *distance* — a two-body term. It has no way
to represent bond *angle* (a three-body term: two bonds sharing an atom),
so two chemically distinct coordination geometries (e.g. tetrahedral vs.
square-planar) with similar bond lengths are indistinguishable to it.
ALIGNN fixes this via a **line graph**: each bond becomes a node, and two
bond-nodes connect if they share an atom, with the angle between them as
the edge feature. Each ALIGNN layer runs gated message passing twice: once
on the line graph (bonds refined by neighboring bonds + angle), once on the
atom graph (atoms refined using the just-updated bond features instead of
raw distances). `CGCNNConv` is reused unchanged for both passes — it's
graph-agnostic, just takes node vectors + edge index + edge features.

### Graph construction
`scripts/build_graphs_alignn.py` — atom-graph part identical to CGCNN's
build; additionally, for every atom, computes the angle between every pair
of its outgoing bonds (via neighbor coordinates already periodic-corrected
by pymatgen), Gaussian-expands each angle (41 bins over [0, π]), and builds
the corresponding line-graph edge structure. Custom `ALIGNNData(Data)`
subclass overrides `__inc__` so PyG's batching offsets
`line_graph_edge_index` by edge count, not atom count (default batching
would silently produce wrong indices otherwise — no crash, just corrupted
data). CLI args: `--structures_path`, `--output_dir`, `--start`, `--limit`
(the latter two added specifically to support chunked building — see
Infrastructure Lessons below).

### Model
`src/crystal_gnn/models/alignn.py` — `ALIGNNLayer` wraps two `CGCNNConv`
calls (line-graph pass then atom-graph pass); `ALIGNN` stacks 3 such layers,
with separate linear projections for initial atom (92→64) and bond (41→64)
embeddings so bond representations can act as edge features in the
atom-graph pass. Same pooling/multi-task-head/training-loop pattern as
CGCNN throughout.

### Dataset scale — infrastructure story worth knowing
Line-graph edges scale ~quadratically with each atom's neighbor count
(every *pair* of bonds, not each bond once), making ALIGNN graphs
dramatically larger per-material than CGCNN's (~0.86 MB/material vs.
CGCNN's much smaller footprint). This caused a full day of infrastructure
problems before the real training run:
- Building the full 25,125-material set locally exhausted local disk
  (macOS ran out of space mid-build; had to `rm -rf` a partial 12GB output
  and clean up ~377MB of stale zip files from earlier stages).
- Settled on building on **Kaggle directly** instead (never touching local
  disk), using `/kaggle/tmp` (~60GB, non-persistent) for scratch space and
  the Kaggle API (`kaggle datasets create`) to push the result to a
  permanent Kaggle Dataset before the session could lose it.
- First attempt at pushing all 25,125 files via `--dir-mode zip` actually
  uploaded file-by-file (25k individual HTTP requests, ~1-2s each — would
  have taken 7-10+ hours) rather than as one archive; cancelling that run
  triggered a full session reset that wiped `/kaggle/tmp` entirely,
  including the completed graph build. Total loss, redone.
- **Final working approach: chunked building.** Built in 6 chunks of 5,000
  materials (`--start`/`--limit`), each chunk zipped locally on Kaggle
  *before* upload (one archive per chunk, not per-file), pushed as its own
  small Kaggle Dataset (`crystal-gnn-alignn-chunk-0` through `-5`), and
  deleted from Kaggle's disk immediately after each successful upload was
  confirmed — so a failure anywhere costs at most one chunk, not the whole
  build. Validated on a small 2×300 test run before committing real time to
  the full 6×5000 run. Ran as a committed background job (Save Version →
  Save & Run All) rather than an interactive session, given the earlier
  disconnect/reset scare.
- Training then loads all 6 chunks together via a `load_all_graphs()` that
  accepts a comma-separated list of directories (present in both
  `train_alignn.py` and `evaluate_alignn.py` — **this fix was written
  conceptually once but only actually landed in the file after being
  re-verified explicitly; the same "discussed but never committed" gap bit
  both scripts independently and each needed its own separate fix/verify
  cycle**).

**Final dataset:** `crystal-gnn-alignn-chunk-0` through `-5` on Kaggle
Datasets, 5,000/5,000/5,000/5,000/5,000/125 files respectively (25,125
total, confirmed both by file count and manual spot-check of one chunk's
Data Card).

### Training setup
`scripts/train_alignn.py` / `scripts/evaluate_alignn.py` — identical
mechanics to CGCNN's scripts (same split logic, `SEED=42` for both random
sources, normalization, `ReduceLROnPlateau`, weight decay, checkpointing).
Only differences: imports `ALIGNN` instead of `CGCNN`, imports `ALIGNNData`
before loading graphs (required for correct unpickling/batching),
checkpoint filename `alignn_best.pt` (distinct from CGCNN's, so both can
coexist in the same session).

### Results — direct comparison, same 20,100/2,512/2,513 split

| Model | Train size | Best val loss (epoch) | Test FE MAE | Test BG MAE |
|---|---|---|---|---|
| CGCNN (final) | 20,100 | 0.1362 (89) | 0.0639 eV/atom | 0.3712 eV |
| **ALIGNN (final)** | 20,100 | 0.1388 (65) | 0.0689 eV/atom | **0.3572 eV** |
| Published CGCNN | ~28-46k | — | ~0.039 | ~0.29-0.39 |
| Published ALIGNN | ~28-46k | — | ~0.022 | ~0.218 |

*(An earlier ALIGNN run on only 4,000 materials — before the chunking
infrastructure was built — landed at FE MAE 0.1038 / BG MAE 0.5020,
confounded by also having the scheduler that CGCNN's equivalent 4k run
didn't have; not a clean comparison point, superseded by the 20.1k result
above.)*

### Key finding — mixed, honest result, not a clean win
- **Band gap improved**: 0.3712 → 0.3572 (~3.8% better). Physically
  sensible: band gap depends on local coordination geometry/orbital
  overlap, exactly what bond angles capture and CGCNN's distance-only
  edges can't represent.
- **Formation energy got slightly worse**: 0.0639 → 0.0689 (~7.8% worse).
  Likely explanation: ALIGNN has meaningfully more capacity per layer
  (two message-passing passes vs. CGCNN's one) but was run with the same
  layer count/hidden dims as CGCNN, untuned for the new architecture — and
  its val loss plateaued earlier (epoch 65 vs. CGCNN's 89), consistent with
  hitting a capacity/data ceiling sooner. Formation energy is also a more
  "global" structural property, arguably benefiting less from precise local
  angular information than band gap does.
- **Interview framing**: three-body angular information helps the property
  it should mechanistically help (band gap), and doesn't automatically help
  the property it has a weaker mechanistic link to (formation energy)
  without further tuning — a defensible, explainable result, not a
  wash or a failure.

### Checkpoint
`checkpoints/alignn_best.pt` (epoch 65, val loss 0.1388) — not committed to
git, kept locally. To reproduce:
python scripts/train_alignn.py
--graphs_dir <chunk-0-path>,<chunk-1-path>,...,<chunk-5-path>
--epochs 100 --batch_size 64


(all 6 chunk directory paths, comma-separated, no spaces)

---

## Tier 3.1: Interpretability — COMPLETE

### Method
`scripts/interpret_cgcnn.py` — no retraining. Registers a forward hook on
each `CGCNNConv.gate_net` (the `nn.Linear` whose sigmoid output is the
per-bond gate in `CGCNNConv.message`), runs `cgcnn_best.pt` forward on 8
materials sampled from the held-out test split (`SAMPLE_SEED=123`, disjoint
from the training/eval split logic), and reduces each bond's 64-dim gate
vector to a scalar (mean over channels) per layer. Bond identity (element
pair, real distance) is recovered by re-running the exact neighbor-finding
logic from `build_graphs.py` on the raw structure (not by inverting the
Gaussian-expanded `edge_attr`, which isn't invertible) — asserted to produce
the same edge count as the stored graph, so gate values line up with the
right bonds.

### Findings (`results/interpretability/`)
- **Gate strength vs. bond distance:** Pearson r = -0.379 (p ≈ 1.2e-93,
  n=2724 bonds across 8 materials) — shorter bonds get stronger gates.
- **Hetero- vs. homo-nuclear bonds:** mean gate 0.449 (hetero, n=1391) vs.
  0.417 (homo, n=1333); Welch's t-test p ≈ 1.0e-37. The model gates real
  cation-anion/covalent bonding interactions higher than same-element
  contacts — a chemically sensible signal, not noise.
- Per-material breakdown (`top_bonds_per_material.png`) shows real structure,
  not just distance-sorting: e.g. mp-6342 (LiCa₃RuO₆) has O–Ru @ 1.98Å as
  its clearly highest-gated bond type, well above the Ca–Ca/Ca–Li contacts.

---

## Tier 3.2: Ensemble Uncertainty Quantification — COMPLETE

### Method
`scripts/train_ensemble.py` / `scripts/evaluate_ensemble.py` — a 5-member
CGCNN deep ensemble, one fixed train/val/test split shared by every member
(`SPLIT_SEED=42`, identical to the baseline), only differing by
`torch.manual_seed(member_seed)` for model init + minibatch shuffling
(seeds 1–5, deliberately distinct from `SPLIT_SEED` to avoid confusing the
two). Trained on Kaggle T4 GPU, 100 epochs each, same optimizer/scheduler
as the baseline. Evaluation computes ensemble-mean predictions (average
over members) and ensemble std (spread across members, the uncertainty
estimate), then checks calibration: does higher spread actually predict
higher error (spread-skill plots + Pearson correlation), not just whether
the mean prediction improved.

### Results (`results/uncertainty_cgcnn/`, full 25,125-material split:
20,100/2,512/2,513)

| | Mean single-member MAE | Ensemble-mean MAE |
|---|---|---|
| Formation energy | 0.0685 eV/atom | **0.0509 eV/atom** |
| Band gap | 0.3861 eV | **0.3558 eV** |

Ensembling alone closed more of the formation-energy gap to the published
CGCNN benchmark (~0.039) than either the scheduler fix or switching to
ALIGNN did — FE MAE dropped from ~1.6x published (single CGCNN) to ~1.3x
published, with zero architecture changes. Ensemble-mean band gap MAE
(0.3558) is now marginally *better* than the single ALIGNN run (0.3572).

**Calibration — a genuinely mixed, honest result:**
- **Formation energy** is well-calibrated in magnitude, not just direction:
  corr(predicted std, actual |error|) = 0.228 (p ≈ 5.6e-31); miscalibration
  ratio (mean|error| / mean std) = 0.946, close to the ~0.80 ideal for
  Gaussian residuals. The spread-skill plot tracks close to y=x.
- **Band gap** has a *stronger* correlation (r = 0.490, p ≈ 7.0e-152— the
  ensemble is better at ranking *which* materials it'll get wrong) but is
  substantially **underconfident in magnitude**: miscalibration ratio 2.063
  — actual error is on average 2x the predicted std. The spread-skill plot
  sits well below y=x across the entire range, not just at outliers.
- Interview framing: the ensemble's disagreement is real signal for both
  targets (it knows *which* materials are harder), but only formation
  energy's spread is trustworthy as a *magnitude* estimate of "how wrong" —
  band gap uncertainty would need recalibration (e.g. temperature scaling)
  before using it for anything like active learning acquisition, which
  matters directly for Tier 3.3 if pursued.

### Checkpoints
`checkpoints/ensemble_cgcnn/member_seed{1..5}_best.pt` — not committed to
git (gitignored), kept locally, downloaded from the Kaggle Dataset
`jayeshandhale/crystal-gnn-ensemble-cgcnn` (private). To reproduce:
```
python scripts/train_ensemble.py --model cgcnn \
  --graphs_dir <graphs_newfeatures_dir> --epochs 100 --seeds 1,2,3,4,5
python scripts/evaluate_ensemble.py --model cgcnn --graphs_dir <same dir>
```

---

## Tier 3.3: Active Learning Simulation — COMPLETE

### Method
`scripts/active_learning.py` / `scripts/plot_active_learning.py` — a
pool-based active learning simulation, depending directly on Tier 3.2's
ensemble. The held-out test set (same 2,513 materials, `SPLIT_SEED=42`) is
fixed throughout, never touched by training or acquisition. The combined
train+val split (22,612 materials) is the "pool" — a small random 2,000-
material seed set starts labeled, the rest starts unlabeled (labels exist
in the data but are hidden from the acquisition logic).

Each of 6 rounds: train a small fresh 3-member CGCNN ensemble from scratch
(40 epochs, no warm-starting, to avoid confounding round comparisons) on
the current labeled set, evaluate ensemble-mean test MAE, then reveal 3,000
more materials — either the pool's highest-uncertainty materials
("uncertainty" strategy, ranked by one target's ensemble std) or a random
3,000 ("random" baseline). Both strategies share the identical initial
labeled set and round schedule, so the only difference is which materials
get revealed. Run **twice**, once per target (`--acquisition_target bg` and
`--acquisition_target fe`) — deliberately, as a predict-then-confirm test of
Tier 3.2's calibration finding (BG's ensemble std ranks material difficulty
more reliably, r=0.49, than FE's, r=0.23): if that finding is real,
BG-targeted acquisition should beat random cleanly and FE-targeted
acquisition shouldn't.

### Results

**Run 1 — acquiring by band gap uncertainty** (`results/active_learning/`):

| n_labels | uncertainty FE MAE | random FE MAE | uncertainty BG MAE | random BG MAE |
|---|---|---|---|---|
| 2,000 | 0.1212 | 0.1212 | 0.5716 | 0.5716 |
| 5,000 | 0.1133 | 0.0861 | 0.4927 | 0.5051 |
| 8,000 | 0.0870 | 0.0805 | 0.4493 | 0.4574 |
| 11,000 | 0.0711 | 0.0749 | 0.4179 | 0.4487 |
| 14,000 | 0.0837 | 0.0693 | 0.3932 | 0.4176 |
| 17,000 | 0.0668 | 0.0668 | 0.3935 | 0.4055 |
| 20,000 | 0.0627 | 0.0607 | 0.3724 | 0.3946 |

**Run 2 — acquiring by formation energy uncertainty** (`results/active_learning_fe/`):

| n_labels | uncertainty FE MAE | random FE MAE | uncertainty BG MAE | random BG MAE |
|---|---|---|---|---|
| 2,000 | 0.1247 | 0.1247 | 0.5934 | 0.5934 |
| 5,000 | 0.1193 | 0.0877 | 0.5103 | 0.5122 |
| 8,000 | 0.0835 | 0.0777 | 0.4511 | 0.4802 |
| 11,000 | 0.0800 | 0.0686 | 0.4176 | 0.4324 |
| 14,000 | 0.0715 | 0.0641 | 0.4046 | 0.4084 |
| 17,000 | 0.0648 | 0.0660 | 0.3920 | 0.4051 |
| 20,000 | 0.0667 | 0.0597 | 0.3752 | 0.3857 |

**A predicted mechanism, confirmed by two symmetric runs — not two isolated
results:**
- **BG-targeted run:** uncertainty-based acquisition beats random at *every*
  round from 5,000 labels onward. Reaches random's final (20,000-label) BG
  performance using only ~13,831 labels — **31% fewer labels for the same
  accuracy**. FE shows no gain (0% label savings at the one target level
  reached) — unsurprising, since acquisition was never targeting FE.
- **FE-targeted run:** uncertainty-based acquisition does *not* cleanly beat
  random on FE (random wins or ties at 5 of 7 rounds; only ~3% label savings,
  and never reaches the final-round target) — exactly what Tier 3.2's weaker
  FE ranking-correlation (r=0.23) predicts. Band gap, not targeted this run,
  *still* shows a consistent uncertainty-beats-random gap (9–18% fewer
  labels) at nearly every round — plausibly because materials that are
  FE-hard and BG-hard overlap, so FE-uncertainty acquisition incidentally
  picks generally-informative materials.
- Interview framing: this isn't "we tried active learning and it worked" —
  it's "a calibration diagnostic (Tier 3.2) predicted which target's
  uncertainty would support effective acquisition, and two independent runs
  confirmed the prediction in both directions." That's a substantially
  stronger claim than either run alone.

### Checkpoints
`checkpoints/active_learning{,_fe}/{uncertainty,random}/round{0..6}/member{0,1,2}.pt`
plus `round_state.json` per strategy/run (full round-by-round labeled-material
IDs, for exact reproducibility of which materials were acquired when) — not
committed to git, kept locally, downloaded from the Kaggle Datasets
`jayeshandhale/crystal-gnn-active-learning` and
`jayeshandhale/crystal-gnn-active-learning-fe` (private). To reproduce:
```
python scripts/active_learning.py --graphs_dir <graphs_newfeatures_dir> \
  --seed_size 2000 --acquisition_size 3000 --n_rounds 6 \
  --ensemble_size 3 --epochs_per_round 40 --acquisition_target bg
python scripts/plot_active_learning.py --results_dir results/active_learning --acquisition_target bg

# repeat with --acquisition_target fe and a distinct --checkpoint_dir/--output_dir
# for the second run
```

---

## Infrastructure lessons (apply going forward, not just retrospective)
1. **Verify every "should be committed" change with `git log origin/main -1
   --oneline` before trusting a Kaggle session to have it.** Multiple times
   this project, a code change was discussed and even shown as present in
   the local file, but never actually reached `git commit`/`git push` — a
   silent gap between "I wrote the edit" and "it's actually in version
   control." Always confirm the commit hash on `origin/main`, not just that
   the file looks right locally.
2. **`/kaggle/tmp` and `/kaggle/working` are not guaranteed to survive a
   session reset** (including one triggered by clicking "Cancel Run").
   Anything valuable needs to be pushed to a Kaggle Dataset (or downloaded)
   before considering it safe.
3. **For any long-running Kaggle job, use Save Version → Save & Run All
   (Commit) rather than an interactive session** — commits run as
   background batch jobs independent of browser/connection state.
4. **`kaggle datasets create --dir-mode zip` on a folder of many small
   files may upload file-by-file rather than as one archive.** Always zip
   explicitly first and point the create command at a folder containing
   just the zip, rather than trusting the flag on a large flat directory.
5. **Run `git status` before every `git add`**, especially right after
   creating any zip, model checkpoint, or other large output — this is what
   would have caught the `data/` tracking bug immediately instead of
   accumulating 600MB across many commits.
6. **Lesson 1 recurred exactly as described, concretely this time:** the
   Tier 3.1/3.2 commit was made locally but not pushed before starting a
   Kaggle session that `git clone`s from GitHub — the clone silently
   succeeded but was simply missing `train_ensemble.py`, producing a
   confusing "No such file" error with no hint that the real problem was an
   unpushed commit. Checking `git log origin/main -1` first would have
   caught it in seconds instead of a failed run.
7. **Don't assume a Kaggle-mounted dataset's folder layout matches what's
   documented here — verify it live.** By the time of the ensemble run, the
   `crystal-gnn-graphs` dataset's `graphs_newfeatures` folder (documented
   above as the correct, complete 25,125-file format) actually only had
   5,000 files, while an undocumented `graphs_scaled` folder had the full
   25,125 — evidently a later re-upload that was never written back into
   this file. Confirmed correct via a quick empirical check (load one graph
   from each candidate folder, verify `x.shape == (n, 92)` with the expected
   `atom_init.json`-style categorical-block encoding) rather than trusting
   folder names or this document's history.
8. **In the Kaggle notebook editor, click/type only after confirming a
   blinking text cursor is visible in the cell** — a single click just
   *selects* a cell (command mode); typing there is read as a stream of
   Jupyter keyboard shortcuts (`a`/`b`/`c`/`d`/`x`/etc. all do things),
   which silently created several stray empty cells before this was caught.
   Also: Kaggle's "Save Version" can throw a `ConcurrencyViolation` /
   sequence-number error if more than one browser tab/session has the same
   notebook's draft open — close any other open tab on the same notebook
   before committing.

---

## Outstanding / deferred
- CGCNN formation energy MAE still ~1.6x published; ALIGNN's is ~1.8x
  published and regressed slightly vs. CGCNN — likely needs either more
  training data (available: up to 75,119 in the original stability-filtered
  MP pool, well beyond the 25,125 currently used) or architecture-specific
  hyperparameter tuning (more/fewer layers, different hidden dims) rather
  than further optimizer tuning (diminishing returns already observed).
- Git history was rewritten (`filter-repo`) mid-project — old commit hashes
  before that point are no longer valid if ever sharing/collaborating.
- Not yet done: multi-task-vs-separate-model ablation write-up, ALIGNN
  interpretability/UQ/active-learning (Tiers 3.1/3.2/3.3 above only cover
  CGCNN so far), public README + CV bullets (Tier 3.5, deliberately deferred
  until final scope is locked).

## Next candidate directions
- Repeat Tier 3.1/3.2/3.3 for ALIGNN, for a full architecture-vs-architecture
  comparison on interpretability, calibration, and active-learning sample
  efficiency — not just point-prediction MAE.
- Scale to the remaining ~50k available materials if pursuing the
  formation-energy gap further — worth revisiting given how much the FE gap
  closed from ensembling alone (1.6x → 1.3x published) with zero new data.
- Tier 3.5: public README + CV bullets. With Tiers 1-3.3 all complete, scope
  is arguably locked enough to write these now unless ALIGNN parity (above)
  is wanted first.


