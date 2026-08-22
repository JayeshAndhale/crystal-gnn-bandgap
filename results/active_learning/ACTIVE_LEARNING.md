# Active Learning Simulation: Uncertainty vs. Random Acquisition

Pool-based active learning simulation on the same fixed held-out test set used throughout this project. Both strategies share an identical initial labeled set and round schedule -- the only difference is which materials get revealed each round: highest ensemble-std ('uncertainty', acquiring by BG uncertainty) vs. uniformly random ('random', the baseline). A small ensemble is retrained from scratch each round (not warm-started) on the current labeled set.

## Results by label budget

| n_labels | uncertainty FE MAE | random FE MAE | uncertainty BG MAE | random BG MAE |
|---|---|---|---|---|
| 2000 | 0.1212 | 0.1212 | 0.5716 | 0.5716 |
| 5000 | 0.1133 | 0.0861 | 0.4927 | 0.5051 |
| 8000 | 0.0870 | 0.0805 | 0.4493 | 0.4574 |
| 11000 | 0.0711 | 0.0749 | 0.4179 | 0.4487 |
| 14000 | 0.0837 | 0.0693 | 0.3932 | 0.4176 |
| 17000 | 0.0668 | 0.0668 | 0.3935 | 0.4055 |
| 20000 | 0.0627 | 0.0607 | 0.3724 | 0.3946 |

## Label efficiency

**Formation energy**, target MAE 0.0607 eV/atom (random's round -1): uncertainty-based acquisition never reached this MAE within the labels tried.
**Formation energy**, target MAE 0.0668 eV/atom: uncertainty-based acquisition reached it at ~16996 labels vs. random's 17000 labels (0% fewer labels).

**Band gap**, target MAE 0.3946 eV: uncertainty-based acquisition reached it at ~13831 labels vs. random's 20000 labels (31% fewer labels).
**Band gap**, target MAE 0.4055 eV: uncertainty-based acquisition reached it at ~12504 labels vs. random's 17000 labels (26% fewer labels).

## Interpretation

Acquisition targets BG uncertainty specifically because Tier 3.2's calibration analysis found band gap's ensemble std ranks materials by difficulty more reliably (r=0.49) than formation energy's (r=0.23), even though its absolute magnitude is underconfident -- acquisition only needs correct relative ranking, not calibrated magnitude, so that mismatch doesn't invalidate using it here. Whether this actually beats random sampling, and by how much, is exactly what the table and plot above show for this run -- not asserted in advance.

## Caveats

- Each round trains a smaller, faster ensemble (not the full Tier 3.2 setup) to keep the whole multi-round simulation tractable, so absolute MAE at any given round is not directly comparable to the Tier 2/3.2 headline numbers -- only the *relative* gap between the two curves at equal label budgets is the actual finding here.
- A single simulation run has no error bars on the acquisition-strategy comparison itself (unlike Tier 3.2's ensemble members, which do have across-seed variance) -- a repeat run with a different AL_SEED would be needed to know how much of any gap is signal vs. run-to-run noise.
