# CGCNN Ensemble Uncertainty Quantification

5-member deep ensemble, identical architecture, identical train/val/test split (SPLIT_SEED=42) for every member -- only model init + minibatch shuffling differ (per-member torch seed). Evaluated on the held-out test set (2513 materials), never seen during training or checkpoint selection by any member.

## Ensembling vs. a single model

Mean individual-member test MAE: FE 0.0685 eV/atom, BG 0.3861 eV.
Ensemble-mean test MAE: FE 0.0509 eV/atom, BG 0.3558 eV.
Averaging the ensemble improved formation energy MAE by 0.0175 and improved band gap MAE by 0.0303, relative to the average single member -- the expected direction if members make partially independent errors.

## Calibration

**Formation energy:** corr(predicted std, actual |error|) = 0.228 (p = 5.62e-31); mean predicted std = 0.0538 eV/atom; mean actual |error| = 0.0509 eV/atom; miscalibration ratio (mean|error| / mean std) = 0.946 (well-calibrated Gaussian residuals -> ~0.80).
Positive, significant correlation -- the ensemble's disagreement is real signal: materials it's more uncertain about tend to actually be the ones it gets more wrong.

**Band gap:** corr(predicted std, actual |error|) = 0.490 (p = 7.02e-152); mean predicted std = 0.1725 eV; mean actual |error| = 0.3558 eV; miscalibration ratio (mean|error| / mean std) = 2.063 (well-calibrated Gaussian residuals -> ~0.80).
Positive, significant correlation -- the ensemble's disagreement is real signal: materials it's more uncertain about tend to actually be the ones it gets more wrong.

## Files

`calibration.png` -- spread-skill plot (predicted std vs. actual error, binned).
`uncertainty_vs_error_scatter.png` -- same relationship, per-material, unbinned.
`ensemble_predictions.csv` -- raw per-material ensemble mean/std/error for both targets.

## Caveat

Ensemble size (N members) sets the resolution of the std estimate; with a small N, individual per-material std values are noisy even if the aggregate calibration trend above is real. Read the binned spread-skill plot, not single points on the per-material scatter, as the calibration evidence.
