## Setup

Tested with Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

The first run reads the xlsx file, which is slow (several minutes), then caches it as parquet.

```bash
# Quick smoke test of the whole pipeline (small configuration, not for results)
python ct_study.py --stage all --fast --data data/slice_localization_data.xlsx

# Full EDA, cross-validation and report (never touches the held-out test set)
python ct_study.py --stage all --data data/slice_localization_data.xlsx

# Final held-out test evaluation: run ONCE, at the very end
python ct_study.py --stage final
```

Outputs go to `results/` (CSV and JSON) and `figures/` (PNG). The patient-level split,
random seed and library versions are saved in `results/environment_and_split.json`.

## Method summary

- Patient-level dev/test split, then patient-grouped 5-fold cross-validation.
- All preprocessing, selection and extraction are fitted on training folds only.
- Benchmarks: Ridge and Histogram Gradient Boosting.
- Methods compared: filters, wrappers, embedded methods, PCA, PLS and binned-target LDA.
- Metrics: MAE, RMSE and R2, plus runtime and selection stability (Jaccard, Kuncheva).

## Status and known issues

This is an exploratory run. Known issues:
- The SFS candidate pool equals k_max, so SFS at k = 40 returns the F-test top 40.
- The baseline appears as two summary rows because its k varies by fold.
- The k grid ends at 160, and the filters have not plateaued.
- One repeat of 5-fold CV; the held-out test set was already used once.

## Provenance

The pipeline started from a reference implementation provided during mentoring with an
AI assistant (Claude). State here which parts you re-implemented yourself.
