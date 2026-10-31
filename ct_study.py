#!/usr/bin/env python3
"""
CT Slice Localization: feature selection and dimensionality reduction study
===========================================================================
REFERENCE IMPLEMENTATION (end to end). Use it to check your own work against,
not as a substitute for understanding every decision in it.

Protocol (all decisions are defaults you must be able to defend or change):
  * Exact duplicate rows are dropped, then PATIENTS are split into a dev set and
    a held-out test set. The test set is touched only by `--stage final`.
  * EDA runs on the dev set only, so nothing is learned from the test patients.
  * Outer CV on dev is grouped by patientId (slices of one patient never appear in
    both train and validation). Repeats re-shuffle the patient-to-fold assignment.
  * Inside every fold: drop constant columns -> StandardScaler -> feature
    selector / extractor -> model. Everything is fitted on the training part only.
  * Selectors produce a RANKING on the training fold; the top-k are then evaluated for
    every k in the grid. Selected sets are stored for stability analysis.
  * Hyperparameters of Ridge are tuned by an inner grouped CV on the training fold.
  * Paired comparisons use the Nadeau-Bengio corrected resampled t-test + Holm.

Stages:  eda | cv | report | final | all   (`all` = eda + cv + report; it never
touches the test set. Run `final` once, at the very end.)

Usage:
  python ct_study.py --stage all --data slice_localization_data.xlsx
  python ct_study.py --stage all --fast        # small smoke-test configuration
  python ct_study.py --stage final
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, Ridge, lars_path
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

FEATURES = [f"value{i}" for i in range(384)]   # 0-239 bone, 240-383 air
N_BONE = 240
SENTINEL = -0.25                                 # "bin outside the image"
TARGET, GROUP = "reference", "patientId"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    data_path: str = "slice_localization_data.xlsx"
    cache_path: str = "data/ct_slices.parquet"
    out_dir: str = "results"
    fig_dir: str = "figures"
    seed: int = 42
    test_patient_frac: float = 0.2
    n_folds: int = 5
    n_repeats: int = 1
    row_stride: int = 1            # keep every n-th dev row (compute saver; slices are near-duplicates)
    k_grid: tuple = (5, 10, 20, 40, 80, 160)
    comp_grid: tuple = (5, 10, 20, 40, 80)
    models: tuple = ("ridge", "hgb")
    selectors: tuple = ("variance", "f_test", "spearman", "mutual_info", "corr_filter",
                        "rfe_ridge", "sfs_ridge", "lasso_path", "elastic_net",
                        "rf_impurity", "rf_permutation")
    extractors: tuple = ("pca", "pls", "lda_binned")
    # method hyperparameters
    mi_subsample: int = 5000
    corr_threshold: float = 0.95
    corr_rows: int = 20000
    rfe_alpha: float = 10.0
    rfe_drop_frac: float = 0.10
    sfs_pool: int = 40
    sfs_max: int = 40
    sfs_rows: int = 8000
    rf_trees: int = 100
    rf_rows: int = 10000
    hgb_iter: int = 200
    lda_bins: int = 10
    stab_k: int = 40               # k used for cross-method similarity / frequency heatmap
    n_jobs: int = -1


def fast_config(cfg: Config) -> Config:
    """Tiny configuration for smoke-testing the code path (NOT for results)."""
    cfg.row_stride, cfg.n_folds = 10, 3
    cfg.k_grid, cfg.comp_grid, cfg.stab_k = (10, 40), (10,), 40
    cfg.mi_subsample, cfg.corr_rows = 1500, 3000
    cfg.sfs_pool, cfg.sfs_max, cfg.sfs_rows = 12, 6, 1500
    cfg.rf_trees, cfg.rf_rows, cfg.hgb_iter = 20, 2000, 30
    cfg.corr_threshold = 0.9
    return cfg


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _read_any(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s == ".parquet":
        return pd.read_parquet(path)
    if s in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    if s == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)          # xlsx: slow (minutes), so we cache once


def load_data(cfg: Config) -> pd.DataFrame:
    cache = Path(cfg.cache_path)
    if cache.exists():
        df = _read_any(cache)
    else:
        log(f"Reading {cfg.data_path} (slow for xlsx; cached afterwards)")
        df = _read_any(Path(cfg.data_path))
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(cache)
        except ImportError:
            df.to_pickle(cache.with_suffix(".pkl"))
    expected = [GROUP] + FEATURES + [TARGET]
    if list(df.columns) != expected:
        raise ValueError("Unexpected columns: the file does not match patientId, value0..value383, reference")
    return df


def split_dev_test(df: pd.DataFrame, cfg: Config):
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log(f"Dropped {n0 - len(df)} exact duplicate rows")
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_patient_frac, random_state=cfg.seed)
    dev_i, test_i = next(gss.split(df, groups=df[GROUP]))
    dev, test = df.iloc[dev_i].reset_index(drop=True), df.iloc[test_i].reset_index(drop=True)
    assert set(dev[GROUP]).isdisjoint(set(test[GROUP]))
    return dev, test


def to_arrays(df: pd.DataFrame):
    return df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float), df[GROUP].to_numpy()


def grouped_folds(groups, n_folds: int, seed: int):
    """Patients are shuffled and greedily assigned to the currently lightest fold."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    load, assign = np.zeros(n_folds), {}
    for i in rng.permutation(len(uniq)):
        f = int(np.argmin(load))
        assign[uniq[i]] = f
        load[f] += counts[i]
    fold_of = pd.Series(groups).map(assign).to_numpy()
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(n_folds)]


class Prep:
    """Drop columns that are constant in the TRAINING data, then standardise."""
    def fit(self, X):
        self.raw_var_all = X.var(0)
        self.keep_ = np.where(self.raw_var_all > 0)[0]
        self.scaler_ = StandardScaler().fit(X[:, self.keep_])
        return self

    def transform(self, X):
        return self.scaler_.transform(X[:, self.keep_])

    @property
    def raw_var(self):
        return self.raw_var_all[self.keep_]


def regression_metrics(y, p) -> dict:
    return dict(mae=float(mean_absolute_error(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                r2=float(r2_score(y, p)))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def fit_model(name: str, X, y, g, cfg: Config):
    if name == "ridge":
        inner = grouped_folds(g, 3, cfg.seed)
        gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1, 10, 100, 1000]}, cv=inner,
                          scoring="neg_root_mean_squared_error", n_jobs=cfg.n_jobs)
        gs.fit(X, y)
        return gs.best_estimator_, gs.best_params_
    if name == "hgb":
        m = HistGradientBoostingRegressor(max_iter=cfg.hgb_iter, learning_rate=0.1,
                                          early_stopping=False, random_state=cfg.seed)
        return m.fit(X, y), {}
    if name == "dummy":
        return DummyRegressor(strategy="mean").fit(X, y), {}
    raise ValueError(name)


def evaluate(model: str, Xtr, ytr, gtr, Xva, yva, cfg: Config) -> dict:
    t0 = time.perf_counter()
    est, params = fit_model(model, Xtr, ytr, gtr, cfg)
    fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = est.predict(Xva)
    pred_t = time.perf_counter() - t0
    return {**regression_metrics(yva, pred), "fit_time": fit_t, "pred_time": pred_t,
            "best_params": json.dumps(params)}


# --------------------------------------------------------------------------- #
# Feature selection: every ranker returns (scores, n_valid); higher score = better.
# n_valid = number of positions with a meaningful order (None = all).
# --------------------------------------------------------------------------- #
def _abs_pearson(c):
    yz = (c.y - c.y.mean()) / c.y.std()
    return np.abs(c.X.T @ yz) / len(yz)       # X is already standardised


def score_variance(c):          # unsupervised, on RAW (unscaled) training variance
    return c.raw_var.copy(), None


def score_f_test(c):            # univariate linear F-test (regression form of ANOVA F)
    f, _ = f_regression(c.X, c.y)
    return np.nan_to_num(f), None


def score_spearman(c):          # |rank correlation|: monotonic, not only linear
    Xr = stats.rankdata(c.X, axis=0)
    yr = stats.rankdata(c.y)
    Xr = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-12)
    yr = (yr - yr.mean()) / yr.std()
    return np.abs(Xr.T @ yr) / len(yr), None


def score_mutual_info(c):       # k-NN (Kraskov) estimator on a row subsample
    idx = c.rng.choice(len(c.y), min(len(c.y), c.cfg.mi_subsample), replace=False)
    mi = mutual_info_regression(c.X[idx], c.y[idx], n_neighbors=3,
                                random_state=int(c.rng.integers(1_000_000_000)))
    return mi, None


def score_corr_filter(c):       # relevance-ordered greedy redundancy removal
    n = c.X.shape[0]
    idx = c.rng.choice(n, min(n, c.cfg.corr_rows), replace=False)
    C = np.nan_to_num(np.abs(np.corrcoef(c.X[idx], rowvar=False)))
    rel = _abs_pearson(c)
    order = np.argsort(-rel, kind="stable")
    kept = []
    for f in order:
        if not kept or C[f, kept].max() < c.cfg.corr_threshold:
            kept.append(f)
    rest = [f for f in order if f not in set(kept)]
    ranking = np.array(kept + rest)
    scores = np.empty(len(order))
    scores[ranking] = np.arange(len(order))[::-1]
    return scores, None


def score_rfe_ridge(c):         # backward elimination, 10% of the remaining features per round
    p = c.X.shape[1]
    active, elim = np.arange(p), []
    est = Ridge(alpha=c.cfg.rfe_alpha)
    while len(active) > 1:
        est.fit(c.X[:, active], c.y)
        n_drop = min(max(1, int(c.cfg.rfe_drop_frac * len(active))), len(active) - 1)
        worst = np.argsort(np.abs(est.coef_), kind="stable")[:n_drop]
        elim.extend(active[worst].tolist())
        active = np.delete(active, worst)
    elim.extend(active.tolist())
    scores = np.empty(p)
    scores[np.array(elim)] = np.arange(p)       # eliminated last = best
    return scores, None


def score_sfs_ridge(c):         # greedy forward selection on a pre-filtered pool (cost control)
    cfg = c.cfg
    f, _ = f_regression(c.X, c.y)
    pool = list(np.argsort(-np.nan_to_num(f), kind="stable")[:cfg.sfs_pool])
    idx = c.rng.choice(len(c.y), min(len(c.y), cfg.sfs_rows), replace=False)
    Xs, ys, gs = c.X[idx], c.y[idx], c.g[idx]
    folds = grouped_folds(gs, 3, int(c.rng.integers(1_000_000)))
    selected, remaining = [], pool.copy()

    def cv_rmse(cols):
        errs = []
        for tr, va in folds:
            m = Ridge(alpha=1.0).fit(Xs[tr][:, cols], ys[tr])
            errs.append(mean_squared_error(ys[va], m.predict(Xs[va][:, cols])))
        return float(np.mean(errs))

    for _ in range(min(cfg.sfs_max, len(pool))):
        best = min(remaining, key=lambda j: cv_rmse(selected + [j]))
        selected.append(best)
        remaining.remove(best)
    scores = np.zeros(c.X.shape[1])
    for pos, j in enumerate(selected):
        scores[j] = len(selected) - pos
    return scores, len(selected)


def score_lasso_path(c):        # order in which features enter the Lasso (LARS) path
    p = c.X.shape[1]
    _, _, coefs = lars_path(c.X, c.y, method="lasso", max_iter=4 * p)
    nz = coefs != 0
    entered = nz.any(axis=1)
    first = np.argmax(nz, axis=1)
    scores = np.where(entered, 10 * p - first, 0).astype(float)
    return scores + 1e-9 * _abs_pearson(c), int(entered.sum())


def score_elastic_net(c):       # |coef| at the CV-chosen alpha (inner grouped CV)
    inner = grouped_folds(c.g, 3, int(c.rng.integers(1_000_000)))
    m = ElasticNetCV(l1_ratio=0.5, n_alphas=20, cv=inner, max_iter=2000, tol=1e-3,
                     n_jobs=c.cfg.n_jobs, random_state=int(c.rng.integers(1_000_000)))
    m.fit(c.X, c.y)
    coef = np.abs(m.coef_)
    return coef + 1e-9 * _abs_pearson(c), int((coef > 0).sum())


def _rf(c, n_rows):
    idx = c.rng.choice(len(c.y), min(len(c.y), n_rows), replace=False)
    rf = RandomForestRegressor(n_estimators=c.cfg.rf_trees, max_features=0.33, min_samples_leaf=5,
                               n_jobs=c.cfg.n_jobs, random_state=int(c.rng.integers(1_000_000)))
    return rf, idx


def score_rf_impurity(c):       # mean decrease in impurity (biased toward high-cardinality features)
    rf, idx = _rf(c, c.cfg.rf_rows)
    rf.fit(c.X[idx], c.y[idx])
    return rf.feature_importances_, None


def score_rf_permutation(c):    # permutation importance on an inner, patient-disjoint hold-out
    rf, idx = _rf(c, c.cfg.rf_rows)
    Xs, ys, gs = c.X[idx], c.y[idx], c.g[idx]
    tr, ho = next(GroupShuffleSplit(1, test_size=0.25,
                                    random_state=int(c.rng.integers(1_000_000))).split(Xs, groups=gs))
    rf.fit(Xs[tr], ys[tr])
    pi = permutation_importance(rf, Xs[ho], ys[ho], scoring="neg_root_mean_squared_error",
                                n_repeats=3, random_state=int(c.rng.integers(1_000_000)),
                                n_jobs=c.cfg.n_jobs)
    return pi.importances_mean, None


RANKERS = {
    "variance": ("filter", score_variance),
    "f_test": ("filter", score_f_test),
    "spearman": ("filter", score_spearman),
    "mutual_info": ("filter", score_mutual_info),
    "corr_filter": ("filter", score_corr_filter),
    "rfe_ridge": ("wrapper", score_rfe_ridge),
    "sfs_ridge": ("wrapper", score_sfs_ridge),
    "lasso_path": ("embedded", score_lasso_path),
    "elastic_net": ("embedded", score_elastic_net),
    "rf_impurity": ("embedded", score_rf_impurity),
    "rf_permutation": ("embedded", score_rf_permutation),
}
EXTRACTOR_FAMILY = {"pca": "extraction", "pls": "extraction", "lda_binned": "extraction"}


def make_ctx(Xtr, ytr, gtr, prep: Prep, cfg: Config, seed: int):
    return SimpleNamespace(X=Xtr, y=ytr, g=gtr, raw_var=prep.raw_var, cfg=cfg,
                           rng=np.random.default_rng(seed))


# --------------------------------------------------------------------------- #
# Feature extraction (fitted on training data only)
# --------------------------------------------------------------------------- #
def run_extractor(method: str, Xtr, ytr, Xva, cfg: Config):
    """Returns (Ztr, Zva, fit_time, extra). Columns are ordered, so Z[:, :n] = n components."""
    p = Xtr.shape[1]
    t0 = time.perf_counter()
    extra = None
    if method == "pca":
        m = PCA(n_components=min(max(cfg.comp_grid), p - 1), svd_solver="randomized",
                random_state=cfg.seed).fit(Xtr)
        extra = np.cumsum(m.explained_variance_ratio_)
        Ztr, Zva = m.transform(Xtr), m.transform(Xva)
    elif method == "pls":      # supervised linear extraction; components are sequential (NIPALS)
        m = PLSRegression(n_components=min(max(cfg.comp_grid), p - 1), scale=False).fit(Xtr, ytr)
        Ztr, Zva = m.transform(Xtr), m.transform(Xva)
    elif method == "lda_binned":   # EXPLORATORY: LDA needs classes, so the target is quantile-binned
        edges = np.quantile(ytr, np.linspace(0, 1, cfg.lda_bins + 1)[1:-1])
        m = LinearDiscriminantAnalysis(solver="svd").fit(Xtr, np.digitize(ytr, edges))
        Ztr, Zva = m.transform(Xtr), m.transform(Xva)
    else:
        raise ValueError(method)
    return Ztr, Zva, time.perf_counter() - t0, extra


def extractor_sizes(method: str, Z, cfg: Config):
    return sorted({min(n, Z.shape[1]) for n in cfg.comp_grid})


# --------------------------------------------------------------------------- #
# Stage: EDA (dev set only)
# --------------------------------------------------------------------------- #
def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eda(dev: pd.DataFrame, cfg: Config, out: Path, figs: Path):
    X, y, g = to_arrays(dev)
    rng = np.random.default_rng(cfg.seed)
    s = {"n_rows": len(dev), "n_patients": int(dev[GROUP].nunique()),
         "n_features": len(FEATURES), "n_missing": int(dev.isna().sum().sum()),
         "n_exact_duplicates": int(dev.duplicated().sum()),
         "target_min": float(y.min()), "target_max": float(y.max()),
         "target_mean": float(y.mean()), "target_std": float(y.std()),
         "target_skew": float(stats.skew(y))}
    nun = dev[FEATURES].nunique()
    var = X.var(0)
    s["n_constant_cols"] = int((nun == 1).sum())
    s["n_near_zero_var_cols(<1e-4)"] = int(((var < 1e-4) & (nun > 1)).sum())
    s["n_cols_le5_unique_values"] = int((nun <= 5).sum())
    sent, zero = (X == SENTINEL).mean(0), (X == 0).mean(0)
    s["sentinel_share_bone"], s["sentinel_share_air"] = float(sent[:N_BONE].mean()), float(sent[N_BONE:].mean())
    s["zero_share_bone"], s["zero_share_air"] = float(zero[:N_BONE].mean()), float(zero[N_BONE:].mean())
    s["rows_per_patient_min_med_max"] = [int(v) for v in dev[GROUP].value_counts().agg(["min", "median", "max"])]
    # between-patient share of target variance (eta^2): why patientId must never be a feature
    pm = dev.groupby(GROUP)[TARGET].transform("mean")
    s["eta2_target_by_patient"] = float(((pm - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    same = (g[1:] == g[:-1])
    s["median_target_gap_adjacent_rows"] = float(np.median(np.abs(np.diff(y))[same]))

    # sample-based statistics
    idx = rng.choice(len(dev), min(len(dev), 10000), replace=False)
    Xs, ys = X[idx], y[idx]
    live = np.where(Xs.std(0) > 0)[0]
    R = np.apply_along_axis(stats.rankdata, 0, Xs[:, live])
    rho = np.corrcoef(R, rowvar=False)
    iu = np.triu_indices_from(rho, 1)
    s["n_pairs_abs_spearman_gt_0.95"] = int((np.abs(rho[iu]) > 0.95).sum())
    ev = np.sort(np.linalg.eigvalsh(np.nan_to_num(np.corrcoef(Xs[:, live], rowvar=False))))[::-1]
    ev = np.clip(ev, 0, None)
    s["n_pcs_for_95pct_variance_(standardised)"] = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)
    s["condition_number_corr"] = float(ev.max() / max(ev.min(), 1e-12))

    # feature-target relationships (linear vs monotonic vs general)
    pear = np.array([np.corrcoef(Xs[:, j], ys)[0, 1] if Xs[:, j].std() > 0 else 0 for j in range(384)])
    sp = np.array([stats.spearmanr(Xs[:, j], ys)[0] if Xs[:, j].std() > 0 else 0 for j in range(384)])
    mi_idx = rng.choice(len(Xs), min(len(Xs), 3000), replace=False)
    mi = mutual_info_regression(Xs[mi_idx], ys[mi_idx], n_neighbors=3, random_state=cfg.seed)
    ft = pd.DataFrame({"feature": FEATURES, "block": ["bone"] * N_BONE + ["air"] * (384 - N_BONE),
                       "sentinel_share": sent, "zero_share": zero, "skew": stats.skew(X, axis=0),
                       "pearson": pear, "spearman": sp, "mutual_info": mi})
    ft["nonlinear_candidate"] = (ft.mutual_info > ft.mutual_info.quantile(0.9)) & (ft.pearson.abs() < ft.pearson.abs().median())
    out.mkdir(parents=True, exist_ok=True)
    ft.to_csv(out / "eda_feature_table.csv", index=False)
    pp = dev.groupby(GROUP)[TARGET].agg(["count", "min", "max", "mean"]).reset_index()
    pp.to_csv(out / "eda_patient_table.csv", index=False)
    (out / "eda_summary.json").write_text(json.dumps(s, indent=2))

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(y, bins=60); ax[0].set(title="Target distribution (dev)", xlabel="reference")
    ax[1].bar(range(len(pp)), pp["count"].sort_values(ascending=False).to_numpy())
    ax[1].set(title="Rows per patient", xlabel="patient (sorted)", ylabel="rows")
    pps = pp.sort_values("min")
    ax[2].vlines(range(len(pps)), pps["min"], pps["max"])
    ax[2].set(title="Target range covered by each patient", xlabel="patient (sorted by min)", ylabel="reference")
    savefig(fig, figs / "eda_target_patients.png")

    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].plot(sent, label="share = -0.25 (outside image)"); ax[0].plot(zero, label="share = 0")
    ax[0].axvline(N_BONE, color="k", ls="--"); ax[0].legend(); ax[0].set(title="Sentinel and zero share per column")
    ax[1].plot(ft.mutual_info, label="mutual information"); ax[1].plot(ft.spearman.abs(), label="|Spearman|")
    ax[1].plot(ft.pearson.abs(), label="|Pearson|"); ax[1].axvline(N_BONE, color="k", ls="--")
    ax[1].legend(); ax[1].set(xlabel="feature index (bone | air)", title="Feature-target association (sample)")
    savefig(fig, figs / "eda_sentinel_association.png")

    order = leaves_list(linkage(squareform(1 - np.abs(rho), checks=False), "average"))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.abs(rho)[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
    ax.set(title="|Spearman| between features (clustered order)"); fig.colorbar(im)
    savefig(fig, figs / "eda_corr_clustered.png")
    log(f"EDA done -> {out}/eda_summary.json")
    return s


def plot_tsne(dev: pd.DataFrame, cfg: Config, figs: Path):
    """VISUALISATION ONLY: t-SNE/UMAP are not used as predictive preprocessing."""
    X, y, _ = to_arrays(dev)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(dev), min(len(dev), 4000), replace=False)
    Xs = StandardScaler().fit_transform(X[idx])
    Xs = PCA(n_components=min(50, Xs.shape[1] - 1), random_state=cfg.seed).fit_transform(Xs)
    panels = [("t-SNE", TSNE(perplexity=30, init="pca", random_state=cfg.seed).fit_transform(Xs))]
    try:
        import umap  # optional dependency
        panels.append(("UMAP", umap.UMAP(random_state=cfg.seed).fit_transform(Xs)))
    except ImportError:
        pass
    fig, ax = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    for a, (name, Z) in zip(ax[0], panels):
        sc = a.scatter(Z[:, 0], Z[:, 1], c=y[idx], s=4, cmap="viridis")
        a.set(title=f"{name} (visualisation only)"); fig.colorbar(sc, ax=a, label="reference")
    savefig(fig, figs / "embedding_visualisation.png")


# --------------------------------------------------------------------------- #
# Stage: grouped cross-validation experiments
# --------------------------------------------------------------------------- #
def run_cv(dev: pd.DataFrame, cfg: Config, out: Path):
    X, y, g = to_arrays(dev)
    out.mkdir(parents=True, exist_ok=True)
    records, selections, pca_ev = [], {}, []
    jobs = [(rep, f, tr, va) for rep in range(cfg.n_repeats)
            for f, (tr, va) in enumerate(grouped_folds(g, cfg.n_folds, cfg.seed + rep))]
    for rep, f, tr, va in jobs:
        log(f"=== repeat {rep}, fold {f}: train {len(tr)} rows / validation {len(va)} rows ===")
        prep = Prep().fit(X[tr])
        Xtr, Xva = prep.transform(X[tr]), prep.transform(X[va])
        ytr, yva, gtr = y[tr], y[va], g[tr]
        p = Xtr.shape[1]
        base = dict(rep=rep, fold=f, n_train=len(tr), n_val=len(va))

        def add(kind, family, method, k, model, res, sel_t=0.0):
            records.append({**base, "kind": kind, "family": family, "method": method, "k": k,
                            "model": model, "select_time": sel_t, **res})

        add("reference", "reference", "mean_predictor", 0, "dummy",
            evaluate("dummy", Xtr, ytr, gtr, Xva, yva, cfg))
        for m in cfg.models:
            add("baseline", "baseline", "baseline_all", p, m, evaluate(m, Xtr, ytr, gtr, Xva, yva, cfg))

        for method in cfg.selectors:
            family, fn = RANKERS[method]
            ctx = make_ctx(Xtr, ytr, gtr, prep, cfg, cfg.seed + 1000 * rep + 10 * f)
            t0 = time.perf_counter()
            try:
                scores, n_valid = fn(ctx)
            except Exception as e:                     # keep the run alive, but never silently
                log(f"  !! {method} failed: {type(e).__name__}: {e}")
                continue
            sel_t = time.perf_counter() - t0
            order = np.argsort(-scores, kind="stable")
            log(f"  {method:<15} ranked in {sel_t:6.1f}s")
            for k in cfg.k_grid:
                if k >= p or (n_valid is not None and k > n_valid):
                    continue
                cols = order[:k]
                selections[(method, rep, f, k)] = frozenset(prep.keep_[cols].tolist())
                for m in cfg.models:
                    add("selection", family, method, k, m,
                        evaluate(m, Xtr[:, cols], ytr, gtr, Xva[:, cols], yva, cfg), sel_t)

        for method in cfg.extractors:
            Ztr, Zva, t_fit, extra = run_extractor(method, Xtr, ytr, Xva, cfg)
            if extra is not None:
                pca_ev.append(extra)
            log(f"  {method:<15} fitted in {t_fit:6.1f}s")
            for n in extractor_sizes(method, Ztr, cfg):
                for m in cfg.models:
                    add("extraction", EXTRACTOR_FAMILY[method], method, n, m,
                        evaluate(m, Ztr[:, :n], ytr, gtr, Zva[:, :n], yva, cfg), t_fit)
        pd.DataFrame(records).to_csv(out / "cv_results.csv", index=False)   # checkpoint every fold

    with open(out / "selections.pkl", "wb") as fh:
        pickle.dump(selections, fh)
    if pca_ev:
        np.save(out / "pca_cumvar.npy", np.array([e[:min(len(e) for e in pca_ev)] for e in pca_ev]))
    log("CV finished")


# --------------------------------------------------------------------------- #
# Stage: report (summary, stability, statistics, figures)
# --------------------------------------------------------------------------- #
def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["kind", "family", "method", "k", "model"]
    agg = df.groupby(keys).agg(
        n_splits=("rmse", "size"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        select_time=("select_time", "mean"), fit_time=("fit_time", "mean"),
        pred_time=("pred_time", "mean")).reset_index()
    return agg


def kuncheva(a, b, k, n):
    if k <= 0 or k >= n:
        return np.nan
    return (len(a & b) * n - k * k) / (k * (n - k))


def stability_table(selections: dict, n_total: int = 384) -> pd.DataFrame:
    groups = {}
    for (method, rep, f, k), s in selections.items():
        groups.setdefault((method, k), []).append(s)
    rows = []
    for (method, k), sets in sorted(groups.items()):
        if len(sets) < 2:
            continue
        jac = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
        kun = [kuncheva(a, b, k, n_total) for a, b in combinations(sets, 2)]
        freq = pd.Series([j for s in sets for j in s]).value_counts() / len(sets)
        rows.append({"method": method, "k": k, "n_sets": len(sets),
                     "jaccard_mean": np.mean(jac), "jaccard_std": np.std(jac, ddof=1) if len(jac) > 1 else np.nan,
                     "kuncheva_mean": np.nanmean(kun),
                     "n_always_selected": int((freq == 1).sum()), "n_ever_selected": int(len(freq))})
    return pd.DataFrame(rows)


def method_similarity(selections: dict, k: int) -> pd.DataFrame:
    by = {}
    for (method, rep, f, kk), s in selections.items():
        if kk == k:
            by.setdefault(method, {})[(rep, f)] = s
    methods = sorted(by)
    M = pd.DataFrame(np.nan, index=methods, columns=methods)
    for a in methods:
        for b in methods:
            common = by[a].keys() & by[b].keys()
            if common:
                M.loc[a, b] = np.mean([len(by[a][c] & by[b][c]) / len(by[a][c] | by[b][c]) for c in common])
    return M


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    m = int(np.sum(~np.isnan(p)))
    adj, run = np.full(len(p), np.nan), 0.0
    for rank, i in enumerate(order[:m]):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def compare_to_baseline(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Paired RMSE differences vs the all-feature baseline (negative = better than baseline).
    Nadeau-Bengio corrected resampled t-test; Holm correction within each model."""
    out = []
    for model in cfg.models:
        d = df[df.model == model]
        base = d[d.method == "baseline_all"].set_index(["rep", "fold"])
        if base.empty:
            continue
        ratio = float((base.n_val / base.n_train).mean())
        rows = []
        for (kind, family, method, k), grp in d[d.method != "baseline_all"].groupby(["kind", "family", "method", "k"]):
            g2 = grp.set_index(["rep", "fold"])
            common = base.index.intersection(g2.index)
            diff = (g2.loc[common, "rmse"] - base.loc[common, "rmse"]).to_numpy()
            J = len(diff)
            if J < 2:
                continue
            mean, var = diff.mean(), diff.var(ddof=1)
            se = np.sqrt((1 / J + ratio) * var) if var > 0 else np.nan
            t = mean / se if se and not np.isnan(se) else np.nan
            pval = 2 * stats.t.sf(abs(t), J - 1) if not np.isnan(t) else np.nan
            half = stats.t.ppf(0.975, J - 1) * se if not np.isnan(se) else np.nan
            rows.append({"model": model, "kind": kind, "family": family, "method": method, "k": k,
                         "n_splits": J, "rmse_diff_mean": mean, "ci95_low": mean - half,
                         "ci95_high": mean + half, "p_raw": pval})
        if rows:
            t = pd.DataFrame(rows)
            t["p_holm"] = holm(t.p_raw.to_numpy())
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def best_per_method(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary[~summary.method.isin(["baseline_all", "mean_predictor"])]
    return s.sort_values("rmse_mean").groupby(["model", "method"]).head(1).reset_index(drop=True)


def make_report(cfg: Config, out: Path, figs: Path):
    df = pd.read_csv(out / "cv_results.csv")
    with open(out / "selections.pkl", "rb") as fh:
        selections = pickle.load(fh)
    summary = summarise(df)
    summary.to_csv(out / "summary.csv", index=False)
    stab = stability_table(selections)
    stab.to_csv(out / "stability.csv", index=False)
    stat = compare_to_baseline(df, cfg)
    stat.to_csv(out / "stats_vs_baseline.csv", index=False)
    best = best_per_method(summary)
    best.to_csv(out / "best_k_per_method.csv", index=False)
    sim = method_similarity(selections, cfg.stab_k)
    sim.to_csv(out / f"method_similarity_k{cfg.stab_k}.csv")

    methods = sorted(summary.method.unique())
    cmap = {m: plt.cm.tab20(i % 20) for i, m in enumerate(methods)}

    # performance vs dimensionality
    fig, axes = plt.subplots(1, len(cfg.models), figsize=(7 * len(cfg.models), 5), squeeze=False)
    for ax, model in zip(axes[0], cfg.models):
        s = summary[summary.model == model]
        for method, grp in s[s.kind.isin(["selection", "extraction"])].groupby("method"):
            grp = grp.sort_values("k")
            ls = "--" if grp.kind.iloc[0] == "extraction" else "-"
            ax.errorbar(grp.k, grp.rmse_mean, yerr=grp.rmse_std, label=method, color=cmap[method],
                        ls=ls, marker="o", ms=3, capsize=2, alpha=0.9)
        b = s[s.method == "baseline_all"]
        if len(b):
            ax.axhline(b.rmse_mean.iloc[0], color="k", lw=1.5, label="baseline (all features)")
        ax.set(xscale="log", xlabel="features / components (log)", ylabel="RMSE (mean ± std over folds)",
               title=f"{model}: performance vs dimensionality")
    axes[0][-1].legend(fontsize=7, ncol=2)
    savefig(fig, figs / "performance_vs_k.png")

    # stability
    if len(stab):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for method, grp in stab.groupby("method"):
            ax[0].plot(grp.k, grp.jaccard_mean, marker="o", label=method, color=cmap[method])
            ax[1].plot(grp.k, grp.kuncheva_mean, marker="o", color=cmap[method])
        ax[0].set(xscale="log", title="Mean pairwise Jaccard across folds", xlabel="k", ylim=(0, 1.02))
        ax[1].set(xscale="log", title="Mean pairwise Kuncheva index", xlabel="k")
        ax[0].legend(fontsize=7, ncol=2)
        savefig(fig, figs / "stability_vs_k.png")

    # selection frequency heatmap at stab_k
    freq = {}
    for (method, rep, f, k), s in selections.items():
        if k == cfg.stab_k:
            freq.setdefault(method, []).append(s)
    if freq:
        mat = pd.DataFrame({m: pd.Series([j for s in sets for j in s]).value_counts().reindex(range(384)).fillna(0) / len(sets)
                            for m, sets in freq.items()}).T
        fig, ax = plt.subplots(figsize=(14, 0.45 * len(mat) + 1.5))
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index)
        ax.axvline(N_BONE - 0.5, color="w", ls="--")
        ax.set(xlabel="feature index (bone | air)", title=f"Selection frequency across folds (k={cfg.stab_k})")
        fig.colorbar(im)
        savefig(fig, figs / "selection_frequency.png")

    if len(sim) > 1:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(sim.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(sim))); ax.set_xticklabels(sim.columns, rotation=60, ha="right")
        ax.set_yticks(range(len(sim))); ax.set_yticklabels(sim.index)
        ax.set(title=f"Jaccard similarity between methods' subsets (k={cfg.stab_k})"); fig.colorbar(im)
        savefig(fig, figs / "method_similarity.png")

    # PCA explained variance
    pc = out / "pca_cumvar.npy"
    if pc.exists():
        ev = np.load(pc)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(1, ev.shape[1] + 1), ev.mean(0), marker="o", ms=3)
        ax.fill_between(np.arange(1, ev.shape[1] + 1), ev.min(0), ev.max(0), alpha=0.2)
        for thr in (0.9, 0.95, 0.99):
            ax.axhline(thr, color="grey", lw=0.5)
        ax.set(xlabel="components", ylabel="cumulative explained variance", title="PCA (standardised features)")
        savefig(fig, figs / "pca_explained_variance.png")

    # computational cost of the best configuration of each method
    if len(best):
        fig, ax = plt.subplots(1, len(cfg.models), figsize=(6.5 * len(cfg.models), 4.5), squeeze=False)
        for a, model in zip(ax[0], cfg.models):
            b = best[best.model == model].sort_values("rmse_mean")
            a.barh(b.method, b.select_time, label="selection / extraction time")
            a.barh(b.method, b.fit_time, left=b.select_time, label="model fit time")
            a.set(title=f"{model}: seconds per fold at each method's best k", xlabel="seconds"); a.invert_yaxis()
        ax[0][0].legend(fontsize=8)
        savefig(fig, figs / "computational_cost.png")
    log(f"Report written to {out} and {figs}")


# --------------------------------------------------------------------------- #
# Stage: final test evaluation (run ONCE)
# --------------------------------------------------------------------------- #
def final_test(dev: pd.DataFrame, test: pd.DataFrame, cfg: Config, out: Path):
    log("FINAL: refit on all dev data and evaluate once on the held-out test patients")
    best = pd.read_csv(out / "best_k_per_method.csv")
    Xd, yd, gd = to_arrays(dev)
    Xt, yt, _ = to_arrays(test)
    prep = Prep().fit(Xd)
    Xtr, Xte = prep.transform(Xd), prep.transform(Xt)
    rows = []

    def add(method, kind, family, k, model, res, sel_t=0.0):
        rows.append({"method": method, "kind": kind, "family": family, "k": k, "model": model,
                     "select_time": sel_t, **res})

    add("mean_predictor", "reference", "reference", 0, "dummy", evaluate("dummy", Xtr, yd, gd, Xte, yt, cfg))
    for m in cfg.models:
        add("baseline_all", "baseline", "baseline", Xtr.shape[1], m, evaluate(m, Xtr, yd, gd, Xte, yt, cfg))
    for method in cfg.selectors:
        sub = best[best.method == method]
        if sub.empty:
            continue
        ctx = make_ctx(Xtr, yd, gd, prep, cfg, cfg.seed + 777)
        t0 = time.perf_counter()
        scores, _ = RANKERS[method][1](ctx)
        sel_t = time.perf_counter() - t0
        order = np.argsort(-scores, kind="stable")
        for _, r in sub.iterrows():
            cols = order[:int(r.k)]
            add(method, "selection", r.family, int(r.k), r.model,
                evaluate(r.model, Xtr[:, cols], yd, gd, Xte[:, cols], yt, cfg), sel_t)
    for method in cfg.extractors:
        sub = best[best.method == method]
        if sub.empty:
            continue
        Ztr, Zte, t_fit, _ = run_extractor(method, Xtr, yd, Xte, cfg)
        for _, r in sub.iterrows():
            n = int(r.k)
            add(method, "extraction", r.family, n, r.model,
                evaluate(r.model, Ztr[:, :n], yd, gd, Zte[:, :n], yt, cfg), t_fit)
    res = pd.DataFrame(rows)
    res.to_csv(out / "final_test.csv", index=False)
    log("Final test results saved to final_test.csv (do not tune anything after looking at them)")
    return res


# --------------------------------------------------------------------------- #
# Reproducibility + CLI
# --------------------------------------------------------------------------- #
def save_environment(cfg: Config, out: Path, dev, test):
    out.mkdir(parents=True, exist_ok=True)
    env = {"config": asdict(cfg), "python": platform.python_version(), "numpy": np.__version__,
           "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
           "dev_patients": sorted(map(int, dev[GROUP].unique())),
           "test_patients": sorted(map(int, test[GROUP].unique()))}
    (out / "environment_and_split.json").write_text(json.dumps(env, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["eda", "cv", "report", "final", "all"], default="all")
    ap.add_argument("--data", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figs", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="tiny config for smoke testing only")
    a = ap.parse_args()

    cfg = Config()
    if a.fast:
        cfg = fast_config(cfg)
    for name in ("data", "cache", "out", "figs", "seed"):
        v = getattr(a, name)
        if v is not None:
            setattr(cfg, {"data": "data_path", "cache": "cache_path", "figs": "fig_dir"}.get(name, name), v)
    out, figs = Path(cfg.out_dir), Path(cfg.fig_dir)
    np.random.seed(cfg.seed)

    full = load_data(cfg)
    dev, test = split_dev_test(full, cfg)
    log(f"dev: {len(dev)} rows / {dev[GROUP].nunique()} patients | test: {len(test)} rows / {test[GROUP].nunique()} patients")
    save_environment(cfg, out, dev, test)
    dev_small = dev.iloc[::cfg.row_stride].reset_index(drop=True)   # stride is a compute saver only

    if a.stage in ("eda", "all"):
        run_eda(dev, cfg, out, figs)
        plot_tsne(dev_small, cfg, figs)
    if a.stage in ("cv", "all"):
        run_cv(dev_small, cfg, out)
    if a.stage in ("report", "all"):
        make_report(cfg, out, figs)
    if a.stage == "final":
        final_test(dev_small, test, cfg, out)


if __name__ == "__main__":
    main()
