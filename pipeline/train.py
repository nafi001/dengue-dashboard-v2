"""
pipeline/train.py
====================
Trains the GA-tuned SVR models (one per horizon) and writes VERSIONED
artifacts under artifacts/v{timestamp}/. Never overwrites a previous
version in place — retrain_check.py decides whether to point the registry
at the new version after checking its test metrics aren't worse than
production.

This is the same GA-SVR approach as the original train_ga_svr_dashboard.py,
refactored to:
  - import feature engineering from pipeline.features (no more copy-paste
    drift between training and serving)
  - read from data/canonical.csv via pipeline.ingest (no more separate
    historical_data.csv you have to remember to hand it)
  - write to a fresh, timestamped artifacts/ subfolder every run

Can be run standalone:
    python -m pipeline.train

...but is normally invoked by scheduler.py after retrain_check.py decides
a retrain is warranted.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .features import HORIZONS, TARGET, add_features, make_supervised_dataset
from .ingest import load_canonical

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

STEPS_TO_REPORT = {
    7: [1, 3, 7],
    28: [1, 3, 7, 14, 21, 28],
}

TEST_FRACTION = 0.20
MIN_TRAIN_SIZE = 60

GA_POP_SIZE = 25
GA_N_GEN = 10
SVR_BOUNDS = [(1.0, 900.0), (1e-4, 1.0), (0.001, 0.2)]  # C, gamma, epsilon

ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts"


def chrono_split(n_rows: int, test_fraction: float = TEST_FRACTION):
    split_idx = int(n_rows * (1 - test_fraction))
    tr = np.zeros(n_rows, dtype=bool)
    te = np.zeros(n_rows, dtype=bool)
    tr[:split_idx] = True
    te[split_idx:] = True
    return tr, te


def clip_preds(arr):
    return np.clip(np.asarray(arr), 0, None)


class GeneticAlgorithmOptimiser:
    """Real-valued GA: tournament selection, single-point crossover,
    Gaussian mutation with annealed scale. Minimizes RMSE on a validation split."""

    def __init__(self, param_bounds, pop_size=25, n_gen=30,
                 crossover_prob=0.8, mutation_prob=0.15, tournament_k=3, seed=SEED):
        self.bounds = param_bounds
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cr = crossover_prob
        self.mr = mutation_prob
        self.k = tournament_k
        self.rng = np.random.RandomState(seed)

    def _rand_individual(self):
        return [self.rng.uniform(lo, hi) for lo, hi in self.bounds]

    def _fitness(self, ind, X_tr, y_tr, X_va, y_va, model_fn):
        try:
            m = model_fn(ind)
            m.fit(X_tr, y_tr)
            p = np.nan_to_num(m.predict(X_va), nan=1e9, posinf=1e9, neginf=1e9)
            rmse = np.sqrt(mean_squared_error(np.asarray(y_va).flatten(), np.asarray(p).flatten()))
            return rmse if np.isfinite(rmse) else 1e9
        except Exception:
            return 1e9

    def _tournament(self, pop, fits):
        idx = self.rng.choice(len(pop), self.k, replace=False)
        best = idx[int(np.argmin([fits[i] for i in idx]))]
        return pop[best][:]

    def _crossover(self, p1, p2):
        if self.rng.rand() < self.cr:
            pt = self.rng.randint(1, len(p1))
            return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
        return p1[:], p2[:]

    def _mutate(self, ind, gen):
        scale = 1.0 - 0.7 * gen / max(self.n_gen, 1)
        for k in range(len(ind)):
            if self.rng.rand() < self.mr:
                lo, hi = self.bounds[k]
                ind[k] = float(np.clip(ind[k] + (hi - lo) * scale * self.rng.randn() * 0.2, lo, hi))
        return ind

    def optimise(self, X_tr, y_tr, X_va, y_va, model_fn, verbose=True):
        pop = [self._rand_individual() for _ in range(self.pop_size)]
        best_ind, best_fit = pop[0][:], np.inf

        for gen in range(self.n_gen):
            fits = [self._fitness(p, X_tr, y_tr, X_va, y_va, model_fn) for p in pop]
            order = np.argsort(fits)
            pop = [pop[i] for i in order]
            fits = [fits[i] for i in order]

            if fits[0] < best_fit:
                best_fit, best_ind = fits[0], pop[0][:]

            if verbose and (gen % 5 == 0 or gen == self.n_gen - 1):
                print(f"    GA gen {gen + 1}/{self.n_gen}  best RMSE so far: {best_fit:.4f}")

            new_pop = pop[:2]
            while len(new_pop) < self.pop_size:
                p1, p2 = self._tournament(pop, fits), self._tournament(pop, fits)
                c1, c2 = self._crossover(p1, p2)
                new_pop += [self._mutate(c1, gen), self._mutate(c2, gen)]
            pop = new_pop[: self.pop_size]

        return best_ind, best_fit


def make_ga_svr(params):
    C, gamma, eps = params
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps)),
    ])


def eval_regression(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    if label:
        print(f"    {label}: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_all_horizons(df: pd.DataFrame | None = None, horizons=HORIZONS,
                        version: str | None = None) -> dict:
    """Train GA-SVR models for each horizon and write a new versioned
    artifact folder. Returns a dict summary including the version id and
    per-horizon mean test MAE (what retrain_check.py compares against the
    current production model before promoting).

    `df` defaults to the canonical dataset if not given explicitly — pass it
    in directly for testing without touching disk state.
    """
    if df is None:
        canonical = load_canonical()
        df = canonical[[TARGET] + [c for c in canonical.columns if c not in ("source", TARGET)]]

    # Drop weather columns that are entirely empty. load_canonical() always
    # creates a column for every WEATHER_COLS_CANDIDATES entry even if it was
    # never populated (e.g. no 'max_ta' ever ingested) — add_features would
    # otherwise treat that all-NaN column as "found" and build lag/rolling
    # features from it, which then makes every row get dropped downstream.
    df = df.dropna(axis=1, how="all")
    # Keep all rows for feature construction (weather may be present even when
    # a case count is still pending) but the supervised dataset step will
    # drop rows with any NaN in the feature/target window regardless.

    if version is None:
        version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACTS_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training version {version}")
    print(f"  Data range: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} days)")

    df_feat, weather_cols_found = add_features(df)
    print(f"  Weather columns found & expanded: {weather_cols_found}")

    all_metrics = []
    summary_by_horizon = {}

    for H in horizons:
        print(f"\n{'=' * 70}\nHORIZON = {H} days\n{'=' * 70}")
        ds = make_supervised_dataset(df_feat, horizon=H)
        n = len(ds["dates"])
        if n < MIN_TRAIN_SIZE:
            print(f"  Skipping horizon={H}, not enough rows ({n} < {MIN_TRAIN_SIZE})")
            continue

        tr_mask, te_mask = chrono_split(n)
        X_tr, X_te = ds["X"][tr_mask], ds["X"][te_mask]
        y_tr, y_te = ds["y"][tr_mask], ds["y"][te_mask]
        dates_te = ds["dates"][te_mask]

        ga_val_frac = 0.15
        ga_split = int(len(X_tr) * (1 - ga_val_frac))
        X_trg, X_vag = X_tr[:ga_split], X_tr[ga_split:]
        y_trg, y_vag = y_tr[:ga_split], y_tr[ga_split:]

        print("\n  Running GA search for SVR hyperparameters...")
        ga = GeneticAlgorithmOptimiser(SVR_BOUNDS, pop_size=GA_POP_SIZE, n_gen=GA_N_GEN, seed=SEED)

        def svr_fn(params):
            return MultiOutputRegressor(make_ga_svr(params), n_jobs=-1)

        best_params, best_rmse = ga.optimise(X_trg, y_trg, X_vag, y_vag, svr_fn)
        best_params_named = {"C": best_params[0], "gamma": best_params[1], "epsilon": best_params[2]}
        print(f"  >>> Best SVR params for H={H}: {best_params_named}  (val RMSE={best_rmse:.4f})")

        svr_model = MultiOutputRegressor(make_ga_svr(best_params), n_jobs=-1)
        svr_model.fit(X_tr, y_tr)
        pred_te = clip_preds(svr_model.predict(X_te))

        steps = STEPS_TO_REPORT.get(H, list(range(1, H + 1)))
        h_metrics = []
        for step in steps:
            step_idx = step - 1
            if step_idx >= H:
                continue
            yt, yp = y_te[:, step_idx], pred_te[:, step_idx]
            m = eval_regression(yt, yp, label=f"GA-SVR H={H} step={step}")
            row = {"horizon": H, "step": step, "model": "GA_SVR",
                   "mae": m["MAE"], "rmse": m["RMSE"], "r2": m["R2"]}
            all_metrics.append(row)
            h_metrics.append(row)

        mean_mae = float(np.mean([m["mae"] for m in h_metrics])) if h_metrics else float("nan")
        print(f"  Mean test MAE across reported steps (H={H}): {mean_mae:.3f}")

        residuals = y_te - pred_te
        residual_std = residuals.std(axis=0)

        print(f"  Refitting GA-SVR on full dataset for production (H={H})...")
        prod_model = MultiOutputRegressor(make_ga_svr(best_params), n_jobs=-1)
        prod_model.fit(ds["X"], ds["y"])

        model_path = out_dir / f"model_h{H}.pkl"
        joblib.dump(prod_model, model_path, compress=3)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  Saved {model_path.name} ({size_mb:.1f} MB)")

        pd.DataFrame({"step": range(1, H + 1), "residual_std": residual_std}).to_csv(
            out_dir / f"residual_std_h{H}.csv", index=False
        )
        with open(out_dir / f"feature_columns_h{H}.json", "w") as f:
            json.dump(ds["fcols"], f, indent=2)
        with open(out_dir / f"ga_best_params_h{H}.json", "w") as f:
            json.dump(best_params_named, f, indent=2)

        summary_by_horizon[H] = {
            "params": best_params_named,
            "mean_test_mae": mean_mae,
            "n_train_rows": int(tr_mask.sum()),
            "n_test_rows": int(te_mask.sum()),
            "feature_cols": ds["fcols"],
        }

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "test_metrics.csv", index=False)

    metadata = {
        "version": version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "model": "GA-SVR (RBF kernel, GA-tuned C/gamma/epsilon, MultiOutputRegressor)",
        "horizons_trained": list(summary_by_horizon.keys()),
        "best_params_by_horizon": {str(h): s["params"] for h, s in summary_by_horizon.items()},
        "mean_test_mae_by_horizon": {str(h): s["mean_test_mae"] for h, s in summary_by_horizon.items()},
        "last_training_date": str(df.index.max().date()),
        "n_rows_total": len(df),
    }
    with open(out_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n=== DONE: version {version} ===")
    print(f"Artifacts saved under: {out_dir.resolve()}")

    return {
        "version": version,
        "out_dir": str(out_dir),
        "summary_by_horizon": summary_by_horizon,
        "metadata": metadata,
    }


if __name__ == "__main__":
    train_all_horizons()
