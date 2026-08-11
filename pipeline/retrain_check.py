"""
pipeline/retrain_check.py
============================
Two jobs that didn't exist anywhere in the original system:

  1. DECIDE whether a retrain is warranted right now (should_retrain).
  2. PROMOTE a freshly trained version to "production" only if it clears a
     sanity bar against the current production model (promote_if_better) —
     giving automatic rollback safety: a bad retrain never overwrites a
     good one, because the registry pointer just doesn't move.

The registry (artifacts/registry.json) is the one file the Streamlit app
and the scheduler both read to know "which version is live right now."
Structure:
{
  "production_version": "v20260811T060000Z",
  "last_trained_version": "v20260811T060000Z",
  "last_check_utc": "...",
  "last_training_row_count": 1234,
  "last_trained_at_utc": "...",
  "history": [
    {"version": "...", "trained_at_utc": "...", "promoted": true/false,
     "reason": "...", "mean_test_mae_by_horizon": {...}}
  ]
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .features import HORIZONS, TARGET
from .ingest import load_canonical

ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts"
REGISTRY_PATH = ARTIFACTS_ROOT / "registry.json"

# ---- retrain trigger policy -------------------------------------------
# Retrain if ANY of these hold:
NEW_ROWS_TRIGGER = 7          # this many new reported-case rows since last training
SCHEDULE_DAYS_TRIGGER = 7     # or this many days elapsed since last training, regardless
DRIFT_MAE_RATIO_TRIGGER = 1.5  # or recent realized error is this many times the model's
                                # own historical test MAE (a real drift signal, not just noise)

# ---- promotion safety bar ----------------------------------------------
# A newly trained version is promoted to production only if its test MAE
# isn't worse than current production's by more than this fraction. This
# stops a single bad/unlucky retrain (weird data day, GA landing somewhere
# poor) from silently replacing a working model.
MAX_MAE_REGRESSION = 0.15  # allow up to 15% worse before refusing promotion


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {
            "production_version": None,
            "last_trained_version": None,
            "last_check_utc": None,
            "last_training_row_count": 0,
            "last_trained_at_utc": None,
            "history": [],
        }
    return json.loads(REGISTRY_PATH.read_text())


def save_registry(reg: dict) -> None:
    ARTIFACTS_ROOT.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2, default=str))


def _production_metadata(reg: dict) -> dict | None:
    v = reg.get("production_version")
    if v is None:
        return None
    meta_path = ARTIFACTS_ROOT / v / "model_metadata.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def should_retrain(reg: dict | None = None) -> tuple[bool, str]:
    """Returns (should_retrain: bool, reason: str). Pure decision logic —
    doesn't train anything itself."""
    reg = reg or load_registry()
    canonical = load_canonical()
    n_reported = int(canonical[TARGET].notna().sum())

    if reg["production_version"] is None:
        return True, "no production model exists yet"

    rows_since = n_reported - reg.get("last_training_row_count", 0)
    if rows_since >= NEW_ROWS_TRIGGER:
        return True, f"{rows_since} new reported rows since last training (>= {NEW_ROWS_TRIGGER})"

    last_trained_at = reg.get("last_trained_at_utc")
    if last_trained_at:
        days_elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_trained_at)).days
        if days_elapsed >= SCHEDULE_DAYS_TRIGGER:
            return True, f"{days_elapsed} days since last training (>= {SCHEDULE_DAYS_TRIGGER})"
    else:
        return True, "no last_trained_at_utc on record"

    drift_reason = _check_drift(reg, canonical)
    if drift_reason:
        return True, drift_reason

    return False, f"no trigger met ({rows_since} new rows, model is current)"


def _check_drift(reg: dict, canonical) -> str | None:
    """Compare the production model's own historical test MAE (h7, step 1)
    against how far off its recent 1-day-ahead forecasts actually were,
    using whatever got reported since. A lightweight, dependency-free drift
    check — not a replacement for a proper backtest, but enough to catch a
    model that's gone stale between scheduled retrains."""
    meta = _production_metadata(reg)
    if meta is None:
        return None
    baseline_mae = meta.get("mean_test_mae_by_horizon", {}).get("7")
    if baseline_mae is None or baseline_mae <= 0:
        return None

    # Realized error: for each of the last few days with a reported case
    # count, how far was it from the trailing 7-day average at the time?
    # (A cheap proxy for "the recent signal has moved a lot" without
    # re-running the model here.)
    s = canonical[TARGET].dropna()
    if len(s) < 14:
        return None
    recent_actual = s.tail(7)
    trailing_baseline = s.shift(1).rolling(7).mean().reindex(recent_actual.index)
    if trailing_baseline.isna().all():
        return None
    realized_mae = (recent_actual - trailing_baseline).abs().mean()
    if realized_mae >= DRIFT_MAE_RATIO_TRIGGER * baseline_mae:
        return (f"recent realized error ({realized_mae:.1f}) is "
                f">= {DRIFT_MAE_RATIO_TRIGGER}x the model's test MAE ({baseline_mae:.1f})")
    return None


def promote_if_better(train_result: dict, reg: dict | None = None) -> dict:
    """Given the dict returned by train.train_all_horizons(...), decide
    whether to move the registry's production pointer to it. Always records
    the attempt in history, whether promoted or not.

    Returns the updated registry dict (also persisted to disk).
    """
    reg = reg or load_registry()
    version = train_result["version"]
    new_mae = train_result["metadata"].get("mean_test_mae_by_horizon", {})

    prod_meta = _production_metadata(reg)
    promoted = True
    reason = "no existing production model to compare against"

    if prod_meta is not None:
        old_mae = prod_meta.get("mean_test_mae_by_horizon", {})
        regressions = []
        for h_str, new_m in new_mae.items():
            old_m = old_mae.get(h_str)
            if old_m is None or old_m <= 0:
                continue
            if new_m > old_m * (1 + MAX_MAE_REGRESSION):
                regressions.append(f"h{h_str}: {new_m:.2f} vs prod {old_m:.2f}")
        if regressions:
            promoted = False
            reason = f"new model worse by >{MAX_MAE_REGRESSION*100:.0f}% on: " + "; ".join(regressions)
        else:
            reason = "new model's test MAE within tolerance of (or better than) production"

    canonical = load_canonical()
    n_reported = int(canonical[TARGET].notna().sum())

    reg["last_trained_version"] = version
    reg["last_trained_at_utc"] = train_result["metadata"]["trained_at_utc"]
    reg["last_check_utc"] = datetime.now(timezone.utc).isoformat()
    if promoted:
        reg["production_version"] = version
        reg["last_training_row_count"] = n_reported
    reg.setdefault("history", []).append({
        "version": version,
        "trained_at_utc": train_result["metadata"]["trained_at_utc"],
        "promoted": promoted,
        "reason": reason,
        "mean_test_mae_by_horizon": new_mae,
    })
    reg["history"] = reg["history"][-50:]  # keep the log bounded

    save_registry(reg)
    return reg
