"""
migrate_legacy.py
====================
Run this ONCE to move from the old repo layout (root-level historical_data.csv
/ production_model.pkl + dashboard_artifacts/ duplicate + live_data.csv) into
the new continuous-pipeline layout (data/canonical.csv + versioned
artifacts/v.../ + artifacts/registry.json).

What it does:
  1. Folds dashboard_artifacts/historical_data.csv (source=static) and, if
     present, dashboard_artifacts/live_data.csv (source=manual) into
     data/canonical.csv.
  2. Takes your EXISTING trained models (dashboard_artifacts/production_model_h7.pkl,
     h28.pkl + their feature_columns/residual_std/ga_best_params/metadata
     files) and copies them into a new versioned folder
     artifacts/v_migrated_YYYYMMDD/, then registers that version as
     production — so you keep serving your current models immediately
     instead of being forced into a cold retrain before the app works again.
  3. Leaves the old files in place; it only ever copies/creates.

After running this, the ongoing loop is: scheduler.py (via cron / GitHub
Actions) handles ingestion + retrain checks going forward. You can safely
delete the old root-level duplicate files (app.py, root historical_data.csv,
production_model.pkl, etc.) once you've confirmed the new streamlit_app.py
works.

Usage:
    python migrate_legacy.py --legacy-dir dashboard_artifacts
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pipeline.features import HORIZONS
from pipeline.ingest import migrate_from_legacy
from pipeline.retrain_check import load_registry, save_registry

ARTIFACTS_ROOT = Path(__file__).resolve().parent / "artifacts"


def migrate(legacy_dir: str) -> None:
    legacy = Path(legacy_dir)
    if not legacy.exists():
        raise FileNotFoundError(f"Legacy directory not found: {legacy}")

    print(f"Step 1: migrating data from {legacy} into data/canonical.csv ...")
    hist_csv = legacy / "historical_data.csv"
    live_csv = legacy / "live_data.csv"
    if not hist_csv.exists():
        raise FileNotFoundError(f"Expected {hist_csv} to exist.")
    combined = migrate_from_legacy(hist_csv, live_csv if live_csv.exists() else None)
    print(f"  canonical.csv now has {len(combined)} rows "
          f"({combined.index.min().date()} -> {combined.index.max().date()})")

    print("\nStep 2: copying existing trained models into a versioned artifact folder ...")
    version = "v_migrated_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACTS_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_any = False
    for H in HORIZONS:
        src_model = legacy / f"production_model_h{H}.pkl"
        if not src_model.exists():
            print(f"  (no production_model_h{H}.pkl found — skipping horizon {H})")
            continue
        shutil.copy2(src_model, out_dir / f"model_h{H}.pkl")
        for fname in (f"feature_columns_h{H}.json", f"residual_std_h{H}.csv", f"ga_best_params_h{H}.json"):
            src = legacy / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)
        copied_any = True
        print(f"  Copied horizon {H} model + supporting files.")

    src_metrics = legacy / "test_metrics_ga_svr.csv"
    if src_metrics.exists():
        shutil.copy2(src_metrics, out_dir / "test_metrics.csv")

    src_meta = legacy / "model_metadata.json"
    if src_meta.exists():
        meta = json.loads(src_meta.read_text())
    else:
        meta = {}
    meta.setdefault("version", version)
    meta.setdefault("trained_at_utc", datetime.now(timezone.utc).isoformat())
    meta.setdefault("mean_test_mae_by_horizon", meta.get("mean_mae_by_horizon", {}))
    (out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    if not copied_any:
        print("  WARNING: no existing production models found to migrate. "
              "You'll need to run `python scheduler.py --force-train` to "
              "train the first version instead.")
        return

    print("\nStep 3: registering this version as production ...")
    reg = load_registry()
    reg["production_version"] = version
    reg["last_trained_version"] = version
    reg["last_trained_at_utc"] = meta["trained_at_utc"]
    reg["last_check_utc"] = datetime.now(timezone.utc).isoformat()
    reg["last_training_row_count"] = int(combined["confirm_dengue"].notna().sum())
    reg.setdefault("history", []).append({
        "version": version,
        "trained_at_utc": meta["trained_at_utc"],
        "promoted": True,
        "reason": "migrated from legacy layout",
        "mean_test_mae_by_horizon": meta.get("mean_test_mae_by_horizon", {}),
    })
    save_registry(reg)
    print(f"  Registered {version} as production in artifacts/registry.json")
    print("\nDone. You can now run: streamlit run streamlit_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", default="dashboard_artifacts",
                         help="Path to the old dashboard_artifacts/ folder (default: dashboard_artifacts)")
    args = parser.parse_args()
    migrate(args.legacy_dir)
