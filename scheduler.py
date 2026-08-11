"""
scheduler.py
=============
The one script a cron job / GitHub Action runs daily. This is what makes
the system "continuous" rather than "someone has to remember to run the
training script."

Flow:
  1. (optional) pull fresh case/weather data from wherever you get it and
     ingest it via pipeline.ingest — see fetch_new_data() below, which is a
     stub you fill in with your actual data source (an API, a manually
     exported CSV dropped in a watched folder, etc).
  2. Ask pipeline.retrain_check.should_retrain() whether today's situation
     warrants a full retrain.
  3. If yes: train a new versioned artifact set, then promote it only if it
     isn't meaningfully worse than current production.
  4. If no: do nothing — the existing production model keeps serving, using
     whatever new data was ingested in step 1 as fresh forecast input (no
     retrain needed for that; the model just consumes the new row through
     the normal feature-row anchoring at forecast time).

Run manually:
    python scheduler.py                 # ingest (if configured) + check + maybe train
    python scheduler.py --force-train   # skip the check, always retrain
    python scheduler.py --check-only    # print the retrain decision, do nothing else

GitHub Actions (daily cron) — see .github/workflows/daily.yml alongside this
file for a ready-made example.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from pipeline.ingest import load_canonical  # noqa: F401  (import kept for fetch_new_data stub)
from pipeline.retrain_check import load_registry, promote_if_better, should_retrain
from pipeline.train import train_all_horizons


def fetch_new_data() -> None:
    """STUB — wire this up to your real data source.

    Whatever you plug in here should end by calling pipeline.ingest.add_row
    or pipeline.ingest.add_rows_bulk. Examples of what belongs here:
      - hit a case-surveillance API for yesterday's confirmed count
      - hit a weather API (or NASA POWER, matching your original data source)
        for yesterday's ta/rha/ra/etc.
      - or, if data still arrives as a manually exported CSV, watch a folder
        and call add_rows_bulk() on any new file found.

    Left as a no-op by default so this script is safe to run before you've
    wired up a live feed — ingestion can also happen purely through the
    Streamlit "Add / edit / delete data" tab in the meantime.
    """
    pass


def run(force_train: bool = False, check_only: bool = False, skip_ingest: bool = False) -> int:
    print(f"=== scheduler run @ {datetime.now(timezone.utc).isoformat()} ===")

    if not skip_ingest:
        print("Step 1: fetching new data (if configured)...")
        fetch_new_data()
    else:
        print("Step 1: skipped (--skip-ingest)")

    reg = load_registry()
    do_train, reason = should_retrain(reg)
    print(f"Step 2: retrain decision -> {do_train} ({reason})")

    if check_only:
        return 0

    if not (do_train or force_train):
        print("Step 3: no retrain needed. Production model stays as-is.")
        return 0

    print(f"Step 3: training new version{' (forced)' if force_train and not do_train else ''}...")
    result = train_all_horizons()

    print("Step 4: evaluating promotion...")
    reg = promote_if_better(result, reg)
    last = reg["history"][-1]
    if last["promoted"]:
        print(f"  Promoted {last['version']} to production. ({last['reason']})")
    else:
        print(f"  NOT promoted — kept {reg['production_version']} as production. "
              f"({last['reason']})")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true",
                         help="Train and evaluate promotion regardless of should_retrain().")
    parser.add_argument("--check-only", action="store_true",
                         help="Only print the retrain decision; don't train or ingest.")
    parser.add_argument("--skip-ingest", action="store_true",
                         help="Skip the fetch_new_data() step.")
    args = parser.parse_args()
    sys.exit(run(force_train=args.force_train, check_only=args.check_only,
                  skip_ingest=args.skip_ingest))
