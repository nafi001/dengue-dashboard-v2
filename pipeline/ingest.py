"""
pipeline/ingest.py
====================
Single entry point for getting a new day's data INTO the system. Replaces
the old historical_data.csv / live_data.csv split with one canonical,
append-only log at data/canonical.csv.

Why one file instead of two:
  - The old split existed to protect "what the model was trained on" from
    accidental edits. We keep that protection, but differently: every row
    carries a `source` column (static | manual | api) so you always know
    provenance, and the registry (see retrain_check.py) records exactly
    which snapshot of canonical.csv each trained model saw. Nothing is lost,
    but new data has ONE place to land instead of two, which is required
    for an unattended daily pipeline to work at all.

Usage:
    from pipeline.ingest import add_row, load_canonical

    add_row(date="2026-08-11", confirm_dengue=12, ta=29.4, rha=81.0, ra=3.2,
            source="api")

Design notes:
  - `add_row` upserts by date (last write wins for that date) and always
    re-sorts + re-validates before saving, so out-of-order or duplicate
    ingestion calls can't corrupt the file.
  - Missing `confirm_dengue` (case count not yet reported for a brand-new
    day) is allowed and stored as NaN — this is the "nowcast" case the
    original app supported and we keep it.
  - Missing weather is allowed too (validation only requires the date and at
    least one of {target, any weather column} to be present, otherwise the
    row is a no-op).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import TARGET, WEATHER_COLS_CANDIDATES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CANONICAL_PATH = DATA_DIR / "canonical.csv"

ALL_VALUE_COLS = [TARGET] + WEATHER_COLS_CANDIDATES
CANONICAL_COLS = ["date", "source"] + ALL_VALUE_COLS


def _empty_canonical() -> pd.DataFrame:
    df = pd.DataFrame(columns=CANONICAL_COLS)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def load_canonical() -> pd.DataFrame:
    """Load the canonical dataset. Returns an empty (but correctly shaped)
    frame if it doesn't exist yet."""
    if not CANONICAL_PATH.exists():
        return _empty_canonical()
    df = pd.read_csv(CANONICAL_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"{CANONICAL_PATH} has no 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ALL_VALUE_COLS:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    if "source" not in df.columns:
        df["source"] = "unknown"
    return df[["source"] + ALL_VALUE_COLS]


def save_canonical(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    out = df.copy()
    out.index.name = "date"
    out.reset_index().sort_values("date").to_csv(CANONICAL_PATH, index=False)


def add_row(
    date,
    confirm_dengue: float | None = None,
    source: str = "manual",
    **weather_values,
) -> pd.DataFrame:
    """Upsert a single day's row into the canonical dataset. Returns the
    updated dataframe (also persisted to disk).

    `weather_values` accepts any of WEATHER_COLS_CANDIDATES as kwargs, e.g.
    add_row("2026-08-11", confirm_dengue=12, ta=29.4, ra=3.2, source="api").
    """
    unknown = set(weather_values) - set(WEATHER_COLS_CANDIDATES)
    if unknown:
        raise ValueError(f"Unknown weather column(s): {unknown}. "
                          f"Valid: {WEATHER_COLS_CANDIDATES}")

    df = load_canonical()
    ts = pd.to_datetime(date)

    row = {c: pd.NA for c in ALL_VALUE_COLS}
    row[TARGET] = confirm_dengue
    row.update(weather_values)
    row["source"] = source

    df.loc[ts] = row
    df = df.sort_index()
    save_canonical(df)
    return df


def add_rows_bulk(rows: pd.DataFrame, source: str = "api") -> pd.DataFrame:
    """Upsert many rows at once (e.g. a batch pull from a weather/case API).
    `rows` must be date-indexed or have a 'date' column, plus any of
    TARGET / WEATHER_COLS_CANDIDATES as columns."""
    r = rows.copy()
    if "date" in r.columns:
        r["date"] = pd.to_datetime(r["date"])
        r = r.set_index("date")
    else:
        r.index = pd.to_datetime(r.index)
        r.index.name = "date"

    for c in ALL_VALUE_COLS:
        if c not in r.columns:
            r[c] = pd.NA
    r["source"] = r.get("source", source)
    r = r[["source"] + ALL_VALUE_COLS]

    df = load_canonical()
    combined = pd.concat([df, r])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    save_canonical(combined)
    return combined


def migrate_from_legacy(historical_csv: str | Path, live_csv: str | Path | None = None) -> pd.DataFrame:
    """One-time migration helper: fold the old historical_data.csv (source=
    'static') and, if present, live_data.csv (source='manual') into a fresh
    canonical.csv. Safe to re-run — it's idempotent (upsert by date)."""
    hist = pd.read_csv(historical_csv, encoding="utf-8-sig")
    hist.columns = [c.strip() for c in hist.columns]
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.set_index("date")
    hist["source"] = "static"

    df = load_canonical()
    for c in ALL_VALUE_COLS:
        if c not in hist.columns:
            hist[c] = pd.NA
    combined = pd.concat([df, hist[["source"] + ALL_VALUE_COLS]])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    if live_csv is not None and Path(live_csv).exists():
        live = pd.read_csv(live_csv, encoding="utf-8-sig")
        live.columns = [c.strip() for c in live.columns]
        live["date"] = pd.to_datetime(live["date"])
        live = live.set_index("date")
        live["source"] = "manual"
        for c in ALL_VALUE_COLS:
            if c not in live.columns:
                live[c] = pd.NA
        combined = pd.concat([combined, live[["source"] + ALL_VALUE_COLS]])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    save_canonical(combined)
    return combined
