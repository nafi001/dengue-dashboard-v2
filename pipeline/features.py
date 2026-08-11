"""
pipeline/features.py
=====================
The ONE feature-engineering function used everywhere: training, retraining,
and the live Streamlit forecast. This module exists specifically to kill the
old failure mode where the dashboard and the trainer each had their own copy
of `add_features()` and could silently drift apart.

Nothing in this file reads or writes files. It's pure transformation logic,
so it's cheap to unit-test and safe to import from both a batch script and a
Streamlit app.
"""

from __future__ import annotations

import pandas as pd

TARGET = "confirm_dengue"

TARGET_LAGS = [1, 2, 3, 7, 14]
TARGET_ROLL_WINDOWS = [3, 7, 14]

WEATHER_COLS_CANDIDATES = ["ta", "rha", "ra", "max_ta", "min_ta", "ws", "sr"]
WEATHER_LABELS = {
    "ta": "Avg temperature (°C)",
    "max_ta": "Max temperature (°C)",
    "min_ta": "Min temperature (°C)",
    "rha": "Relative humidity (%)",
    "ra": "Rainfall (mm)",
    "ws": "Wind speed",
    "sr": "Solar radiation",
}
WEATHER_LAGS = [1, 2, 3, 7]
WEATHER_ROLL_WINDOWS = [3, 7, 14]

HORIZONS = [7, 28]


def add_features(data: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, list[str]]:
    """Build the full feature set from a date-indexed dataframe containing at
    least `target` and (optionally) any of WEATHER_COLS_CANDIDATES.

    Returns (feature_df, weather_cols_found). Row i's target-based features
    (lags, rolling stats) use only data through day i-1 (shift(1)) to avoid
    leakage. Weather features use same-day weather as of day i directly,
    since weather is knowable in advance / same-day, unlike case counts.
    """
    d = data.copy()

    for l in TARGET_LAGS:
        d[f"lag{l}"] = d[target].shift(l)

    s = d[target].shift(1)
    for w in TARGET_ROLL_WINDOWS:
        d[f"roll_mean_{w}"] = s.rolling(w).mean()
        d[f"roll_std_{w}"] = s.rolling(w).std()
        d[f"roll_min_{w}"] = s.rolling(w).min()
        d[f"roll_max_{w}"] = s.rolling(w).max()
        d[f"roll_sum_{w}"] = s.rolling(w).sum()

    d["ema_7"] = s.ewm(span=7, adjust=False).mean()
    d["ema_14"] = s.ewm(span=14, adjust=False).mean()
    d["diff_1"] = s.diff(1)
    d["diff_7"] = s.diff(7)

    d["dayofweek"] = d.index.dayofweek
    d["month"] = d.index.month
    d["quarter"] = d.index.quarter
    d["is_monsoon"] = d.index.month.isin([6, 7, 8, 9, 10]).astype(int)
    d["is_friday"] = (d.index.dayofweek == 4).astype(int)

    weather_cols_found = [c for c in WEATHER_COLS_CANDIDATES if c in d.columns]
    for wcol in weather_cols_found:
        ws = d[wcol].shift(1)
        for l in WEATHER_LAGS:
            d[f"{wcol}_lag{l}"] = d[wcol].shift(l)
        for w in WEATHER_ROLL_WINDOWS:
            d[f"{wcol}_rmean_{w}"] = ws.rolling(w).mean()
            d[f"{wcol}_rstd_{w}"] = ws.rolling(w).std()

    return d, weather_cols_found


def make_supervised_dataset(df_features: pd.DataFrame, horizon: int, target: str = TARGET) -> dict:
    """Turn a feature dataframe into (X, y, dates, fcols) for a given horizon.
    Row i's y-vector is target[i : i+horizon] — the forecast starts on the
    SAME day as the feature row (matches how weather/features are anchored)."""
    feat_df = df_features.dropna().copy()
    fcols = [c for c in feat_df.columns if c != target]
    X = feat_df[fcols].values
    y = [feat_df[target].iloc[i: i + horizon].values for i in range(len(feat_df) - horizon)]
    import numpy as np
    y = np.array(y)
    X = X[: len(y)]
    dates = feat_df.index[: len(y)]
    return dict(X=X, y=y, dates=dates, fcols=fcols)


def latest_feature_row(history_df: pd.DataFrame, fcols: list[str]):
    """The anchor row used to produce a live forecast: the most recent row
    with a complete feature set (weather present; case count for that day
    itself may still be unknown — that's fine, it's never used to predict
    its own day)."""
    feat_df, weather_cols_found = add_features(history_df)
    feat_df = feat_df.dropna(subset=[c for c in fcols if c in feat_df.columns])
    if feat_df.empty:
        return None, weather_cols_found
    missing = [c for c in fcols if c not in feat_df.columns]
    for c in missing:
        feat_df[c] = 0.0
    row = feat_df.iloc[[-1]][fcols]
    return row, weather_cols_found


