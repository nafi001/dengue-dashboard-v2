"""
Dengue Forecast Dashboard (continuous pipeline edition)
=========================================================
Streamlit app for public-health decision-makers to view GA-SVR dengue
forecasts, understand the drivers behind them, judge how much to trust
them — and see the status of the continuous retraining pipeline behind it.

What changed from the original single-shot version:
  - ONE canonical dataset (data/canonical.csv, via pipeline.ingest) replaces
    the old historical_data.csv (frozen) + live_data.csv (editable) split.
    Every row carries a `source` column (static/manual/api) so provenance
    is still visible, but there's one place data lives.
  - The app reads a VERSIONED artifact registry (artifacts/registry.json)
    instead of a single fixed set of model files. It always serves whatever
    version is currently marked "production" — which scheduler.py updates
    automatically after a retrain clears its safety check.
  - A "Pipeline status" panel shows: which model version is live, when it
    was trained, how many rows it saw, whether a retrain is currently due,
    and a manual "Run retrain check now" button for out-of-band runs
    (the automated path is scheduler.py via GitHub Actions / cron).
  - Feature engineering is imported from pipeline.features — identical
    logic to what trained the model, guaranteed by import rather than by a
    comment asking you to keep two copies in sync.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.features import (
    HORIZONS,
    TARGET,
    WEATHER_COLS_CANDIDATES,
    WEATHER_LABELS,
    add_features,
    latest_feature_row,
)
from pipeline.ingest import ALL_VALUE_COLS, add_row, load_canonical
from pipeline.retrain_check import load_registry, should_retrain

ARTIFACTS_ROOT = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Dengue Forecast Dashboard", layout="wide", page_icon="🦟")


# ==========================================================================
# LOADERS (cached)
# ==========================================================================
@st.cache_data(ttl=300)
def load_canonical_cached():
    return load_canonical()


@st.cache_data(ttl=60)
def load_registry_cached():
    return load_registry()


@st.cache_resource
def load_model_bundle(version: str, horizon: int):
    v_dir = ARTIFACTS_ROOT / version
    model = joblib.load(v_dir / f"model_h{horizon}.pkl")
    fcols = json.loads((v_dir / f"feature_columns_h{horizon}.json").read_text())
    residual_std = pd.read_csv(v_dir / f"residual_std_h{horizon}.csv")["residual_std"].values
    ga_params = json.loads((v_dir / f"ga_best_params_h{horizon}.json").read_text())
    return model, fcols, residual_std, ga_params


@st.cache_data(ttl=60)
def load_version_metadata(version: str):
    path = ARTIFACTS_ROOT / version / "model_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data(ttl=60)
def load_version_metrics(version: str):
    path = ARTIFACTS_ROOT / version / "test_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def clear_data_caches():
    load_canonical_cached.clear()
    load_registry_cached.clear()


# ==========================================================================
# HELPERS
# ==========================================================================
def risk_badge(forecast_sum_7d: float, history_df: pd.DataFrame) -> tuple:
    s = history_df[TARGET].dropna()
    roll7 = s.rolling(7).sum().dropna()
    if len(roll7) < 10:
        return "Not enough history", "gray"
    p50, p75, p90 = np.percentile(roll7, [50, 75, 90])
    if forecast_sum_7d >= p90:
        return "High / Outbreak risk", "red"
    elif forecast_sum_7d >= p75:
        return "Elevated risk", "orange"
    elif forecast_sum_7d >= p50:
        return "Moderate", "blue"
    else:
        return "Low", "green"


def confidence_zone(mae_series: pd.Series) -> pd.Series:
    t1, t2 = np.percentile(mae_series, [33, 66])
    def label(v):
        if v <= t1:
            return "High"
        elif v <= t2:
            return "Medium"
        return "Low"
    return mae_series.apply(label)


# ==========================================================================
# LOAD BASE DATA + REGISTRY
# ==========================================================================
canonical = load_canonical_cached()
registry = load_registry_cached()
weather_cols_present = [
    c for c in WEATHER_COLS_CANDIDATES
    if c in canonical.columns and canonical[c].notna().any()
]

production_version = registry.get("production_version")

if canonical.empty or canonical[TARGET].notna().sum() == 0:
    st.error(
        "No data in `data/canonical.csv` yet. Run the one-time migration "
        "(`pipeline.ingest.migrate_from_legacy(...)`) or add rows via the "
        "'Add / edit / delete data' tab below, then train a model with "
        "`python -m pipeline.train` or `python scheduler.py --force-train`."
    )
    st.stop()

if production_version is None:
    st.error(
        "No production model in `artifacts/registry.json` yet. Run "
        "`python scheduler.py --force-train` (or `python -m pipeline.train` "
        "followed by promotion) to train and register the first model."
    )
    st.stop()

# ==========================================================================
# SIDEBAR
# ==========================================================================
st.sidebar.title("🦟 Forecast controls")

horizon = st.sidebar.radio(
    "Forecast horizon",
    options=HORIZONS,
    format_func=lambda h: f"{h} days ahead ({h} daily values)",
    help="7-day forecasts are generally more reliable; 28-day is useful for "
         "longer-range planning but carries more uncertainty (see the "
         "reliability tab below).",
)

version_meta = load_version_metadata(production_version)

st.sidebar.divider()
with st.sidebar.expander("ℹ️ About this model"):
    params = version_meta.get("best_params_by_horizon", {}).get(str(horizon), {})
    st.write(f"**Model type:** GA-tuned SVR (RBF kernel), one model per horizon")
    st.write(f"**Horizon:** {horizon} days, predicted in a single call (multi-output)")
    if params:
        st.write(
            f"**Tuned hyperparameters:** C={params.get('C', 0):.2f}, "
            f"gamma={params.get('gamma', 0):.4f}, epsilon={params.get('epsilon', 0):.4f}"
        )
    st.write(f"**Production version:** `{production_version}`")
    st.write(f"**Trained on data through:** {version_meta.get('last_training_date', 'n/a')}")
    st.write(f"**Total training days:** {version_meta.get('n_rows_total', 'n/a')}")

st.title("Dengue Case Forecast Dashboard")
st.caption(
    "For dengue surveillance and response planning — case forecasts with "
    "uncertainty ranges, the weather signals behind them, and how much to "
    "trust each part of the forecast."
)

# ==========================================================================
# 0. PIPELINE STATUS — new: makes the continuous retraining loop visible
# ==========================================================================
st.subheader("⚙️ Pipeline status")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Production model", production_version.lstrip("v"))
trained_at = registry.get("last_trained_at_utc", "n/a")
p2.metric("Last trained (UTC)", trained_at.split("T")[0] if trained_at != "n/a" else "n/a")
p3.metric("Rows seen at training", registry.get("last_training_row_count", "n/a"))

due, reason = should_retrain(registry)
p4.markdown("**Retrain due?**")
p4.markdown(f":{'orange' if due else 'green'}[**{'Yes' if due else 'No'}**]")
st.caption(f"Reason: {reason}")

with st.expander("📜 Training history"):
    hist = registry.get("history", [])
    if not hist:
        st.write("No training runs recorded yet.")
    else:
        hist_df = pd.DataFrame(hist)[["version", "trained_at_utc", "promoted", "reason"]]
        st.dataframe(hist_df.iloc[::-1], width='stretch', hide_index=True)

st.caption(
    "The automated pipeline (scheduler.py, run daily via GitHub Actions or "
    "cron) ingests new data, checks the conditions above, retrains when "
    "warranted, and only promotes a new model if it isn't meaningfully "
    "worse than the current one. This button runs the same check on demand "
    "— it does **not** train a model (training is a scheduled/CI job, not "
    "something to trigger from a page load)."
)
if st.button("🔄 Refresh pipeline status"):
    clear_data_caches()
    st.rerun()

st.divider()

# ==========================================================================
# 1. DATA — single canonical dataset, add/edit/delete
# ==========================================================================
st.subheader("🗂️ Data")
tab_summary, tab_editor = st.tabs(["Summary", "Add / edit / delete data"])

with tab_summary:
    c1, c2, c3 = st.columns(3)
    n_total = len(canonical)
    n_by_source = canonical["source"].value_counts().to_dict() if "source" in canonical.columns else {}
    c1.metric("Total rows", f"{n_total} days",
               help=f"{canonical.index.min().date()} → {canonical.index.max().date()}")
    c2.metric("Reported case-count rows", f"{int(canonical[TARGET].notna().sum())}")
    c3.metric("Most recent date", f"{canonical.index.max().date()}")
    if n_by_source:
        st.caption("By source: " + ", ".join(f"{k}: {v}" for k, v in n_by_source.items()))

with tab_editor:
    st.info(
        "**Important:** editing the case count on the most recent date will "
        "**not** move that date's own forecast step — the model never uses "
        "a day's own case count to predict that same day (that would be "
        "leakage). To reflect a newly reported count, add it as part of a "
        "**new row for the next day** — it then becomes real input (as "
        "yesterday's lag) for the next forecast. Weather is different: "
        "editing **today's weather** *does* change today's forecast "
        "immediately, since same-day weather is a valid input.\n\n"
        "Rows added here are saved with `source = manual` to "
        "`data/canonical.csv` immediately — the automated pipeline "
        "(scheduler.py) picks them up on its next scheduled run and decides "
        "whether they warrant a retrain.",
        icon="ℹ️",
    )

    if st.button("➕ Add a row for the next day (auto-fills weather from the last known day)"):
        next_date = canonical.index.max() + pd.Timedelta(days=1)
        weather_fill = {
            wcol: canonical[wcol].dropna().iloc[-1]
            if wcol in canonical.columns and canonical[wcol].notna().any() else None
            for wcol in weather_cols_present
        }
        add_row(date=next_date, confirm_dengue=None, source="manual", **{
            k: v for k, v in weather_fill.items() if v is not None
        })
        clear_data_caches()
        st.success(f"Added a row for {next_date.date()} — adjust its weather below if needed, then Save.")
        st.rerun()

    st.caption(
        "Edit any cell, add a row at the bottom, or check a row's box and "
        "press the trash icon to delete it. Leave the case-count cell blank "
        "if that day's count isn't reported yet — the model will nowcast it "
        "as part of the forecast instead of you guessing a number."
    )
    editor_cols = ["date", "source", TARGET] + weather_cols_present
    edit_source = canonical.reset_index()
    for c in editor_cols:
        if c not in edit_source.columns:
            edit_source[c] = pd.NA
    edit_source = edit_source[editor_cols]
    edit_source[TARGET] = pd.to_numeric(edit_source[TARGET], errors="coerce")
    for wcol in weather_cols_present:
        edit_source[wcol] = pd.to_numeric(edit_source[wcol], errors="coerce").astype("float64")

    column_config = {
        "date": st.column_config.DateColumn("Date", required=True),
        "source": st.column_config.SelectboxColumn("Source", options=["static", "manual", "api", "unknown"]),
        TARGET: st.column_config.NumberColumn(f"Cases ({TARGET})", min_value=0, step=1),
    }
    for wcol in weather_cols_present:
        column_config[wcol] = st.column_config.NumberColumn(
            WEATHER_LABELS.get(wcol, wcol), format="%.2f", step=0.01
        )

    edited = st.data_editor(
        edit_source,
        column_config=column_config,
        num_rows="dynamic",
        width='stretch',
        key="canonical_data_editor",
    )

    bcol1, bcol2 = st.columns([1, 3])
    if bcol1.button("💾 Save changes", type="primary", width='stretch'):
        clean = edited.dropna(subset=["date"]).copy()
        clean["date"] = pd.to_datetime(clean["date"])
        clean = clean.drop_duplicates(subset="date", keep="last").set_index("date")
        clean[TARGET] = pd.to_numeric(clean[TARGET], errors="coerce")
        for wcol in weather_cols_present:
            if wcol in clean.columns:
                clean[wcol] = pd.to_numeric(clean[wcol], errors="coerce").astype("float64")
        if "source" not in clean.columns:
            clean["source"] = "manual"
        from pipeline.ingest import save_canonical
        save_canonical(clean[["source"] + [c for c in ALL_VALUE_COLS if c in clean.columns]])
        clear_data_caches()
        st.success(f"Saved {len(clean)} row(s) to the canonical dataset.")
        st.rerun()
    bcol2.download_button(
        "⬇️ Download canonical.csv",
        data=canonical.reset_index().to_csv(index=False).encode(),
        file_name="canonical.csv", mime="text/csv", width='stretch',
    )

# ==========================================================================
# COMPUTE FORECAST — uses the production model version
# ==========================================================================
history_df = canonical[[TARGET] + weather_cols_present]
model, fcols, residual_std, ga_params = load_model_bundle(production_version, horizon)

feat_row, weather_cols_found = latest_feature_row(history_df, fcols)

if feat_row is None:
    st.error(
        "Not enough consecutive recent history to build a feature row (the "
        "model needs up to 14 prior days without gaps). Add the missing "
        "recent days in the 'Add / edit / delete data' tab above."
    )
    st.stop()

X_input = feat_row.values
raw_pred = model.predict(X_input)[0]
pred = np.clip(raw_pred, 0, None)[:horizon]
ci_width = 1.96 * residual_std[:horizon]
lower = np.clip(pred - ci_width, 0, None)
upper = pred + ci_width

last_date = feat_row.index.max()
forecast_dates = pd.date_range(last_date, periods=horizon)

forecast_df = pd.DataFrame({
    "date": forecast_dates,
    "forecast": pred,
    "lower_95": lower,
    "upper_95": upper,
})

# ==========================================================================
# 2. BASIC STATISTICS (KPI ROW)
# ==========================================================================
st.subheader("📊 Current situation")

s = history_df[TARGET].dropna()
last_val = s.iloc[-1] if len(s) else np.nan
last_reported_date = s.index.max() if len(s) else None
ma7_now = s.tail(7).mean() if len(s) >= 7 else np.nan
ma7_prev = s.tail(14).head(7).mean() if len(s) >= 14 else np.nan
pct_change_wow = ((ma7_now - ma7_prev) / ma7_prev * 100) if ma7_prev and not np.isnan(ma7_prev) and ma7_prev != 0 else np.nan

cols = st.columns(5)
latest_label = f"Latest reported ({last_reported_date.date()})" if last_reported_date is not None else "Latest reported"
cols[0].metric(latest_label, f"{last_val:.0f}" if not np.isnan(last_val) else "n/a")
cols[1].metric(
    "7-day avg cases",
    f"{ma7_now:.1f}" if not np.isnan(ma7_now) else "n/a",
    delta=f"{pct_change_wow:+.1f}% vs prior week" if not np.isnan(pct_change_wow) else None,
    delta_color="inverse",
)
forecast_sum_7 = float(np.sum(pred[:7])) if horizon >= 7 else float(np.sum(pred))
cols[2].metric(f"Forecast total, next {min(horizon,7)}d", f"{forecast_sum_7:.0f}")
risk_label, risk_color = risk_badge(forecast_sum_7, history_df)
cols[3].markdown(f"**Risk level**")
cols[3].markdown(f":{risk_color}[**{risk_label}**]")
if weather_cols_found and "ra" in weather_cols_found:
    rain_14 = history_df["ra"].tail(14).sum()
    cols[4].metric("Rainfall, last 14 days", f"{rain_14:.0f} mm")
elif weather_cols_found:
    wcol = weather_cols_found[0]
    cols[4].metric(f"{WEATHER_LABELS.get(wcol, wcol)}, 7d avg", f"{history_df[wcol].tail(7).mean():.1f}")

# ==========================================================================
# 3. FORECAST CHART + TABLE
# ==========================================================================
st.subheader(f"📈 {horizon}-day forecast with 95% interval")

recent_days = 60
hist_recent = history_df[TARGET].dropna().tail(recent_days)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_recent.index, y=hist_recent.values, mode="lines+markers",
    name="Reported cases", line=dict(color="#1f77b4", width=2), marker=dict(size=4),
))
fig.add_trace(go.Scatter(
    x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]),
    y=list(forecast_df["upper_95"]) + list(forecast_df["lower_95"][::-1]),
    fill="toself", fillcolor="rgba(255,127,14,0.18)",
    line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
    name="95% interval", showlegend=True,
))
fig.add_trace(go.Scatter(
    x=forecast_df["date"], y=forecast_df["forecast"], mode="lines+markers",
    name="Forecast", line=dict(color="#ff7f0e", width=2, dash="dash"), marker=dict(size=5),
))
peak_idx = int(np.argmax(pred))
fig.add_trace(go.Scatter(
    x=[forecast_df["date"].iloc[peak_idx]], y=[pred[peak_idx]], mode="markers+text",
    name="Forecast peak", marker=dict(size=11, color="red", symbol="star"),
    text=["peak"], textposition="top center",
))
fig.update_layout(
    height=440, margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Date", yaxis_title="Confirmed dengue cases",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, width='stretch')

c1, c2 = st.columns([2, 1])
with c1:
    show_df = forecast_df.copy()
    show_df["date"] = show_df["date"].dt.strftime("%Y-%m-%d")
    for c in ["forecast", "lower_95", "upper_95"]:
        show_df[c] = show_df[c].round(1)
    st.dataframe(show_df, width='stretch', hide_index=True)
    st.download_button(
        "⬇️ Download forecast (CSV)",
        data=show_df.to_csv(index=False).encode(),
        file_name=f"dengue_forecast_h{horizon}_{last_date.date()}.csv",
        mime="text/csv",
    )
with c2:
    st.markdown("**Planning summary**")
    st.write(f"- Peak day: **{forecast_dates[peak_idx].date()}** (~{pred[peak_idx]:.0f} cases)")
    if horizon >= 7:
        weekly = forecast_df.set_index("date")["forecast"].resample("7D").sum()
        st.write("- Weekly totals:")
        for wk_start, wk_sum in weekly.items():
            st.write(f"&nbsp;&nbsp;&nbsp;{wk_start.date()}: **{wk_sum:.0f}** cases")
    st.write(f"- Risk level vs. recent history: **{risk_label}**")

# ==========================================================================
# 4. WHY IS DENGUE RISING OR FALLING
# ==========================================================================
st.subheader("🔎 Why is dengue rising or falling right now?")

bullets = []

if not np.isnan(pct_change_wow):
    direction = "risen" if pct_change_wow > 0 else "fallen"
    bullets.append(
        f"Reported cases have **{direction} {abs(pct_change_wow):.0f}%** "
        f"(7-day average) compared with two weeks ago."
    )

pred_vs_last = (pred[0] - last_val) / last_val * 100 if last_val else np.nan
if not np.isnan(pred_vs_last):
    dir2 = "increase" if pred_vs_last > 0 else "decrease"
    bullets.append(
        f"The model's nowcast for {last_date.date()} implies a **{dir2} of "
        f"about {abs(pred_vs_last):.0f}%** versus the most recent reported day."
    )

if "ra" in weather_cols_found:
    rain = history_df["ra"].fillna(0)
    recent_rain = rain.tail(14).sum()
    prior_rain = rain.tail(28).head(14).sum()
    if prior_rain > 0:
        rain_change = (recent_rain - prior_rain) / prior_rain * 100
        if abs(rain_change) > 15:
            dirn = "higher" if rain_change > 0 else "lower"
            bullets.append(
                f"Rainfall over the last 14 days was **{abs(rain_change):.0f}% "
                f"{dirn}** than the 14 days before that. Aedes mosquito "
                f"breeding typically responds to rainfall with roughly a "
                f"1-3 week lag, so this is a leading indicator worth watching "
                f"over the coming weeks, not a same-day driver."
            )
    if len(history_df) > 60:
        rain_roll = history_df["ra"].rolling(14).sum()
        corr = rain_roll.corr(history_df[TARGET].shift(-14))
        if not np.isnan(corr) and abs(corr) > 0.2:
            bullets.append(
                f"Historically in this dataset, 14-day rainfall totals and "
                f"case counts ~14 days later have a correlation of "
                f"**r={corr:.2f}** — a descriptive association, not a proven "
                f"causal driver, but consistent with the usual rainfall-to-"
                f"breeding-to-case lag."
            )

if "rha" in weather_cols_found:
    rha_recent = history_df["rha"].tail(7).mean()
    rha_prior = history_df["rha"].tail(21).head(14).mean()
    if not np.isnan(rha_recent) and not np.isnan(rha_prior) and rha_prior != 0:
        rha_change = (rha_recent - rha_prior) / rha_prior * 100
        if abs(rha_change) > 10:
            dirn = "higher" if rha_change > 0 else "lower"
            bullets.append(
                f"Humidity has been **{abs(rha_change):.0f}% {dirn}** than "
                f"three weeks ago — humidity affects mosquito survival and "
                f"can amplify or dampen the rainfall effect above."
            )

if "ta" in weather_cols_found or "max_ta" in weather_cols_found:
    tcol = "ta" if "ta" in weather_cols_found else "max_ta"
    t_recent = history_df[tcol].tail(7).mean()
    if not np.isnan(t_recent):
        if 25 <= t_recent <= 30:
            bullets.append(
                f"Average temperature is around **{t_recent:.1f}°C**, inside "
                f"the range generally considered most favorable for Aedes "
                f"mosquito activity and dengue transmission (roughly 25-30°C)."
            )
        elif t_recent > 32:
            bullets.append(
                f"Average temperature is **{t_recent:.1f}°C**, high enough "
                f"that it may start to reduce mosquito survival despite "
                f"favoring faster larval development."
            )

if not bullets:
    bullets.append("Not enough recent data yet to characterize a clear trend or driver.")

for b in bullets:
    st.markdown(f"- {b}")

st.caption(
    "These are descriptive, rule-based observations from the recent data — "
    "they explain the context around the forecast, not the internal logic "
    "of the SVR model itself, which does not produce feature-importance scores."
)

# ==========================================================================
# 5. EXPLANATION OF THE FORECAST
# ==========================================================================
with st.expander("📘 How to read this forecast"):
    st.markdown(f"""
- **Model:** a Support Vector Regression (SVR) model, one trained for each
  horizon (7-day and 28-day), with hyperparameters (C, gamma, epsilon)
  tuned by a genetic algorithm to minimize validation error. It uses recent
  case counts (lags and rolling averages/std/min/max/sum), recent weather
  (same features, lagged), and calendar signals (day of week, month,
  monsoon season) as inputs.
- **The line** is the model's single best-guess forecast for each future day.
- **The shaded band** is an approximate 95% interval: `forecast ± 1.96 ×
  (typical historical error at that step)`. It is built from how wrong the
  model actually was on held-out test data — it widens for later days
  because errors compound the further out you forecast.
- **What it can't do:** it can't foresee events outside its training
  history — a new outbreak driver, a change in reporting practices, or an
  unprecedented weather event may make real outcomes fall outside the band.
  Treat day-20+ forecasts as directional trend guidance, not a precise count.
- **What to do with it:** short-horizon values (days 1-7) are generally
  precise enough to inform staffing and supply decisions for the coming
  week; longer-horizon values are best used to spot an emerging upward or
  downward trend early, cross-checked against the driver explanations above.
- **Continuous updates:** this model is retrained automatically as new data
  arrives (see the Pipeline status panel above) — a new version is only
  promoted to production if it isn't meaningfully worse than the one it
  would replace, so the forecast you're seeing is never from a regression.
""")

# ==========================================================================
# 6. MODEL RELIABILITY BY STEP
# ==========================================================================
st.subheader("🎯 Model reliability, step by step")

metrics_df = load_version_metrics(production_version)
if metrics_df.empty:
    st.info("No test_metrics.csv found for this version — reliability details unavailable.")
else:
    m = metrics_df[metrics_df["horizon"] == horizon].sort_values("step").copy()
    m["confidence"] = confidence_zone(m["mae"])

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=m["step"], y=m["mae"], name="MAE (avg error, cases)",
        marker_color=m["confidence"].map({"High": "#2ca02c", "Medium": "#ff7f0e", "Low": "#d62728"}),
    ))
    fig2.update_layout(
        height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Forecast step (days ahead)", yaxis_title="Mean Absolute Error (cases)",
        showlegend=False,
    )
    st.plotly_chart(fig2, width='stretch')

    best_step = m.loc[m["mae"].idxmin()]
    worst_step = m.loc[m["mae"].idxmax()]
    r1, r2 = st.columns(2)
    r1.success(f"Most reliable: **day {int(best_step['step'])}** (MAE ≈ {best_step['mae']:.1f} cases, R²={best_step['r2']:.2f})")
    r2.warning(f"Least reliable: **day {int(worst_step['step'])}** (MAE ≈ {worst_step['mae']:.1f} cases, R²={worst_step['r2']:.2f})")

    with st.expander(f"Full step-by-step table (all steps 1–{horizon})"):
        disp = m[["step", "mae", "rmse", "r2", "confidence"]].round(2)
        disp.columns = ["Step (days ahead)", "MAE", "RMSE", "R²", "Confidence"]
        st.dataframe(disp, width='stretch', hide_index=True)

    st.caption(
        "🟩 High confidence: among the model's most accurate steps for this "
        "horizon. 🟧 Medium. 🟥 Low confidence: still directionally useful, "
        "but treat the exact number loosely. Zones are relative to this "
        "model only, based on tertiles of test-set error across its own steps."
    )

# ==========================================================================
# 7. YEAR-OVER-YEAR COMPARISON
# ==========================================================================
years_covered = history_df.index.year.nunique()
if years_covered > 1:
    st.subheader("📅 Compared with the same period last year")
    same_period_last_year = history_df[TARGET].reindex(
        forecast_dates - pd.DateOffset(years=1)
    )
    if same_period_last_year.notna().sum() >= max(3, horizon // 3):
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=forecast_dates, y=pred, mode="lines", name=f"{horizon}-day forecast",
            line=dict(color="#ff7f0e", dash="dash"),
        ))
        fig3.add_trace(go.Scatter(
            x=forecast_dates, y=same_period_last_year.values, mode="lines",
            name="Same dates, last year", line=dict(color="#7f7f7f"),
        ))
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                            xaxis_title="Date", yaxis_title="Cases")
        st.plotly_chart(fig3, width='stretch')
        ly_total = same_period_last_year.dropna().sum()
        fc_total = pred[:len(same_period_last_year.dropna())].sum()
        if ly_total > 0:
            yoy = (fc_total - ly_total) / ly_total * 100
            st.write(f"Forecast total for this window is **{yoy:+.0f}%** vs. the same calendar dates last year.")
    else:
        st.caption("Not enough same-period data from last year to compare yet.")

# ==========================================================================
# 8. DOWNLOADABLE DECISION BRIEF
# ==========================================================================
st.subheader("🗒️ Decision brief")
brief_lines = [
    f"Dengue forecast brief — generated {pd.Timestamp.today().date()}",
    f"Model version: {production_version}",
    f"Horizon: {horizon} days (from {forecast_dates[0].date()} to {forecast_dates[-1].date()})",
    f"Risk level: {risk_label}",
    f"Forecast total, next {min(horizon,7)} days: {forecast_sum_7:.0f} cases",
    f"Forecast peak day: {forecast_dates[peak_idx].date()} (~{pred[peak_idx]:.0f} cases)",
    "",
    "Key observations:",
] + [f"- {b}" for b in bullets] + [
    "",
    f"Model reliability note: forecasts for the first days of this horizon are "
    f"more reliable than later days (see step-by-step reliability table in the dashboard).",
]
brief_text = "\n".join(brief_lines)
st.text_area("Copyable summary", value=brief_text, height=220)
st.download_button(
    "⬇️ Download brief (.txt)", data=brief_text.encode(),
    file_name=f"dengue_brief_{last_date.date()}.txt", mime="text/plain",
)

st.divider()
st.caption(
    "This dashboard supports, but does not replace, clinical and "
    "programmatic judgment. Forecasts are statistical estimates based on "
    "historical patterns and may not capture unprecedented events."
)
