# deviation_bvol_case_study_app.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit app that implements:
#   1) Deviation Scatter + Hit Rates
#   2) Deviation Backtest (cooldown, event-aligned charts, hit rate, Sharpe)
#   3) Short-term Trading Strategy on XRT using BVOL (signals, stats, ATR stop-loss)
# ─────────────────────────────────────────────────────────────────────────────

import io
import os
import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dataclasses import dataclass
from bvol_optimizer import optimize_stoploss
TOOLTIP_CSS = """
<style>
.tooltip {
  position: relative;
  display: inline-block;
  font-size: 14px;
  color: #555;
  cursor: help;
  border-bottom: 1px dotted #888;
}

.tooltip .tooltiptext {
  visibility: hidden;
  width: 260px;
  background-color: #2c2c2c;
  color: #fff;
  text-align: left;
  border-radius: 6px;
  padding: 10px 12px;
  position: absolute;
  z-index: 10;
  bottom: 125%; 
  left: 50%;
  margin-left: -130px;
  opacity: 0;
  transition: opacity 0.3s;
  font-size: 13px;
  line-height: 1.35;
}

.tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}
</style>
"""

st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)


st.set_page_config(page_title="Deviation & BVOL Case Study", layout="wide")


PLOT_BG = "#ffffff"
ACCENT_RED = "#d62728"
BLACK = "#000000"
RED = '#FF0000'
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColMap:
    date: str
    deviation: str | None = None
    price: str | None = None
    ret: str | None = None
    fwd20: str | None = None



def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def lowess(y, x, frac=0.3):
    """Lightweight LOWESS approximation (no statsmodels required)."""
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]

    # window size proportional to frac
    n = max(5, int(len(y_s) * frac))

    smth = (
        pd.Series(y_s)
        .rolling(n, center=True, min_periods=max(3, n // 2))
        .mean()
        .to_numpy()
    )

    return x_s, smth



def forward_return(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-horizon) / series - 1.0


def make_bins_hit_rate(x: pd.Series, y: pd.Series, left=-5, right=5, step=1):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    bins = np.arange(left, right + step, step)
    df["bin"] = pd.cut(df["x"], bins=bins, right=False, include_lowest=True)
    df["range"] = df["bin"].astype(str)
    df.loc[df["range"] == "nan", "range"] = "Outside range"

    agg = (
        df.groupby("range")["y"]
        .agg(
            count="count",
            hit_rate=lambda s: (s > 0).mean() if s.size else np.nan,
            avg_ret="mean",
            median_ret="median",
        )
        .reset_index()
    )

    def _sort_key(val):
        if val == "Outside range":
            return (1e9,)
        try:
            a, _ = val.strip("[]()").split(",")
            return (float(a),)
        except Exception:
            return (0,)

    agg = agg.sort_values(by="range", key=lambda s: s.map(_sort_key))
    return agg


def zscore(s: pd.Series, lookback: int) -> pd.Series:
    return (s - s.rolling(lookback).mean()) / s.rolling(lookback).std(ddof=0)


def event_study(price: pd.Series, triggers: pd.DatetimeIndex, pre=60, post=60):
    frames = []
    for t in triggers:
        if t not in price.index:
            continue
        idx = price.index.get_loc(t)
        start = max(0, idx - pre)
        end = min(len(price) - 1, idx + post)
        window = price.iloc[start : end + 1]
        base = price.loc[t]
        curve = window / base - 1.0
        rel_ix = np.arange(start - idx, end - idx + 1)
        s = pd.Series(curve.values, index=rel_ix)
        s.name = t
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, axis=1)
    full_index = np.arange(-pre, post + 1)
    panel = panel.reindex(full_index)
    return panel


def sharpe_ratio(returns: pd.Series, ann_factor=math.sqrt(252)):
    r = returns.dropna()
    if r.empty or r.std(ddof=0) == 0:
        return np.nan
    return (r.mean() / r.std(ddof=0)) * ann_factor


def cooldown_filter_indexed(dates, full_index, cooldown_bars=30):
    dates = pd.DatetimeIndex(sorted(dates))
    kept = []
    last_bar = -10**9

    for d in dates:
        if d not in full_index:
            continue
        bar = full_index.get_loc(d)
        if bar - last_bar >= cooldown_bars:
            kept.append(d)
            last_bar = bar
    return pd.DatetimeIndex(kept)


def download_button_for_fig(fig, filename: str, label: str):
    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
    st.download_button(label, data=html_str.encode("utf-8"), file_name=filename, mime="text/html")

def optimize_atr_stoploss(bv, atr, signals, xrt_px, hold_days, atr_grid):
    results = []

    for mult in atr_grid:
        rets = []

        for t in signals:
            if t not in xrt_px.index:
                continue

            start_ix = xrt_px.index.get_loc(t) + 1
            end_ix = min(start_ix + hold_days, len(xrt_px) - 1)
            window_idx = xrt_px.index[start_ix:end_ix+1]

            entry_time = window_idx[0]
            entry_price = bv.loc[entry_time, "open"]

            exit_price = None
            stop_hit = False

            for ts in window_idx:
                if ts == entry_time:
                    continue

                o = bv.loc[ts, "open"]
                h = bv.loc[ts, "high"]
                l = bv.loc[ts, "low"]

                atr_val = atr.loc[ts]
                if np.isnan(atr_val):
                    continue

                stop_level = entry_price - atr_val * mult

                # gap-stop
                if o <= stop_level:
                    exit_price = o
                    stop_hit = True
                    break

                # intraday-stop
                if l <= stop_level:
                    exit_price = stop_level
                    stop_hit = True
                    break

            if exit_price is None:
                exit_price = bv.loc[window_idx[-1], "close"]

            rets.append(exit_price / entry_price - 1.0)

        if len(rets):
            arr = np.array(rets)
            sharpe = (arr.mean() / (arr.std(ddof=0) + 1e-12)) * np.sqrt(252 / hold_days)
            results.append((mult, sharpe, arr.mean()))

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────

st.title("Deviation & BVOL Case Study")


with st.expander("Methodology Notes"):
    st.markdown(
"""
Deviation Scatter
- Computes 20-day forward return as:  
- All deviation readings and returns are plotted  
- Trend can be fitted using:  
  - **Linear OLS** (global slope)  
  - **LOWESS** (local non-parametric smoothing)  
- Range filters and percentile/quartile filters highlight tail behavior  

Deviation Backtest
- Trigger occurs when Deviation crosses a threshold (default: **≥ +2.0**)  
- A 30-day wait period ensures only independent events are used  
- For each trigger:  
  - Forward curves are plotted relative to trigger date  
  - 60 day forward results summarized via hit rate & Sharpe  

BVOL Short-Term Strategy
- Signals when BVOL spikes based on:  
  - Percentile threshold (e.g., > 87.8th percentile)  
  - Z-score threshold (e.g., z > 3.50)  
- Exits occur via the user's choice:  
  - Fixed % stop-loss, or  
  - ATR-based stop-loss, accounting for:  
    - Gap-downs at the open  
    - Intraday lows breaking the ATR level  
- All trades compiled into trade-level & strategy-level statistics  
"""
    )

data_path = "BW_Test_Pack_Data.xlsx"



try:
    xls = pd.ExcelFile(data_path)
except Exception as e:
    st.error(f"Could not load Excel file: {e}")
    st.stop()


# Deviation sheet
if "Deviation" not in xls.sheet_names:
    st.error("Workbook must contain sheet named 'Deviation'.")
    st.stop()

dev_df = pd.read_excel(xls, "Deviation")
dev_df.columns = [c.strip() for c in dev_df.columns]

required_dev = ["Date", "XRT Price", "Deviation"]
missing = [c for c in required_dev if c not in dev_df.columns]
if missing:
    st.error(f"'Deviation' sheet missing required columns: {missing}")
    st.stop()

# BVOL sheet
if "BVOL" in xls.sheet_names:
    bvol_df = pd.read_excel(xls, "BVOL")
elif "BW Test Pack Data" in xls.sheet_names:
    bvol_df = pd.read_excel(xls, "BW Test Pack Data")
else:
    st.error("Workbook must contain sheet 'BVOL' or 'BW Test Pack Data'.")
    st.stop()

bvol_df.columns = [c.strip() for c in bvol_df.columns]

required_bvol = ["Date", "XRT Price", "Bvol"]
missing = [c for c in required_bvol if c not in bvol_df.columns]
if missing:
    st.error(f"BVOL sheet missing required columns: {missing}")
    st.stop()

# OHLC sheet
ohlc_df = None
if "XRT_OHLC" in xls.sheet_names:
    ohlc_df = pd.read_excel(xls, "XRT_OHLC")
    ohlc_df.columns = [c.strip() for c in ohlc_df.columns]

    # Accept BOTH "Date" and "date"
    date_col = None
    for c in ohlc_df.columns:
        if c.lower() == "date":
            date_col = c
            break

    if date_col is None:
        st.warning("XRT_OHLC sheet exists but has no 'date' column (case-insensitive).")
        ohlc_df = None
    else:
        # Normalize column name to lowercase "date"
        ohlc_df = ohlc_df.rename(columns={date_col: "date"})

        # Convert all other columns to lowercase as well
        ohlc_df.columns = [c.lower() for c in ohlc_df.columns]

        ohlc_df["date"] = pd.to_datetime(ohlc_df["date"], errors="coerce")
        ohlc_df = ohlc_df.dropna(subset=["date"]).sort_values("date")



# ─────────────────────────────────────────────────────────────
# Standardize and Prepare DataFrames (FINAL FIX)
# ─────────────────────────────────────────────────────────────

# 1. Clean column names exactly once
dev_df.columns = [c.strip() for c in dev_df.columns]
bvol_df.columns = [c.strip() for c in bvol_df.columns]
if ohlc_df is not None:
    ohlc_df.columns = [c.strip() for c in ohlc_df.columns]

# 2. Rename master schemas
dev = dev_df.rename(columns={
    "Date": "date",
    "XRT Price": "xrt",
    "Deviation": "deviation",
})
bvol = bvol_df.rename(columns={
    "Date": "date",
    "XRT Price": "xrt",
    "Bvol": "bvol",
})

# 3. Parse dates (mandatory BEFORE set_index)
dev["date"] = pd.to_datetime(dev["date"], errors="coerce")
bvol["date"] = pd.to_datetime(bvol["date"], errors="coerce")

# 4. Convert numerics
for col in ["xrt", "deviation"]:
    dev[col] = pd.to_numeric(dev[col], errors="coerce")

for col in ["xrt", "bvol"]:
    bvol[col] = pd.to_numeric(bvol[col], errors="coerce")

# 5. Drop rows without dates before indexing
dev = dev.dropna(subset=["date"]).sort_values("date")
bvol = bvol.dropna(subset=["date"]).sort_values("date")

# 6. FINAL: set dev index (date definitely exists here)
dev = dev.set_index("date").sort_index()

# 7. Build price series AFTER index is healthy
price_series = dev["xrt"].astype(float)
dev["fwd20"] = forward_return(price_series, 20)

# ─────────────────────────────────────────────────────────────
# Merge OHLC into BVOL safely
# ─────────────────────────────────────────────────────────────

# -------------------------------------------------------------------------
# FIXED MERGE BLOCK — correct date handling + guaranteed OHLC merge
# -------------------------------------------------------------------------

if ohlc_df is not None:
    # normalize column names
    ohlc_df.columns = [c.strip().lower() for c in ohlc_df.columns]

    # ensure date column exists
    if "date" not in ohlc_df.columns:
        st.error("XRT_OHLC sheet found but *no 'date' column* detected.")
        ohlc_df = None
    else:
        # force datetime + remove timezone
        ohlc_df["date"] = pd.to_datetime(ohlc_df["date"], errors="coerce").dt.tz_localize(None)

# Normalize bvol dates
bvol["date"] = pd.to_datetime(bvol["date"], errors="coerce").dt.tz_localize(None)

# Merge only if OHLC exists

# ──────────────────────────────────────────────
# REINDEX BVOL ONTO THE OHLC CALENDAR
# (ATR requires continuous OHLC data)
# ──────────────────────────────────────────────

if ohlc_df is not None:

    # Clean OHLC
    ohlc_df["date"] = pd.to_datetime(ohlc_df["date"], errors="coerce")
    ohlc_df = ohlc_df.dropna(subset=["date"]).sort_values("date")
    ohlc_df.columns = [c.lower() for c in ohlc_df.columns]
    ohlc_df = ohlc_df.set_index("date")

    # Clean BVOL index
    bvol["date"] = pd.to_datetime(bvol["date"], errors="coerce")
    bvol = bvol.dropna(subset=["date"]).sort_values("date")
    bvol = bvol.set_index("date")

    # Align BVOL to full OHLC trading calendar
    bvol = bvol.reindex(ohlc_df.index)

    # Pull OHLC fields into BVOL
    for col in ["open", "high", "low", "close", "volume"]:
        bvol[col] = ohlc_df[col]

    # Convert OHLC to numeric
    for col in ["open", "high", "low", "close"]:
        bvol[col] = pd.to_numeric(bvol[col], errors="coerce")

# Remove any dupes
bvol = bvol[~bvol.index.duplicated(keep="first")]


# ─────────────────────────────────────────────────────────────────────────────
# 1) Deviation Scatter + Hit Rates
# ─────────────────────────────────────────────────────────────────────────────

st.header("1. Deviation Scatter & Hit Rates")


scatter_col, hits_col = st.columns([2, 1])

with scatter_col:
    x = pd.to_numeric(dev["deviation"], errors="coerce")
    y = pd.to_numeric(dev["fwd20"], errors="coerce")
    base = pd.DataFrame({"Deviation": x, "Fwd20d": y}).dropna()

    # Compute deciles & quartiles
    ranks = base["Deviation"].rank(pct=True)
    base["decile"] = np.ceil(ranks * 10).clip(1, 10).astype(int)
    base["quartile"] = np.ceil(ranks * 4).clip(1, 4).astype(int)

    filt_mode = st.selectbox(
        "Filter",
        [
            "All (with quartile shading)",
            "Top vs Bottom Decile",
            "Top vs Bottom Quartile",
            "Top Half vs Bottom Half",
        ],
        index=0,
        key="scatter_filter_mode",
    )

    st.markdown("""
    <div style="
        width:100%;
        display:flex;
        justify-content:center;
        margin: 1px 0 1px 0;
    ">
        <div class="tooltip" style="
            background:#ffffff;
            padding:10px 22px;
            border-radius:40px;
            font-size:14px;
            color:#31333F;
            border:#dcdcdc;
        ">
            ℹ︎ Interactive Controls
            <span class="tooltiptext" style="width:300px;">
                <b>Note:</b> All charts, tables, and strategy results in this app are interactive.<br><br>
                You can adjust:
                <ul style="margin-top:6px;">
                    <li><b>Deviation filters</b> — limit the range, quartiles, or bins</li>
                    <li><b>Trend method</b> — LOWESS vs Linear (OLS)</li>
                    <li><b>BVOL signal rules</b> — percentile or z-score</li>
                    <li><b>Lookback windows</b> for BVOL signals</li>
                    <li><b>Hold period</b> and <b>stop-loss settings</b></li>
                    <li><b>Time horizons</b> on deviation event studies etc.</li>
                </ul>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    LEFT, RIGHT = -5.0, 5.0
    xr = st.slider(
        "Limit deviation range shown",
        min_value=LEFT,
        max_value=RIGHT,
        value=(LEFT, RIGHT),
        step=0.5,
        key="dev_range",
    )
    df_xy = base[(base["Deviation"] >= xr[0]) & (base["Deviation"] <= xr[1])].copy()

    if filt_mode == "All (with quartile shading)":
        df_xy["color_label"] = "All"
        title_suffix = "All points (quartile shading)"

    elif filt_mode == "Top vs Bottom Decile":
        mask = df_xy["decile"].isin([1, 10])
        df_xy = df_xy.loc[mask].copy()
        df_xy["color_label"] = np.where(df_xy["decile"] == 10, "Top 10%", "Bottom 10%")
        title_suffix = "Top 10% vs Bottom 10%"

    elif filt_mode == "Top vs Bottom Quartile":
        mask = df_xy["quartile"].isin([1, 4])
        df_xy = df_xy.loc[mask].copy()
        df_xy["color_label"] = np.where(df_xy["quartile"] == 4, "Top 25%", "Bottom 25%")
        title_suffix = "Top 25% vs Bottom 25%"

    else:  # Top Half vs Bottom Half
        med = df_xy["Deviation"].median()
        df_xy["color_label"] = np.where(df_xy["Deviation"] >= med, "Top 50%", "Bottom 50%")
        title_suffix = "Top 50% vs Bottom 50%"

    st.markdown("""
    <div style="
        width:100%;
        display:flex;
        justify-content:center;
        margin: 1px 0 1px 0;
    ">
        <div class="tooltip" style="
            background:#ffffff;
            padding:10px 22px;
            border-radius:40px;
            font-size:14px;
            color:#31333F;
            border:#dcdcdc;
        ">
            ℹ︎ Trend Method
            <span class="tooltiptext" style="width:300px;">
                <b>LOWESS Smoothing</b><br>
                • A non-parametric local regression<br>
                • Excellent for detecting curved trends<br>
                • More robust when scatter plot has clusters or heteroskedasticity<br><br>
                <b>Linear (OLS)</b><br>
                • Simple straight-line fit<br>
                • Best for monotonic relationships<br>
                • Faster and easier to interpret
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    trend_mode = st.selectbox(
        "Trend line",
        ["Linear (OLS)", "LOWESS (local)"],
        index=0,
        key="trend_scatter",
    )
    # Choose dashed vs solid
    if trend_mode == "Linear (OLS)":
        dash_style = "solid"
    else:
        dash_style = "dash"

    if "quartile" in df_xy.columns:
        df_xy["quartile_label"] = pd.Categorical(
            ["Q" + str(int(q)) for q in df_xy["quartile"]],
            categories=["Q1", "Q2", "Q3", "Q4"],
            ordered=True,
        )

    color_field = (
        "quartile_label"
        if (filt_mode == "All (with quartile shading)") and ("quartile_label" in df_xy.columns)
        else ("color_label" if "color_label" in df_xy.columns else None)
    )

    palette = {
        "Q1": "#1F77B4",
        "Q2": "#FF7F0E",
        "Q3": "#2CA02C",
        "Q4": BLACK,
        "Top 10%": "#1F77B4",
        "Bottom 10%": BLACK,
        "Top 25%": "#1F77B4",
        "Bottom 25%": BLACK,
        "Top 50%": "#1F77B4",
        "Bottom 50%": BLACK,
        "All": "#1F77B4",
    }

    scatter_traces = []
    groups = df_xy[color_field].unique() if color_field else ["All"]

    for group in groups:
        mask = df_xy[color_field] == group if color_field else [True] * len(df_xy)
        scatter_traces.append(
            go.Scatter(
                x=df_xy.loc[mask, "Deviation"],
                y=df_xy.loc[mask, "Fwd20d"],
                mode="markers",
                marker=dict(size=6, color=palette.get(group, "#1F77B4"), opacity=1.0),
                name=str(group),
                legendgroup=str(group),
                showlegend=True,
            )
        )

    # Trend line
    if trend_mode.startswith("Linear"):
        xx = df_xy["Deviation"].to_numpy()
        yy = df_xy["Fwd20d"].to_numpy()
        m, b = np.polyfit(xx, yy, 1)
        xs = np.linspace(xx.min(), xx.max(), 200)
        ys = m * xs + b
    else:
        xs, ys = lowess(
            df_xy["Fwd20d"].values,
            df_xy["Deviation"].values,
            frac=0.25,
        )

    trend_trace = go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        name=trend_mode,
        line=dict(color=RED, dash="dash", width=3),
    )

    fig = go.Figure(scatter_traces + [trend_trace])

    fig.update_layout(
        title=f"Forward 20-Day Return vs Deviation — {title_suffix}",
        xaxis_title="Deviation",
        yaxis_title="Forward 20-Day Return",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        legend_title=None,
        height=740,
        width=980,
        margin=dict(t=70, r=20, b=60, l=70),
    )
    fig.update_xaxes(range=[LEFT, RIGHT], zeroline=True, showgrid=True)
    fig.update_yaxes(tickformat=".2%", zeroline=True, showgrid=True)

    st.plotly_chart(fig, use_container_width=True)
    download_button_for_fig(fig, "deviation_scatter.html", "Download scatter as HTML")


# ─────────────────────────────────────────────────────────────────────────────
# Hit rates
# ─────────────────────────────────────────────────────────────────────────────

with hits_col:

    left, right = -5, 5
    step = 1
    hits = make_bins_hit_rate(df_xy["Deviation"], df_xy["Fwd20d"], left, right, step)
    st.subheader("Hit Rate by Deviation Bin")
    st.dataframe(
        hits.style.format(
            {"hit_rate": "{:.1%}", "avg_ret": "{:.2%}", "median_ret": "{:.2%}"}
        )
    )
    fig_hr = px.bar(hits, x="range", y="hit_rate", title="% Positive by Bin", template="plotly_white")
    fig_hr.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig_hr, use_container_width=True)
    download_button_for_fig(fig_hr, "hit_rates_by_bin.html", "Download hit-rate chart")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Deviation Backtest (+2.0 with 30-day cooldown)
# ─────────────────────────────────────────────────────────────────────────────

st.header("2. Deviation Backtest (+2.0 with 30-day wait period)")


cooldown_days = st.number_input(
    "Wait Period (trading days)",
    min_value=1,
    max_value=60,
    value=30,
)

pre_days = st.slider("Event window: days before trigger", 0, 120, 60)
post_days = st.slider("Event window: days after trigger", 30, 180, 60)

side = st.selectbox(
    "Trigger direction",
    ["Momentum (≥ +T)", "Mean Reversion (≤ -T)"],
    index=0,
)

thr = st.number_input("Deviation Level", min_value=0.0, value=2.0, step=0.1)

dev_series = dev["deviation"]

# Trigger selection
if side == "Momentum (≥ +T)":
    trig_mask = dev_series >= thr
    side_label = f"≥ +{thr:g}"
elif side == "Mean Reversion (≤ -T)":
    trig_mask = dev_series <= -thr
    side_label = f"≤ -{thr:g}"


trigs_raw = dev_series.index[trig_mask]
trigs = cooldown_filter_indexed(trigs_raw, dev.index, cooldown_days)

# Ensure valid windows
valid_trigs = []
for t in trigs:
    i = price_series.index.get_loc(t)
    window = price_series.iloc[i - pre_days : i + post_days + 1]

    if len(window) < (pre_days + post_days + 1):
        continue
    if window.isna().any():
        continue
    if i + post_days >= len(price_series):
        continue

    valid_trigs.append(t)

trigs = pd.DatetimeIndex(valid_trigs)

if len(trigs) == 0:
    st.warning(f"No {side_label} deviation triggers found with current mapping.")
else:
    st.caption(f"Found {len(trigs)} triggers after wait period.")

panel = event_study(price_series, trigs, pre=pre_days, post=post_days)

if panel.empty:
    st.warning("Insufficient data to build event-aligned panel.")
else:
    if side == "abs ≥ T (direction-normalized)":
        signs = pd.Series(
            np.where(dev.loc[panel.columns, "deviation"] < 0, -1.0, 1.0),
            index=panel.columns,
        )
        panel = panel.mul(signs, axis=1)

    # Event Study Plot
    fig_ev = go.Figure()

    # Individual lines
    for col in panel.columns:
        fig_ev.add_trace(
            go.Scatter(
                x=panel.index,
                y=panel[col],
                mode="lines",
                line=dict(color="rgba(150,150,150,0.25)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    mean_curve = panel.mean(axis=1)
    q_lo = panel.quantile(0.1, axis=1)
    q_hi = panel.quantile(0.9, axis=1)

    # Confidence band
    fig_ev.add_trace(
        go.Scatter(
            x=list(q_lo.index) + list(q_hi.index[::-1]),
            y=list(q_lo.values) + list(q_hi.values[::-1]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="10–90% range",
            hoverinfo="skip",
        )
    )

    # Mean line
    fig_ev.add_trace(
        go.Scatter(
            x=mean_curve.index,
            y=mean_curve.values,
            mode="lines",
            name="Average",
            line=dict(color="#1F77B4", width=3),
        )
    )

    fig_ev.add_hline(y=0, line=dict(color="#aaa", width=1))
    fig_ev.update_layout(
        title="Historical and Forward Trigger Performance",
        xaxis_title="Days from Trigger",
        yaxis_title="% Change",
        template="plotly_white",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        height=520,
        margin=dict(t=60, r=20, b=60, l=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_ev.update_yaxes(tickformat=".2%")
    st.plotly_chart(fig_ev, use_container_width=True)
    download_button_for_fig(fig_ev, "backtest_event_avg_band.html", "Download event-study chart")

    # +60 Day Returns Summary
    if post_days >= 60 and 60 in panel.index:
        per_event_60 = panel.loc[60].dropna()
        df60 = per_event_60.reset_index()
        df60.columns = ["date", "ret"]
        df60 = df60.sort_values("date")
        df60["label"] = df60["date"].dt.strftime("%Y-%m-%d")
        df60["sign"] = np.where(df60["ret"] >= 0, "Positive", "Negative")

        hit_rate_60 = (per_event_60 > 0).mean()

        fig_bar60 = px.bar(
            df60, x="label", y="ret", color="sign",
            color_discrete_map={"Positive": "#2CA02C", "Negative": "#D62728"},
            title="Each Trigger's 60d Return",
            template="plotly_white"
        )
        fig_bar60.update_traces(
            hovertemplate="%{x}<br>+60d: %{y:.2%}<extra></extra>",
            marker_line_width=0,
        )
        fig_bar60.update_layout(
            xaxis_type="category",
            xaxis_tickangle=-45,
            bargap=0.15,
            height=480,
            margin=dict(t=60, r=20, b=100, l=70),
        )
        fig_bar60.update_yaxes(tickformat=".1%")

        st.plotly_chart(fig_bar60, use_container_width=True)
        download_button_for_fig(fig_bar60, "backtest_plus60_bars_bydate.html", "Download +60d bars")


        # Sharpe per trade
        sharpes = []
        for c in panel.columns:
            v = panel[c].dropna()
            V = 1.0 + v
            r = V.pct_change()
            r = r.loc[(r.index >= 1) & (r.index <= 60)]
            sharpes.append(sharpe_ratio(r))

        s_ser = pd.Series(sharpes, index=panel.columns, name="Sharpe(+60d)").dropna()
        avg_sharpe = float(np.nanmean(s_ser.values)) if len(s_ser) else np.nan

        st.metric("Average Sharpe", f"{avg_sharpe:.2f}")

        dfS = s_ser.reset_index()
        dfS.columns = ["date", "sharpe"]
        dfS = dfS.sort_values("date")
        dfS["label"] = dfS["date"].dt.strftime("%Y-%m-%d")
        dfS["sign"] = np.where(dfS["sharpe"] >= 0, "Positive", "Negative")

        fig_sharpe = px.bar(
            dfS,
            x="label",
            y="sharpe",
            color="sign",
            color_discrete_map={"Positive": "#2CA02C", "Negative": "#D62728"},
            title="Each Trigger's 60d Sharpe",
            template="plotly_white",
        )
        fig_sharpe.update_traces(
            hovertemplate="%{x}<br>Sharpe: %{y:.2f}<extra></extra>",
            marker_line_width=0,
        )
        fig_sharpe.update_layout(
            xaxis_type="category",
            xaxis_tickangle=-45,
            bargap=0.15,
            height=480,
            margin=dict(t=60, r=20, b=100, l=70),
        )

        st.plotly_chart(fig_sharpe, use_container_width=True)
        download_button_for_fig(fig_sharpe, "sharpe_per_trade_bars_bydate.html", "Download Sharpe bars")

    else:
        st.info("Increase post window to ≥60 to display +60d charts.")


# Signal-level summary (Deviation Backtest)
signal_summary_rows = []

for trig in trigs:
    date = trig
    dev_val = dev.loc[date, "deviation"]
    price_at_trigger = price_series.loc[date]

    if -pre_days in panel.index:
        ret_minus = panel.loc[-pre_days, date]
    else:
        ret_minus = np.nan

    if post_days in panel.index:
        ret_plus60 = panel.loc[post_days, date]
        hit_60 = float(ret_plus60 > 0)
    else:
        ret_plus60 = np.nan
        hit_60 = np.nan

    sharpe_val = (
        s_ser.loc[date]
        if ("s_ser" in locals() and date in s_ser.index)
        else np.nan
    )

    signal_summary_rows.append(
        {
            "TriggerDate": date,
            "Deviation": dev_val,
            "Price_at_Trigger": price_at_trigger,
            "Minus60d_Return": ret_minus,
            "Plus60d_Return": ret_plus60,
            "Plus60d_Hit": hit_60,
            "Sharpe_60d": sharpe_val,
        }
    )

signal_summary_df = pd.DataFrame(signal_summary_rows).sort_values("TriggerDate")
signal_summary_df["TriggerDate"] = signal_summary_df["TriggerDate"].dt.strftime("%Y-%m-%d")

st.subheader("Signal-Level Summary (Deviation Backtest)")
st.dataframe(
    signal_summary_df.style.format(
        {
            "Deviation": "{:.2f}",
            "Price_at_Trigger": "{:.2f}",
            "Minus60d_Return": "{:.2%}",
            "Plus60d_Return": "{:.2%}",
            "Plus60d_Hit": "{:.0%}",
            "Sharpe_60d": "{:.2f}",
        }
    )
)

st.download_button(
    "Download Signal Summary CSV",
    data=signal_summary_df.to_csv(index=False).encode(),
    file_name="deviation_backtest_signal_summary.csv",
    mime="text/csv",
)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Short-Term Strategy on XRT using BVOL
# ─────────────────────────────────────────────────────────────────────────────

if "z_thr" not in st.session_state:
    st.session_state["z_thr"] = 3.50

st.header("3. Trading Strategy with BVOL Data")

# 1) Always initialize the rule FIRST
if "rule" not in st.session_state:
    st.session_state["rule"] = "BVOL crosses above percentile threshold"

rule = st.session_state["rule"]

rule = st.selectbox(
    "Signal Rule",
    [
        "BVOL crosses above percentile threshold",
        "BVOL z-score crosses above z-threshold"
    ],
    key="rule"
)

# 2) Initialize defaults *based on the rule*
if rule == "BVOL crosses above percentile threshold":
    # Percentile mode defaults
    if "pct_thr" not in st.session_state:
        st.session_state["pct_thr"] = 87.8
    if "pct_lookback" not in st.session_state:
        st.session_state["pct_lookback"] = 120
    if "hold_days" not in st.session_state:
        st.session_state["hold_days"] = 12

else:  # Z-score mode
    if "z_thr" not in st.session_state:
        st.session_state["z_thr"] = 3.50
    if "z_lookback" not in st.session_state:
        st.session_state["z_lookback"] = 25
    if "hold_days" not in st.session_state:
        st.session_state["hold_days"] = 16


bv = bvol.copy()


st.subheader("BVOL Strategy Settings")

signals = []

# Percentile mode
if st.session_state["rule"] == "BVOL crosses above percentile threshold":
    pct_thr = st.session_state["pct_thr"]
    pct_lb = st.session_state["pct_lookback"]

    rolling_pct = bv["bvol"].rolling(pct_lb).quantile(pct_thr / 100)
    sig_mask = (bv["bvol"] > rolling_pct.shift(1)) & (bv["bvol"].shift(1) <= rolling_pct.shift(2))
    signals = list(bv.index[sig_mask])

# Z-score mode
else:
    z_thr = st.session_state["z_thr"]
    z_lb = st.session_state.get("z_lookback", 60)

    z = (bv["bvol"] - bv["bvol"].rolling(z_lb).mean()) / bv["bvol"].rolling(z_lb).std()
    sig_mask = (z > z_thr) & (z.shift(1) <= z_thr)
    signals = list(bv.index[sig_mask])

xrt_px = bv["xrt"].astype(float)

# -------------------------------
# Signal Mode Selector
# -------------------------------


# -------------------------------------------------------
# FORCE-RESET DEFAULTS WHEN USER SWITCHES MODES
# -------------------------------------------------------

if "last_rule" not in st.session_state:
    st.session_state["last_rule"] = rule

# If user changed rule mode → reset defaults
if st.session_state["last_rule"] != rule:

    if rule == "BVOL crosses above percentile threshold":
        # Percentile mode defaults
        st.session_state["hold_days"] = 12
        st.session_state["pct_thr"] = 87.8
        st.session_state["pct_lookback"] = 120
        # Clear z-mode stuff
        st.session_state.pop("z_thr", None)
        st.session_state.pop("z_lookback", None)

    if rule == "BVOL z-score crosses above z-threshold":
        # Z-score mode defaults
        st.session_state["hold_days"] = 16
        st.session_state["z_thr"] = 3.50
        st.session_state["z_lookback"] = 25
        # Clear percentile mode stuff
        st.session_state.pop("pct_thr", None)
        st.session_state.pop("pct_lookback", None)

    # Update mode memory
    st.session_state["last_rule"] = rule
    st.rerun()

# ============================================================
# HOLDING PERIOD — DIFFERENT DEFAULTS PER MODE
# ============================================================
use_atr = st.checkbox("Use OHLC + ATR-based stop-loss", value=False)
hold_days = st.session_state.get("hold_days", 12)

if rule == "BVOL crosses above percentile threshold":
    # Percentile defaults to 20 if not set
    if "hold_days" not in st.session_state:
        st.session_state["hold_days"] = 12

else:
    # Z-score defaults to 10 if not set
    if "hold_days" not in st.session_state:
        st.session_state["hold_days"] = 16

# Holding period slider (shared)
hold_days = st.slider(
    "Holding Period (days)",
    5, 40,
    key="hold_days"
)

# ============================================================
# Percentile Mode Sliders
# ============================================================
if rule == "BVOL crosses above percentile threshold":

    if "pct_thr" not in st.session_state:
        st.session_state["pct_thr"] = 87.8
    if "pct_lookback" not in st.session_state:
        st.session_state["pct_lookback"] = 120
    pct_thr = st.slider(
        "Percentile threshold (0–100)",
        1.0, 100.0,
        step=0.05,
        key="pct_thr"
    )
    pct_lookback = st.slider(
        "Percentile lookback (days)",
        20, 252,
        key="pct_lookback"
    )

# ============================================================
# Z-score Mode Sliders
# ============================================================
else:

    if "z_thr" not in st.session_state:
        st.session_state["z_thr"] = 3.50
    if "z_lookback" not in st.session_state:
        st.session_state["z_lookback"] = 25
    z_thr = st.slider(
        "Z-threshold",
        0.0, 4.0,
        step=0.05,
        key="z_thr"
    )
    z_lookback = st.slider(
        "Z-score lookback (days)",
        20, 252,
        key="z_lookback"
    )

# -------------------------------
# ATR SETTINGS
# -------------------------------


if use_atr:

    # ---------------------------
    # UI CONTROLS
    # ---------------------------
    atr_len = st.slider("ATR Lookback (days)", 5, 30, 15)

    st.subheader("ATR Stop-Loss Optimization")

    atr_grid = [round(x, 2) for x in np.arange(1.0, 5.0 + 0.001, 0.1)]

    # ---------------------------
    # REAL ATR CALCULATION
    # ---------------------------
    required_cols = {"high", "low", "close", "open"}
    missing = required_cols - set(bv.columns)
    if missing:
        st.error(f"ATR mode missing OHLC: {missing}")
        st.stop()

    high = pd.to_numeric(bv["high"], errors="coerce")
    low = pd.to_numeric(bv["low"], errors="coerce")
    close = pd.to_numeric(bv["close"], errors="coerce")
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(atr_len).mean().shift(1)
    atr = atr.fillna(method="bfill").fillna(method="ffill")

    # ---------------------------
    # SIGNALS
    # ---------------------------
    if st.session_state["rule"] == "BVOL crosses above percentile threshold":
        pct_thr = st.session_state["pct_thr"]
        pct_lb = st.session_state["pct_lookback"]

        rolling_pct = bv["bvol"].rolling(pct_lb).quantile(pct_thr / 100)
        sig_mask = (bv["bvol"] > rolling_pct.shift(1)) & (bv["bvol"].shift(1) <= rolling_pct.shift(2))
        signals = list(bv.index[sig_mask])

    else:
        z_thr = st.session_state["z_thr"]
        z_lb = st.session_state["z_lookback"]

        z = (bv["bvol"] - bv["bvol"].rolling(z_lb).mean()) / bv["bvol"].rolling(z_lb).std()
        sig_mask = (z > z_thr) & (z.shift(1) <= z_thr)
        signals = list(bv.index[sig_mask])

    signals = pd.DatetimeIndex(signals)

    if use_atr and atr is not None:
        signals = signals[signals.isin(atr.index[~atr.isna()])]

    # ---------------------------
    # PRICE SERIES
    # ---------------------------

    # ---------------------------
    # OPTIMIZE BUTTON
    # ---------------------------
    if use_atr and st.button("Optimize ATR Stop-Loss"):
        xrt_px = bv["xrt"].astype(float)

        # ATR multiplier search space (your current grid)
        atr_mult_grid = [round(x, 2) for x in np.arange(1.0, 5.0 + 0.001, 0.25)]

        best_mult, best_sharpe, study, best_trades = optimize_stoploss(
            bv=bv,
            signals=signals,
            xrt_px=xrt_px,
            hold_days=hold_days,
            use_atr=True,
            atr=atr,  # <-- REQUIRED ATR SERIES
            atr_mult_grid=atr_mult_grid,  # <-- REQUIRED
            n_trials=50
        )

        st.session_state["optimized_atr_mult"] = best_mult
        st.success(f"Optimal ATR × {best_mult:.2f} | Sharpe = {best_sharpe:.2f}")
        st.rerun()

    # ---------------------------
    # FINAL SLIDER (uses optimized mult)
    # ---------------------------
    atr_mult = st.slider(
        "ATR Stop Multiplier",
        1.0, 5.0,
        value=st.session_state.get("optimized_atr_mult", 2.0),
        step=0.1,
        key="atr_mult"
    )


from bvol_optimizer import optimize_stoploss

# ---- Run Optuna ----
if st.button("Run Stop-Loss Optimization") and not use_atr:

    # Build xrt price series
    xrt_px = bv["xrt"].astype(float)

    # Call optimizer
    best_stop, best_sharpe, study, best_trades = optimize_stoploss(
        bv=bv,
        signals=signals,                      # <-- REQUIRED
        xrt_px=xrt_px,                        # <-- REQUIRED
        hold_days=hold_days,
        use_atr=False,
        atr=None,
        atr_mult_grid=None,
        n_trials=50
    )

    # Save result
    st.session_state["optimized_stop"] = best_stop
    st.success(f"Optimal stop = {best_stop:.2%} | Sharpe = {best_sharpe:.2f}")
    st.rerun()


# ---- Retrieve saved stop-loss if exists ----
opt_stop = st.session_state.get("optimized_stop", None)

# =============================================================
# Stop-Loss Slider UI
# =============================================================

if use_atr:
    stop_vals = [atr_mult]

else:
    # Raw Optuna result (decimal)
    raw_opt = st.session_state.get("optimized_stop", None)

    if raw_opt is not None:
        # Convert decimal --> percent
        slider_default_pct = raw_opt * 100

        # HARD CLAMP
        if slider_default_pct < 1.0:
            slider_default_pct = 1.0
        if slider_default_pct > 15.0:
            slider_default_pct = 15.0

    else:
        slider_default_pct = 5.0

    # Percent slider (1%–15%)
    stop_loss_pct = st.slider(
        "Stop-Loss (%)",
        min_value=1.0,
        max_value=15.0,
        value=float(slider_default_pct),
        step=0.1,
        key="custom_stop",
        format="%.1f%%"
    )

    # Convert percent → decimal
    stop_loss = stop_loss_pct / 100.0
    stop_vals = [stop_loss]

# Copy

# ATR Calculation (uses high, low, close)
# --- DEBUG: Show columns actually present before ATR check ---

if use_atr:
    required_cols = {"high", "low", "close", "open"}
    missing = required_cols - set(bv.columns)
    if missing:
        st.error(f"ATR mode is enabled but OHLC columns are missing: {missing}")
        atr = None
    else:
        high = pd.to_numeric(bv["high"], errors="coerce")
        low = pd.to_numeric(bv["low"], errors="coerce")
        close = pd.to_numeric(bv["close"], errors="coerce")

        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1
        ).max(axis=1)

        # ATR belongs to the *previous* bar
        atr = tr.rolling(atr_len).mean().shift(1)

        # Fill missing ATR values so early signals get valid ATR
        atr = atr.fillna(method="bfill").fillna(method="ffill")
else:
    atr = None

xrt_px = bv["xrt"].astype(float)
# =============================================================
# Determine final stop-loss values to test (single value)
# =============================================================


# ─────────────────────────────────────────────────────────────────────────────
# Signal Rules
# ─────────────────────────────────────────────────────────────────────────────




results = []
all_trades_tables = {}

# ─────────────────────────────────────────────────────────────────────────────
# Core Simulation — FULLY REWRITTEN WITH ATR + OPEN GAP LOGIC
# ─────────────────────────────────────────────────────────────────────────────

# -------------------------------------------------------------
# BUILD SIGNALS BEFORE RUNNING STOP-LOSS GRID
# -------------------------------------------------------------


# Must convert to DatetimeIndex for safe merging
# convert signals

for stop in stop_vals:

    stop_label = f"ATR x {atr_mult}" if use_atr else stop
    trades_table = []
    rets = []

    for t in signals:

        if t not in xrt_px.index:
            continue
        if use_atr and t not in bv.index:
            continue

        start_ix = xrt_px.index.get_loc(t) + 1
        if start_ix >= len(xrt_px) - 1:
            continue
        end_ix = min(start_ix + hold_days, len(xrt_px) - 1)

        window_idx = xrt_px.index[start_ix:end_ix+1]
        close_ser = xrt_px.loc[window_idx]

        # ATR mode requires full OHLC rows
        if use_atr and any(ts not in bv.index for ts in window_idx):
            continue

        exit_time = None
        exit_price = None
        stop_hit = False

        # --------------------------
        # ENTRY PRICE
        # --------------------------
        entry_time = window_idx[0]

        if use_atr:
            entry_price = bv.loc[entry_time, "open"]
            entry_ts = entry_time
        else:
            entry_price = close_ser.iloc[0]

        # --------------------------
        # MAIN LOOP OVER BARS
        # --------------------------
        for ts in window_idx:

            if use_atr:

                # Skip stop check on entry bar
                if ts == entry_ts:
                    continue

                o = bv.loc[ts, "open"]
                h = bv.loc[ts, "high"]
                l = bv.loc[ts, "low"]
                c = bv.loc[ts, "close"]

                atr_val = atr.get(ts, np.nan)
                if pd.isna(atr_val):
                    continue

                stop_level = entry_price - atr_val * atr_mult

                # Gap-down stop
                if o <= stop_level:
                    stop_hit = True
                    exit_time = ts
                    exit_price = o
                    break

                # Intraday stop
                if l <= stop_level:
                    stop_hit = True
                    exit_time = ts
                    exit_price = stop_level
                    break

                # Continue holding
                continue

            else:
                # PERCENT STOP MODE (uses XRT price only)

                # Skip the entry bar
                if ts == entry_time:
                    continue

                # Use BVOL/XRT price (not OHLC close!)
                c = bv.loc[ts, "xrt"]

                ret_now = c / entry_price - 1.0

                if ret_now <= -stop:
                    stop_hit = True
                    exit_time = ts
                    exit_price = c
                    break

        # --------------------------
        # SET EXIT PRICE
        # --------------------------
        if not stop_hit:
            exit_time = window_idx[-1]
            if use_atr:
                exit_price = bv.loc[exit_time, "close"]  # OHLC exit in ATR mode
            else:
                exit_price = bv.loc[exit_time, "xrt"]  # XRT exit in percent mode

        exit_ret = exit_price / entry_price - 1.0
        rets.append(exit_ret)

        trades_table.append(
            {
                "Entry": entry_time,
                "EntryPrice": float(entry_price),
                "Exit": exit_time,
                "ExitPrice": float(exit_price),
                "Trade Length": xrt_px.index.get_loc(exit_time) - xrt_px.index.get_loc(entry_time),
                "Return": float(exit_ret),
                "Stopped": bool(stop_hit),
            }
        )

    all_trades_tables[stop] = pd.DataFrame(trades_table)

    if len(rets) == 0:
        continue

    arr = np.array(rets)
    results.append(
        {
            "stop": stop_label,
            "trades": len(arr),
            "avg": float(arr.mean()),
            "median": float(np.median(arr)),
            "sharpe": float((arr.mean() / (arr.std(ddof=0)+1e-12)) * math.sqrt(252/hold_days)),
            "stopped_trades": int((np.array([t['Stopped'] for t in trades_table])).sum()),
            "stop_hits": float((np.array([t['Stopped'] for t in trades_table])).mean()),
            "pos_rate": float((arr > 0).mean()),
        }
    )


    all_trades_tables[stop] = pd.DataFrame(trades_table)

# ─────────────────────────────────────────────────────────────────────────────
# Results Overview + Stop Optimization
# ─────────────────────────────────────────────────────────────────────────────

res_df = pd.DataFrame(results)

if res_df.empty:
    st.warning("No strategy results (check mappings and BVOL settings).")
else:
    if use_atr:
        # Only one row in ATR mode
        row = res_df.iloc[0]
    else:
        # Normal % stop-loss optimization (no chart)
        best_row = res_df.sort_values(["sharpe", "avg"], ascending=False).iloc[0]


    st.markdown("""
    **Interpreting stop-loss optimization**
    - Changes Stop-Loss with the objective of maximizing sharpe
    - Each stop level is tested across all signals  
    - Wider stops = fewer forced exits but larger downside risk  
    - ATR stops adjust dynamically to current volatility  
    """)

    st.subheader("Stop-Loss Optimization (overview)")
    # Determine correct stop column
    stop_col = "stop"

    show_cols = [stop_col, "avg", "median", "sharpe", "trades", "stopped_trades", "stop_hits", "pos_rate"]

    st.dataframe(
        res_df[show_cols].set_index(stop_col).style.format(
            {
                "avg": "{:.2%}",
                "median": "{:.2%}",
                "sharpe": "{:.2f}",
                "stop_hits": "{:.1%}",
                "pos_rate": "{:.1%}",
            }
        )
    )

# ─────────────────────────────────────────────────────────────
# Drill-Down Trade Tables
# ─────────────────────────────────────────────────────────────

stop_keys = sorted(all_trades_tables.keys())

# --------- ATR MODE DRILLDOWN ----------
# --------- UNIFIED STOP KEY HANDLING (works for ATR or percent stops) ----------
label_map = {}

if use_atr:
    # ATR mode → stop_keys contains ONE element (string, e.g. "ATR x 2.0")
    label = f"ATR x {atr_mult}"
    label_map[label] = stop_keys[0]
    default_label = label

else:
    # Percent stop mode → stop_keys contains floats like 0.02, 0.03, etc.
    for k in stop_keys:
        if isinstance(k, str):
            # Shouldn't happen in percent mode but safe
            label = k
        else:
            label = f"{-k:.2%}"   # Format float as "-3.00%"
        label_map[label] = k

    # Compute best row safely
    if len(res_df):
        best_row = res_df.sort_values(["sharpe", "avg"], ascending=False).iloc[0]
        best_label = f"{best_row['stop']:.2%}"
        default_label = best_label if best_label in label_map else list(label_map.keys())[0]
    else:
        default_label = list(label_map.keys())[0]

# ----- AUTO-SELECT ONLY STOP VALUE -----

# Only one stop exists:
#   • ATR mode → stop key like "ATR x 2.0"
#   • Percent mode → float like 0.03
selected_stop_key = list(all_trades_tables.keys())[0]
td = all_trades_tables[selected_stop_key].copy()

if td.empty:
    st.info("No trades for this stop.")
else:
    st.subheader("Historical Signals & Trades")

    # --- Clean date formatting ---
    if "Entry" in td.columns:
        td["Entry"] = pd.to_datetime(td["Entry"]).dt.strftime("%Y-%m-%d")
    if "Exit" in td.columns:
        td["Exit"] = pd.to_datetime(td["Exit"]).dt.strftime("%Y-%m-%d")

    # --- Display with proper rounding ---
    st.dataframe(
        td.style.format({
            "EntryPrice": "{:.2f}",
            "ExitPrice": "{:.2f}",
            "Return": "{:.2%}",
            "HoldBars": "{:.0f}",
        })
    )


    st.download_button(
        "Download trades CSV",
        data=td.to_csv(index=False).encode(),
        file_name=f"bvol_xrt_trades_{str(selected_stop_key).replace('%','pct')}.csv",
        mime="text/csv",
    )


