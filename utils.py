# utils.py
# -----------------------------------------------------------------------------
# Shared utility functions used by the entire case study app.
# These functions preserve ALL original logic exactly as you had it.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Lightweight LOWESS (no statsmodels) - SAME BEHAVIOR as your original
# -----------------------------------------------------------------------------
def lowess(y, x, frac=0.3):
    """
    Lightweight LOWESS approximation using rolling windows.
    Preserves your exact behavior from the main script.
    """
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]

    n = max(5, int(len(y_s) * frac))

    smoothed = (
        pd.Series(y_s)
        .rolling(n, center=True, min_periods=max(3, n // 2))
        .mean()
        .to_numpy()
    )

    return x_s, smoothed


# -----------------------------------------------------------------------------
# Forward return (same as original)
# -----------------------------------------------------------------------------
def forward_return(series, horizon):
    return series.shift(-horizon) / series - 1.0


# -----------------------------------------------------------------------------
# Hit-rate binning (exact same bin logic you used)
# -----------------------------------------------------------------------------
def make_bins_hit_rate(x, y, left=-5, right=5, step=1):
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
            hit_rate=lambda s: (s > 0).mean() if len(s) else np.nan,
            avg_ret="mean",
            median_ret="median",
        )
        .reset_index()
    )

    # Sorting logic identical to original
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


# -----------------------------------------------------------------------------
# Event study panel (unchanged from your original)
# -----------------------------------------------------------------------------
def event_study(price, triggers, pre=60, post=60):
    frames = []

    for t in triggers:
        if t not in price.index:
            continue

        idx = price.index.get_loc(t)
        start = max(0, idx - pre)
        end = min(len(price) - 1, idx + post)

        window = price.iloc[start:end + 1]
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


# -----------------------------------------------------------------------------
# Sharpe Ratio (same formula you used)
# -----------------------------------------------------------------------------
def sharpe_ratio(returns, ann_factor=math.sqrt(252)):
    r = returns.dropna()
    if r.empty or r.std(ddof=0) == 0:
        return np.nan
    return (r.mean() / r.std(ddof=0)) * ann_factor


# -----------------------------------------------------------------------------
# Cooldown filter (exact same code)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Download button helper (same behavior as original)
# -----------------------------------------------------------------------------
def download_button_for_fig(fig: go.Figure, filename: str, label: str):
    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
    st.download_button(
        label,
        data=html_str.encode("utf-8"),
        file_name=filename,
        mime="text/html"
    )
