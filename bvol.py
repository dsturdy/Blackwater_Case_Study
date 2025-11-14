# bvol.py
# -----------------------------------------------------------------------------
# Full BVOL Strategy module:
#   • Percentile rule
#   • Z-score rule
#   • Fixed hold-days logic
#   • Percent stop-loss
#   • ATR stop-loss (gap-aware)
#   • Trade table formatting (centered, prices rounded)
#   • Strategy summary: Sharpe, Hit Rate, Avg Return
#   • All Streamlit UI preserved exactly as original
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils import (
    sharpe_ratio,
)

# -----------------------------------------------------------------------------
# BVOL Percentile Trigger
# -----------------------------------------------------------------------------
def compute_percentile_trigger(bvol, lookback, threshold):
    out = pd.Series(False, index=bvol.index)

    for i in range(lookback, len(bvol)):
        window = bvol.iloc[i - lookback:i]
        pct = (window <= bvol.iloc[i]).mean() * 100

        if pct >= threshold:
            out.iloc[i] = True

    return out


# -----------------------------------------------------------------------------
# BVOL Z-Score Trigger
# -----------------------------------------------------------------------------
def compute_zscore_trigger(bvol, lookback, z_threshold):
    out = pd.Series(False, index=bvol.index)

    for i in range(lookback, len(bvol)):
        window = bvol.iloc[i - lookback:i]
        mean = window.mean()
        std = window.std()

        if std > 0 and (bvol.iloc[i] - mean) / std >= z_threshold:
            out.iloc[i] = True

    return out


# -----------------------------------------------------------------------------
# Simulate fixed hold-days return
# -----------------------------------------------------------------------------
def simulate_percent_stop(prices, entries, hold_days, stop_pct):
    rows = []
    px = prices.values
    idx = prices.index

    for t in entries:
        i = idx.get_loc(t)
        entry_price = px[i]

        exit_idx = min(i + hold_days, len(px) - 1)
        exit_price = px[exit_idx]

        # Stop-loss: if price ≤ entry * (1 - stop_pct)
        stop_trigger_price = entry_price * (1 - stop_pct)
        stop_hit = False

        # Check each bar during the holding period
        for j in range(i, exit_idx + 1):
            if px[j] <= stop_trigger_price:
                exit_price = px[j]
                stop_hit = True
                break

        ret = exit_price / entry_price - 1

        rows.append({
            "Entry": t,
            "Exit": idx[exit_idx],
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "HoldBars": exit_idx - i,
            "Return": ret,
            "Stopped": stop_hit,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# ATR calculation
# -----------------------------------------------------------------------------
def compute_atr(df, lookback=14):
    hi = df["high"]
    lo = df["low"]
    cl = df["close"].shift(1)

    tr1 = hi - lo
    tr2 = (hi - cl).abs()
    tr3 = (lo - cl).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(lookback).mean()


# -----------------------------------------------------------------------------
# ATR stop-loss simulation
# -----------------------------------------------------------------------------
def simulate_atr_stop(df, entries, hold_days, atr_mult):
    rows = []
    idx = df.index
    px_open = df["open"].values
    px_close = df["close"].values
    px_low = df["low"].values
    atr = df["ATR"].values

    for t in entries:
        i = idx.get_loc(t)
        entry_price = px_open[i]  # entry at open
        exit_idx = min(i + hold_days, len(df) - 1)
        exit_price = px_close[exit_idx]

        stop_hit = False
        stop_bar = None

        # Calculate stop price
        stop_price = entry_price - atr_mult * atr[i]

        # Check intraday lows + gap-down on open
        for j in range(i + 1, exit_idx + 1):

            # Gap-down stop
            if px_open[j] <= stop_price:
                exit_price = px_open[j]
                stop_hit = True
                stop_bar = j
                break

            # Intraday stop
            if px_low[j] <= stop_price:
                exit_price = px_low[j]
                stop_hit = True
                stop_bar = j
                break

        if stop_hit:
            final_exit = idx[stop_bar]
        else:
            final_exit = idx[exit_idx]

        ret = exit_price / entry_price - 1

        rows.append({
            "Entry": t,
            "Exit": final_exit,
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "HoldBars": (exit_idx - i),
            "Return": ret,
            "Stopped": stop_hit
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Trade table formatting helper
# -----------------------------------------------------------------------------
def format_trades_table(td):
    """Format Entry, Exit as YYYY-MM-DD and round prices/returns."""
    if "Entry" in td.columns:
        td["Entry"] = pd.to_datetime(td["Entry"]).dt.strftime("%Y-%m-%d")
    if "Exit" in td.columns:
        td["Exit"] = pd.to_datetime(td["Exit"]).dt.strftime("%Y-%m-%d")

    styler = td.style.format({
        "EntryPrice": "{:.2f}",
        "ExitPrice": "{:.2f}",
        "Return": "{:.2%}",
        "HoldBars": "{:.0f}"
    })

    # Center everything
    styler = styler.set_properties(**{"text-align": "center"})
    styler = styler.set_table_styles([{
        "selector": "th",
        "props": [("text-align", "center")]
    }])

    return styler


# -----------------------------------------------------------------------------
# BVOL Strategy Section (UI + Logic)
# -----------------------------------------------------------------------------
def render_bvol_strategy_section(bvol_df, ohlc_df):

    st.header("BVOL Short-Term Strategy")

    if bvol_df is None:
        st.warning("BVOL sheet missing — strategy unavailable.")
        return

    # -------------------------------------------------------------------------
    # UI — Rule selection
    # -------------------------------------------------------------------------
    rule = st.selectbox(
        "Signal Rule",
        [
            "BVOL crosses above percentile threshold",
            "BVOL z-score crosses above z-threshold"
        ],
        key="rule"
    )

    col1, col2, col3 = st.columns(3)

    if rule == "BVOL crosses above percentile threshold":

        with col1:
            pct_thr = st.number_input("Percentile Threshold", value=87.8, step=0.1)

        with col2:
            pct_lookback = st.number_input("Lookback (days)", value=120, step=1)

        with col3:
            hold_days = st.number_input("Hold Days", value=12, step=1)

        triggers = compute_percentile_trigger(
            bvol_df["bvol"], pct_lookback, pct_thr
        )

    else:
        with col1:
            z_thr = st.number_input("Z-Score Threshold", value=3.50, step=0.05)

        with col2:
            z_lookback = st.number_input("Lookback (days)", value=25, step=1)

        with col3:
            hold_days = st.number_input("Hold Days", value=16, step=1)

        triggers = compute_zscore_trigger(
            bvol_df["bvol"], z_lookback, z_thr
        )

    entries = bvol_df.index[triggers]
    st.write(f"**Signals Found:** {len(entries)}")

    # -------------------------------------------------------------------------
    # Stop-Loss Mode
    # -------------------------------------------------------------------------
    st.subheader("Stop-Loss Settings")

    use_atr = st.checkbox("Use ATR Stop-Loss Instead of Percent", value=False)

    if use_atr:
        if ohlc_df is None:
            st.error("ATR mode requires XRT_OHLC sheet.")
            return

        # ATR computation
        df = bvol_df.copy()
        df = df.join(ohlc_df[["open", "high", "low", "close"]], how="left")

        df["ATR"] = compute_atr(df, lookback=14)

        atr_mult = st.slider(
            "ATR Multiplier",
            min_value=1.0, max_value=5.0, step=0.1, value=2.0
        )

        td = simulate_atr_stop(df, entries, hold_days, atr_mult)

    else:
        stop_loss_pct = st.slider(
            "Percent Stop-Loss (%)",
            min_value=1.0,
            max_value=10.0,
            step=0.1,
            value=3.0,
            format="%.1f%%",
        ) / 100.0

        td = simulate_percent_stop(
            bvol_df["xrt"], entries, hold_days, stop_loss_pct
        )

    # -------------------------------------------------------------------------
    # Strategy Summary
    # -------------------------------------------------------------------------
    if td.empty:
        st.info("No trades for this configuration.")
        return

    st.subheader("Strategy Performance Summary")

    r = td["Return"]

    colA, colB, colC = st.columns(3)
    colA.write(f"**Hit Rate:** { (r > 0).mean():.2% }")
    colB.write(f"**Average Return:** { r.mean():.2% }")
    colC.write(f"**Sharpe:** { sharpe_ratio(r):.2f }")

    # -------------------------------------------------------------------------
    # Trade Table (Formatted)
    # -------------------------------------------------------------------------
    st.subheader("Historical Signals & Trades")
    st.dataframe(format_trades_table(td), use_container_width=True)

    # -------------------------------------------------------------------------
    # Plot cumulative PnL
    # -------------------------------------------------------------------------
    st.subheader("Cumulative Return")

    cum = (1 + r).cumprod() - 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=td["Exit"],
        y=cum,
        mode="lines",
        line=dict(width=3, color="#4c78a8"),
        name="Cumulative"
    ))

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="simple_white"
    )

    st.plotly_chart(fig, use_container_width=True)
