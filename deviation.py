# deviation.py
# -----------------------------------------------------------------------------
# Full Deviation section:
#   1. Deviation Scatter + Trend + Filters + Hit Rates
#   2. Deviation Backtest (cooldown, event study, forward returns)
#
# EVERYTHING here is preserved exactly from your original UI.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    lowess,
    make_bins_hit_rate,
    forward_return,
    sharpe_ratio,
    cooldown_filter_indexed,
    event_study,
)

# -----------------------------------------------------------------------------
# 1. FULL DEVIATION SCATTER SECTION
# -----------------------------------------------------------------------------
def render_deviation_scatter_section(dev_df):

    st.header("Deviation Scatter & Trend")

    # -----------------------------------------------------------
    # Controls (unchanged)
    # -----------------------------------------------------------
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        trend_method = st.selectbox(
            "Trend Method",
            ["LOWESS", "Linear (OLS)"],
            index=0
        )

    with col2:
        apply_range = st.checkbox("Limit Deviation Range", value=False)
        if apply_range:
            left = st.number_input("Min Deviation", value=-3.0, step=0.1)
            right = st.number_input("Max Deviation", value=3.0, step=0.1)
        else:
            left, right = -999, 999

    with col3:
        bin_enable = st.checkbox("Show Binned Hit-Rate Stats", value=False)
        if bin_enable:
            bin_left = st.number_input("Bin Left", value=-5.0)
            bin_right = st.number_input("Bin Right", value=5.0)
            bin_step = st.number_input("Bin Step", value=1.0)

    # -----------------------------------------------------------
    # Filter deviation based on range
    # -----------------------------------------------------------
    df = dev_df.copy()
    mask = (df["deviation"] >= left) & (df["deviation"] <= right)
    df = df[mask]

    x = df["deviation"].values
    y = df["fwd20"].values

    # -----------------------------------------------------------
    # Build scatter plot
    # -----------------------------------------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(size=6, color="#4c78a8", opacity=0.6),
        name="20-day Return"
    ))

    # -----------------------------------------------------------
    # Trend line (LOWESS or OLS)
    # -----------------------------------------------------------
    if trend_method == "LOWESS":
        xs, ys = lowess(y, x, frac=0.3)
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(width=3, color="#d62728"),
            name="LOWESS"
        ))
    else:
        # Linear OLS
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(min(x), max(x), 200)
            ys = m * xs + b
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=3, color="#d62728"),
                name="OLS"
            ))

    fig.update_layout(
        height=500,
        xaxis_title="Deviation",
        yaxis_title="20-Day Forward Return",
        template="simple_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------
    # Optional: hit rates by bins
    # -----------------------------------------------------------
    if bin_enable:
        hr = make_bins_hit_rate(
            x=df["deviation"],
            y=df["fwd20"],
            left=bin_left,
            right=bin_right,
            step=bin_step,
        )
        st.subheader("Binned Hit-Rate Table")
        st.dataframe(
            hr.style.format({
                "count": "{:.0f}",
                "hit_rate": "{:.2%}",
                "avg_ret": "{:.2%}",
                "median_ret": "{:.2%}",
            })
        )


# -----------------------------------------------------------------------------
# 2. FULL DEVIATION BACKTEST SECTION
# -----------------------------------------------------------------------------
def render_deviation_backtest_section(dev_df):

    st.header("Deviation Backtest")

    # -----------------------------------------------------------
    # Controls
    # -----------------------------------------------------------
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        thr = st.number_input("Deviation Trigger (≥)", value=2.0, step=0.1)

    with col2:
        cooldown = st.number_input("Cooldown (days)", value=30, step=1)

    with col3:
        pre = st.number_input("Event Study Lookback (days)", value=60)
        post = st.number_input("Event Study Forward (days)", value=60)

    # -----------------------------------------------------------
    # Compute triggers
    # -----------------------------------------------------------
    df = dev_df.copy()
    df = df.dropna(subset=["deviation", "xrt"])

    raw_triggers = df.index[df["deviation"] >= thr]
    triggers = cooldown_filter_indexed(raw_triggers, df.index, cooldown)

    st.write(f"**Triggers Found:** {len(triggers)}")

    # -----------------------------------------------------------
    # Forward returns distribution from triggers
    # -----------------------------------------------------------
    returns = []
    for t in triggers:
        if t not in df.index:
            continue
        idx = df.index.get_loc(t)
        # Return from t → t+post
        if idx + post < len(df):
            r = df["xrt"].iloc[idx + post] / df["xrt"].iloc[idx] - 1
            returns.append(r)

    if returns:
        ret_series = pd.Series(returns)
        st.subheader("Forward Return Summary")
        st.write(f"Mean: {ret_series.mean():.2%}")
        st.write(f"Median: {ret_series.median():.2%}")
        st.write(f"Sharpe: {sharpe_ratio(ret_series):.2f}")

    # -----------------------------------------------------------
    # Event Study Panel
    # -----------------------------------------------------------
    if len(triggers) > 0:
        st.subheader("Event Study (Normalized Paths)")

        price = df["xrt"].astype(float)
        panel = event_study(price, triggers, pre=pre, post=post)

        if not panel.empty:
            fig = go.Figure()

            # Individual paths
            for col in panel.columns:
                fig.add_trace(go.Scatter(
                    x=panel.index,
                    y=panel[col],
                    mode="lines",
                    line=dict(width=1, color="rgba(150,150,150,0.3)"),
                    showlegend=False
                ))

            # Mean path
            mean_curve = panel.mean(axis=1)
            fig.add_trace(go.Scatter(
                x=panel.index,
                y=mean_curve,
                mode="lines",
                line=dict(width=4, color="#d62728"),
                name="Mean"
            ))

            fig.update_layout(
                height=500,
                xaxis_title="Days from Event",
                yaxis_title="Return from Event",
                template="simple_white"
            )

            st.plotly_chart(fig, use_container_width=True)

