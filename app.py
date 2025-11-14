# app.py
# ---------------------------------------------------------------------
# Main Streamlit app for:
#   1) Deviation Scatter + Hit Rates
#   2) Deviation Backtest
#   3) BVOL Short-Term Strategy (Percentile or Z-Score)
#   4) Stop-Loss Optimization (Percent or ATR)
#
# This file keeps ALL original UI, sliders, tooltips, and charts.
# Logic is split into modules for faster boot and cleaner deployment.
# ---------------------------------------------------------------------

import streamlit as st
import pandas as pd

from data_loader import load_case_study_data
from deviation import (
    render_deviation_scatter_section,
    render_deviation_backtest_section,
)
from bvol import (
    render_bvol_strategy_section,
)

# ---------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Deviation & BVOL Case Study",
    layout="wide"
)

# ---------------------------------------------------------------------
# Title + Methodology Notes Expander (UNCHANGED)
# ---------------------------------------------------------------------
st.title("Deviation & BVOL Case Study")

with st.expander("Methodology Notes"):
    st.markdown("""
**Deviation Scatter**
- Computes 20-day forward return as  
- Trend methods:
  - **Linear OLS** (global slope)
  - **LOWESS** (local smoothing)
- Range filters, quartiles, and bins reveal distribution behavior  

**Deviation Backtest**
- Trigger = Deviation ≥ +T or ≤ -T  
- 30-day cooldown ensures independent signals  
- Event study normalizes forward paths  
- Hit rates, distribution summaries, Sharpe  

**BVOL Short-Term Strategy**
- Signals triggered using:
  - BVOL percentile threshold
  - BVOL z-score threshold
- Exit via fixed stop-loss or ATR stop-loss:
  - Intraday & gap-down logic included
- Optimization (Optuna) finds stop-loss with max Sharpe
""")

# ---------------------------------------------------------------------
# DATA LOADING (relative path)
# ---------------------------------------------------------------------

DATA_PATH = "data/BW_Test_Pack_Data.xlsx"

dev_df, bvol_df, ohlc_df = load_case_study_data(DATA_PATH)

if dev_df is None:
    st.error("Could not load case study data. Ensure the Excel file exists.")
    st.stop()

# ---------------------------------------------------------------------
# 1) DEVIATION SCATTER + HIT RATES
# ---------------------------------------------------------------------
render_deviation_scatter_section(dev_df)

# ---------------------------------------------------------------------
# 2) DEVIATION BACKTEST
# ---------------------------------------------------------------------
render_deviation_backtest_section(dev_df)

# ---------------------------------------------------------------------
# 3) BVOL STRATEGY + STOP-LOSS OPTIMIZATION
# ---------------------------------------------------------------------
render_bvol_strategy_section(bvol_df, ohlc_df)

