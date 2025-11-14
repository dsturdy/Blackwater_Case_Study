# bvol_optimizer.py
# -----------------------------------------------------------------------------
# Optuna optimizers for BVOL Strategy:
#   • Percentile rule + percent stop-loss
#   • Z-score rule   + percent stop-loss
#   • ATR-based      + ATR-multiplier stop-loss
#
# 100% logic preserved from your original implementation.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import optuna

from bvol import (
    compute_percentile_trigger,
    compute_zscore_trigger,
    simulate_percent_stop,
    simulate_atr_stop,
    compute_atr,
)
from utils import sharpe_ratio


# =============================================================================
# Helper to compute Sharpe safely
# =============================================================================
def _safe_sharpe(td):
    if td is None or len(td) == 0:
        return -999
    s = sharpe_ratio(td["Return"])
    if np.isnan(s):
        return -999
    return float(s)


# =============================================================================
# 1. Percentile rule + percent stop-loss
# =============================================================================
def optimize_percentile(bvol_df, pct_thr, pct_lookback, hold_days):
    """
    Runs Optuna search for the percent stop-loss value.
    """

    px = bvol_df["xrt"].astype(float)

    # Precompute triggers once
    triggers = compute_percentile_trigger(
        bvol_df["bvol"],
        pct_lookback,
        pct_thr,
    )
    entries = bvol_df.index[triggers]

    if len(entries) == 0:
        return None

    def objective(trial):
        stop_pct = trial.suggest_float("stop_pct", 0.01, 0.10, step=0.001)
        td = simulate_percent_stop(px, entries, hold_days, stop_pct)
        return _safe_sharpe(td)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=80, show_progress_bar=False)

    best_stop = study.best_params["stop_pct"]
    best_td = simulate_percent_stop(px, entries, hold_days, best_stop)

    result = {
        "stop": best_stop,
        "sharpe": _safe_sharpe(best_td),
        "trades": best_td,
    }

    return result


# =============================================================================
# 2. Z-score rule + percent stop-loss
# =============================================================================
def optimize_zscore(bvol_df, z_thr, z_lookback, hold_days):
    """
    Optimizes the percent stop-loss for the Z-score BVOL rule.
    """

    px = bvol_df["xrt"].astype(float)

    triggers = compute_zscore_trigger(
        bvol_df["bvol"],
        z_lookback,
        z_thr
    )
    entries = bvol_df.index[triggers]

    if len(entries) == 0:
        return None

    def objective(trial):
        stop_pct = trial.suggest_float("stop_pct", 0.01, 0.10, step=0.001)
        td = simulate_percent_stop(px, entries, hold_days, stop_pct)
        return _safe_sharpe(td)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=80, show_progress_bar=False)

    best_stop = study.best_params["stop_pct"]
    best_td = simulate_percent_stop(px, entries, hold_days, best_stop)

    result = {
        "stop": best_stop,
        "sharpe": _safe_sharpe(best_td),
        "trades": best_td,
    }

    return result


# =============================================================================
# 3. ATR-based stop-loss optimization
# =============================================================================
def optimize_atr(bvol_df, ohlc_df, hold_days):
    """
    Searches over ATR multipliers. Requires OHLC.
    """

    if ohlc_df is None:
        return None

    df = bvol_df.copy()
    df = df.join(ohlc_df[["open", "high", "low", "close"]], how="left")
    df["ATR"] = compute_atr(df, lookback=14)

    # Use the percentile rule simply as a baseline signal trigger
    # (This matches your original approach)
    triggers = df.index[df["bvol"] >= df["bvol"].rolling(120).quantile(0.878)]
    entries = triggers

    if len(entries) == 0:
        return None

    def objective(trial):
        atr_mult = trial.suggest_float("atr_mult", 1.0, 5.0, step=0.1)
        td = simulate_atr_stop(df, entries, hold_days, atr_mult)
        return _safe_sharpe(td)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=80, show_progress_bar=False)

    best_multiplier = study.best_params["atr_mult"]
    best_td = simulate_atr_stop(df, entries, hold_days, best_multiplier)

    result = {
        "atr_mult": best_multiplier,
        "sharpe": _safe_sharpe(best_td),
        "trades": best_td,
    }

    return result
