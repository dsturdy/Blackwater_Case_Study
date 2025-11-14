# bvol_optimizer.py
# Fully deterministic, real-strategy optimizer for percent & ATR mode

import numpy as np
import pandas as pd
import optuna
import math
import random

# ============================================================
# 🔒 Deterministic RNG for Optuna + NumPy + Python
# ============================================================
np.random.seed(42)
random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ======================================================================
# Real strategy stop-loss simulation
# ======================================================================

def simulate_strategy(
    bv,
    signals,
    xrt_px,
    hold_days,
    stop,
    use_atr=False,
    atr=None,
    atr_mult=None
):
    """
    Simulates the real strategy:
        • ATR mode: open entry, gap stops, intraday stops, exit at close
        • Percent mode: close entry, close-based stops
    Returns (sharpe, trades_list)
    """

    trades = []
    rets = []

    for t in signals:
        if t not in xrt_px.index:
            continue

        # ——————————————————————————
        # Build the holding window
        # ——————————————————————————
        start_ix = xrt_px.index.get_loc(t) + 1
        if start_ix >= len(xrt_px) - 1:
            continue

        end_ix = min(start_ix + hold_days, len(xrt_px) - 1)
        window_idx = xrt_px.index[start_ix:end_ix + 1]

        if len(window_idx) < 1:
            continue

        # ——————————————————————————
        # ATR MODE
        # ——————————————————————————
        if use_atr:

            entry_time = window_idx[0]
            entry_price = bv.loc[entry_time, "open"]

            stop_hit = False
            exit_price = None
            exit_time = None

            for ts in window_idx:

                if ts == entry_time:
                    continue

                o = bv.loc[ts, "open"]
                h = bv.loc[ts, "high"]
                l = bv.loc[ts, "low"]
                c = bv.loc[ts, "close"]

                atr_val = atr.get(ts, np.nan)
                if pd.isna(atr_val):
                    continue

                stop_level = entry_price - atr_val * atr_mult

                # 1) Gap-down stop
                if o <= stop_level:
                    stop_hit = True
                    exit_price = o
                    exit_time = ts
                    break

                # 2) Intraday stop
                if l <= stop_level:
                    stop_hit = True
                    exit_price = stop_level
                    exit_time = ts
                    break

            # Final exit (no stop)
            if not stop_hit:
                exit_time = window_idx[-1]
                exit_price = bv.loc[exit_time, "close"]

        # ——————————————————————————
        # PERCENT MODE
        # ——————————————————————————
        else:
            entry_time = window_idx[0]
            entry_price = float(xrt_px.loc[entry_time])

            stop_hit = False
            exit_price = None
            exit_time = None

            for ts in window_idx[1:]:
                px = float(xrt_px.loc[ts])
                ret_now = px / entry_price - 1.0

                if ret_now <= -stop:
                    stop_hit = True
                    exit_price = px
                    exit_time = ts
                    break

            if not stop_hit:
                exit_time = window_idx[-1]
                exit_price = float(xrt_px.loc[exit_time])

        final_ret = exit_price / entry_price - 1.0
        rets.append(final_ret)

        trades.append(
            {
                "Entry": entry_time,
                "Exit": exit_time,
                "Return": final_ret,
                "Stopped": stop_hit,
            }
        )

    if len(rets) < 3:
        return -999, []

    arr = np.array(rets)
    sharpe = arr.mean() / (arr.std(ddof=0) + 1e-12)
    sharpe *= math.sqrt(252 / max(hold_days, 1))

    return sharpe, trades


# ======================================================================
# 🌟 Optuna: REAL STOP-LOSS OPTIMIZER
# ======================================================================

def optimize_stoploss(
    bv,
    signals,
    xrt_px,
    hold_days,
    use_atr=False,
    atr=None,
    atr_mult_grid=None,
    n_trials=50
):
    """
    Returns:
        best_stop, best_sharpe, study, best_trades
    """

    if len(signals) == 0:
        raise ValueError("No signals provided — cannot optimize stop-loss.")

    # 🔒 deterministic ordering of signals
    signals = sorted(pd.DatetimeIndex(signals))

    # Percent mode: optimize float stop
    # ATR mode: optimize the integer ATR multiple
    if use_atr:
        search_space = atr_mult_grid
    else:
        search_space = None  # Optuna will use 0.01–0.10

    # ——————————————————————————
    # Optuna objective
    # ——————————————————————————
    def objective(trial):

        if use_atr:
            # ATR multipliers like 1.0, 1.25, 1.5, ..., 5.0
            atr_m = trial.suggest_float(
                "atr_mult",
                low=min(atr_mult_grid),
                high=max(atr_mult_grid),
                step=0.25
            )
            stop_to_test = atr_m
            stop_val = None  # percent stop not used

        else:
            stop_val = trial.suggest_float(
                "stop",
                0.01,
                0.15,
                step=0.001
            )
            atr_m = None

        sharpe, _ = simulate_strategy(
            bv=bv,
            signals=signals,
            xrt_px=xrt_px,
            hold_days=hold_days,
            stop=stop_val,
            use_atr=use_atr,
            atr=atr,
            atr_mult=atr_m
        )

        return sharpe

    # ——————————————————————————
    # Create deterministic study
    # ——————————————————————————
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)

    # ——————————————————————————
    # Extract best parameters
    # ——————————————————————————
    if use_atr:
        best_mult = study.best_params["atr_mult"]
        best_stop = best_mult
        stop_val = None

        # rerun to get trades
        best_sharpe, trades = simulate_strategy(
            bv=bv,
            signals=signals,
            xrt_px=xrt_px,
            hold_days=hold_days,
            stop=stop_val,
            use_atr=True,
            atr=atr,
            atr_mult=best_mult
        )

    else:
        best_stop = study.best_params["stop"]
        best_sharpe, trades = simulate_strategy(
            bv=bv,
            signals=signals,
            xrt_px=xrt_px,
            hold_days=hold_days,
            stop=best_stop,
            use_atr=False,
            atr=None,
            atr_mult=None
        )

    return best_stop, best_sharpe, study, pd.DataFrame(trades)
