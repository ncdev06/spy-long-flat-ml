from __future__ import annotations

import pandas as pd


def make_trend_gate(
    close: pd.Series,
    lookback: int = 120,
    slope_window: int = 10,
    min_slope: float = -0.0005,
) -> pd.Series:
    """Return a simple trend regime gate using information available at each close."""
    close = close.astype(float).sort_index()
    sma = close.rolling(lookback).mean()
    slope = sma.pct_change(slope_window)
    return ((close > sma) & (slope > min_slope)).fillna(False)


def backtest_long_flat(
    close: pd.Series,
    p_up: pd.Series,
    threshold: float = 0.55,
    exit_threshold: float | None = None,
    gate: pd.Series | None = None,
    slippage_bps: float = 1.0,
    commission_bps: float = 0.0,
) -> pd.DataFrame:
    """Backtest a long/flat signal with next-bar execution.

    A probability observed at date t determines the desired position for the
    following close-to-close return. This explicit one-bar delay prevents the
    strategy from earning the return used to generate the signal itself.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    if exit_threshold is None:
        exit_threshold = max(0.0, threshold - 0.01)
    if exit_threshold > threshold:
        raise ValueError("exit_threshold should not exceed threshold")

    frame = pd.concat(
        [close.astype(float).rename("close"), p_up.astype(float).rename("p_up")],
        axis=1,
    ).dropna()

    if frame.empty:
        raise ValueError("No overlapping close prices and predictions to backtest.")

    if gate is None:
        gate_aligned = pd.Series(True, index=frame.index)
    else:
        gate_aligned = gate.reindex(frame.index).fillna(False).astype(bool)

    desired = pd.Series(0.0, index=frame.index, name="desired_pos")
    holding = False

    for date in frame.index:
        if not holding and gate_aligned.loc[date] and frame.loc[date, "p_up"] >= threshold:
            holding = True
        elif holding and (
            not gate_aligned.loc[date] or frame.loc[date, "p_up"] < exit_threshold
        ):
            holding = False
        desired.loc[date] = 1.0 if holding else 0.0

    # Signal at t is held over the next return, from t to t+1.
    position = desired.shift(1).fillna(0.0).rename("pos")
    benchmark_ret = frame["close"].pct_change().fillna(0.0).rename("benchmark_ret")
    gross_ret = (position * benchmark_ret).rename("gross_ret")

    delta_pos = position.diff().fillna(position).rename("delta_pos")
    one_way_cost = (slippage_bps + commission_bps) / 10_000.0
    costs = (delta_pos.abs() * one_way_cost).rename("cost")
    strat_ret = (gross_ret - costs).rename("strat_ret")

    return pd.concat(
        [
            frame["close"],
            frame["p_up"],
            desired,
            position,
            delta_pos,
            benchmark_ret,
            gross_ret,
            costs,
            strat_ret,
        ],
        axis=1,
    )
