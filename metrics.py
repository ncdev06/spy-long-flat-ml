from __future__ import annotations

import numpy as np
import pandas as pd


def performance(df: pd.DataFrame, col: str = "strat_ret", freq: int = 252) -> dict[str, float | int]:
    """Compute return, risk, drawdown, exposure, and turnover diagnostics."""
    if col not in df.columns:
        raise ValueError(f"Column {col!r} is not present in the backtest output.")

    r = df[col].dropna().astype(float)
    if r.empty:
        raise ValueError("No returns are available for performance evaluation.")

    years = len(r) / freq
    ann_return = (1.0 + r).prod() ** (1.0 / years) - 1.0 if years > 0 else 0.0
    ann_vol = r.std(ddof=1) * np.sqrt(freq)

    arithmetic_ann_return = r.mean() * freq
    sharpe = arithmetic_ann_return / ann_vol if ann_vol > 0 else 0.0

    downside = np.minimum(r.to_numpy(), 0.0)
    downside_dev = np.sqrt(np.mean(np.square(downside))) * np.sqrt(freq)
    sortino = arithmetic_ann_return / downside_dev if downside_dev > 0 else 0.0

    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    active = r[df.reindex(r.index).get("pos", pd.Series(1.0, index=r.index)).astype(float) > 0]
    win_rate_active = float((active > 0).mean()) if not active.empty else 0.0

    if "delta_pos" in df.columns:
        delta = df.loc[r.index, "delta_pos"].fillna(0.0).astype(float)
        turnover_pa = float(delta.abs().sum() / years) if years > 0 else 0.0
        n_entries = int((delta > 0).sum())
    else:
        turnover_pa = 0.0
        n_entries = 0

    exposure = (
        float(df.loc[r.index, "pos"].fillna(0.0).astype(float).mean())
        if "pos" in df.columns
        else 1.0
    )

    return {
        "n_days": int(len(r)),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "exposure": exposure,
        "win_rate_active": win_rate_active,
        "n_entries": n_entries,
        "turnover_pa": turnover_pa,
    }
