from __future__ import annotations

import numpy as np
import pandas as pd
import ta


FEATURE_COLUMNS = [
    "ret1",
    "logret1",
    "mom5",
    "mom20",
    "close_sma5",
    "close_sma20",
    "close_ema12",
    "close_ema26",
    "rsi14",
    "atr_pct14",
    "vol20",
    "volume_z20",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stationary-ish features available at the close of each date.

    Each row's label represents the direction of the *next* close-to-close return.
    The final row is removed because its future return is unknown. The backtester
    applies each prediction to the following bar, so features do not need an extra
    one-day shift here.
    """
    out = df.copy().sort_index()

    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    out["ret1"] = close.pct_change()
    out["logret1"] = np.log(close).diff()
    out["mom5"] = close.pct_change(5)
    out["mom20"] = close.pct_change(20)

    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    out["close_sma5"] = close / sma5 - 1.0
    out["close_sma20"] = close / sma20 - 1.0
    out["close_ema12"] = close / ema12 - 1.0
    out["close_ema26"] = close / ema26 - 1.0

    out["rsi14"] = ta.momentum.rsi(close, window=14) / 100.0
    atr14 = ta.volatility.average_true_range(
        out["high"].astype(float),
        out["low"].astype(float),
        close,
        window=14,
    )
    out["atr_pct14"] = atr14 / close
    out["vol20"] = out["logret1"].rolling(20).std() * np.sqrt(252)

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    out["volume_z20"] = (volume - volume_mean) / volume_std.replace(0, np.nan)

    future_return = close.shift(-1) / close - 1.0
    out["label"] = np.where(
        future_return.notna(),
        (future_return > 0).astype(int),
        np.nan,
    )

    result = (
        out[FEATURE_COLUMNS + ["label"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    result["label"] = result["label"].astype(int)

    return result
