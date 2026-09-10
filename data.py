from __future__ import annotations

import pandas as pd
from yahooquery import Ticker


_REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def load_spy(start: str = "2005-01-01", end: str | None = None) -> pd.DataFrame:
    """Load daily SPY OHLCV data and return a clean DatetimeIndex frame.

    The loader intentionally keeps data acquisition separate from feature creation so
    the modeling code can be tested or reused with a pre-downloaded DataFrame.
    """
    history = Ticker("SPY").history(start=start, end=end, interval="1d")
    if history is None or len(history) == 0:
        raise RuntimeError("Yahoo Finance returned no SPY history.")

    df = history.reset_index().rename(columns=str.lower)
    if "symbol" in df.columns:
        df = df[df["symbol"] == "SPY"].copy()

    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise RuntimeError(f"SPY history is missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df[_REQUIRED_COLUMNS]
        .dropna(subset=["date", "open", "high", "low", "close", "volume"])
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .set_index("date")
    )

    if not df.index.is_monotonic_increasing:
        raise RuntimeError("SPY history is not sorted chronologically.")

    return df
