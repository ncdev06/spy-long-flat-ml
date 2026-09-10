import numpy as np
import pandas as pd

from backtest import backtest_long_flat
from features import FEATURE_COLUMNS, make_features
from metrics import performance
from model import rolling_predict


def synthetic_prices(n: int = 700) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-02", periods=n)
    t = np.arange(n, dtype=float)
    daily_move = 0.04 + 0.75 * np.sin(t / 6.0) + 0.25 * np.sin(t / 17.0)
    close = 100.0 + np.cumsum(daily_move)

    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": 1_000_000 + (50_000 * np.sin(t / 9.0)),
        },
        index=dates,
    )


def test_feature_target_alignment_drops_unknown_future():
    prices = synthetic_prices(180)
    dataset = make_features(prices)

    assert set(FEATURE_COLUMNS).issubset(dataset.columns)
    assert dataset.index.max() < prices.index.max()
    assert dataset["label"].isin([0, 1]).all()
    assert np.isfinite(dataset[FEATURE_COLUMNS].to_numpy()).all()


def test_backtest_uses_next_bar_position_and_absolute_turnover():
    dates = pd.bdate_range("2024-01-02", periods=5)
    close = pd.Series([100.0, 101.0, 102.0, 100.0, 103.0], index=dates)
    p_up = pd.Series([0.9, 0.9, 0.1, 0.1, 0.9], index=dates)

    result = backtest_long_flat(close, p_up, threshold=0.55, slippage_bps=0.0)

    # The first signal cannot earn the first visible return; execution is delayed.
    assert result.iloc[0]["pos"] == 0.0
    assert result.iloc[1]["pos"] == 1.0

    stats = performance(result)
    assert stats["turnover_pa"] >= 0.0
    assert stats["n_entries"] >= 1


def test_walk_forward_predictions_are_out_of_sample():
    prices = synthetic_prices()
    dataset = make_features(prices)
    predictions = rolling_predict(
        dataset,
        start_year=2021,
        min_train_days=252,
        calibration_splits=3,
    )

    predicted = predictions.dropna()
    assert not predicted.empty
    assert predicted.between(0.0, 1.0).all()

    result = backtest_long_flat(
        prices["close"],
        predictions,
        threshold=0.55,
        gate=None,
        slippage_bps=1.0,
    )
    stats = performance(result)
    assert stats["n_days"] > 0
    assert 0.0 <= stats["exposure"] <= 1.0
