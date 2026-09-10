from __future__ import annotations

import argparse
import json

from backtest import backtest_long_flat, make_trend_gate
from data import load_spy
from features import make_features
from metrics import performance
from model import rolling_predict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the leakage-aware SPY long/flat research pipeline."
    )
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument(
        "--no-trend-gate",
        action="store_true",
        help="Disable the simple moving-average regime gate.",
    )
    args = parser.parse_args()

    prices = load_spy(start=args.start)
    dataset = make_features(prices)
    probabilities = rolling_predict(
        dataset,
        start_year=args.start_year,
        model_name=args.model,
    )

    gate = None if args.no_trend_gate else make_trend_gate(prices["close"])
    result = backtest_long_flat(
        prices["close"],
        probabilities,
        threshold=args.threshold,
        gate=gate,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
    )

    strategy_metrics = performance(result, "strat_ret")
    benchmark_metrics = performance(result, "benchmark_ret")

    print("Strategy")
    print(json.dumps(strategy_metrics, indent=2))
    print("\nSPY buy-and-hold benchmark over the same prediction window")
    print(json.dumps(benchmark_metrics, indent=2))


if __name__ == "__main__":
    main()
