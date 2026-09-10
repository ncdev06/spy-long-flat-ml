# SPY Long/Flat ML Research Pipeline

A leakage-aware time-series ML and backtesting project for studying daily SPY direction signals under realistic execution assumptions.

This repository is intentionally framed as an **evaluation and research system**, not as a claim of trading alpha. The goal is to make temporal validation, signal alignment, transaction costs, and model stability explicit enough that weak or unstable results are visible instead of hidden.

## What this project demonstrates

- Expanding-window, month-by-month out-of-sample prediction
- Time-aware probability calibration with `TimeSeriesSplit`
- Feature construction using only information available at each prediction date
- Explicit next-bar execution so a signal cannot earn the return used to create it
- Long/flat backtesting with hysteresis, optional trend gating, slippage, and commissions
- Strategy diagnostics including annualized return/volatility, Sharpe, Sortino, drawdown, exposure, turnover, and trade entries
- Same-window comparison against a SPY buy-and-hold benchmark

## Pipeline

```text
Yahoo Finance SPY OHLCV
        ↓
clean data loader
        ↓
stationary-ish technical features
        ↓
expanding monthly walk-forward model
        ↓
time-series probability calibration
        ↓
next-bar long/flat execution
        ↓
transaction costs + benchmark
        ↓
risk / stability diagnostics
```

## Leakage controls

Financial time series are easy to evaluate incorrectly, so the implementation makes several timing decisions explicit:

1. **The target is next-day direction.** Features at date `t` are built only from data known through date `t`.
2. **The final unlabeled row is removed.** It is not silently converted into a negative class.
3. **Each test month is fully out of sample.** Training uses only dates from earlier months.
4. **Calibration is also chronological.** `CalibratedClassifierCV` uses `TimeSeriesSplit` rather than shuffled/stratified folds.
5. **Signals execute one bar later.** A prediction observed on date `t` is applied to the following close-to-close return.
6. **Turnover uses absolute position changes.** Entries and exits cannot cancel each other out in the cost/turnover calculation.

## Features

The current feature set avoids feeding raw price levels directly into the model and instead uses normalized or return-based signals:

- 1-day simple and log returns
- 5-day and 20-day momentum
- Close relative to 5/20-day SMA
- Close relative to 12/26-day EMA
- RSI-14
- ATR as a percentage of price
- 20-day annualized realized volatility
- 20-day volume z-score

## Modeling

The default model is calibrated logistic regression because it provides a simple, interpretable baseline for a noisy prediction problem. A random-forest option is included for comparison.

For every prediction month:

- all prior observations form the training set;
- the current month is held out;
- scaling is fit only inside the training pipeline;
- probability calibration uses chronological folds;
- predictions are written only for the held-out month.

This design makes the experiment more useful as a test of **evaluation discipline and robustness** than as a leaderboard-style model exercise.

## Backtesting

`backtest.py` implements a long/flat strategy with:

- configurable entry probability threshold;
- a slightly lower exit threshold to reduce signal churn;
- optional 120-day trend gate;
- one-bar execution delay;
- configurable slippage and commission assumptions;
- SPY buy-and-hold returns over the exact same evaluation window.

The project does **not** assume that a classifier with >50% directional accuracy is automatically profitable. Costs, turnover, exposure, calibration, and market regime all matter.

## Repository structure

```text
spy-long-flat-ml/
├── data.py              # robust SPY data loading
├── features.py          # aligned feature engineering + target construction
├── model.py             # expanding walk-forward prediction + time-aware calibration
├── backtest.py          # next-bar, transaction-cost-aware backtester
├── metrics.py           # return / risk / drawdown / turnover diagnostics
├── run_pipeline.py      # reproducible command-line runner
├── requirements.txt
├── run.ipynb            # original exploratory end-to-end notebook
├── data.ipynb           # exploratory notebooks retained for project history
├── features.ipynb
├── model.ipynb
├── backtesting5.ipynb
├── metrics2.ipynb
└── README.md
```

## Run it

```bash
git clone https://github.com/ncdev06/spy-long-flat-ml.git
cd spy-long-flat-ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

Useful variations:

```bash
# Random forest instead of logistic regression
python run_pipeline.py --model rf

# Different probability threshold
python run_pipeline.py --threshold 0.57

# Disable the trend regime gate
python run_pipeline.py --no-trend-gate

# Add explicit commission assumptions
python run_pipeline.py --commission-bps 0.5 --slippage-bps 1.5
```

The runner prints strategy diagnostics and a same-window SPY benchmark so results can be judged in context.

## Why I kept the project

The interesting part of this project is not whether one backtest happened to look good. It is the engineering around a difficult evaluation problem: temporal leakage, walk-forward retraining, calibration, execution timing, trading frictions, and reproducibility.

That makes it useful as a compact example of time-series ML infrastructure and experimental design.

## Notes

- Historical market data and model outputs can change as upstream data sources are revised.
- Backtests are sensitive to thresholds, costs, sample windows, and regime definitions.
- This repository is for software/ML research and education only, not investment advice.
