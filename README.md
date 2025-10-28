# SPY Long/Flat ML Strategy
Python (pandas, scikit-learn, Jupyter Notebook)

## Overview
A modular time-series machine learning project designed to forecast SPY (S&P 500 ETF) daily trends and evaluate ML-driven long/flat trading signals under realistic transaction-cost and slippage assumptions. Implements a fully leak-free walk-forward pipeline with feature engineering, model calibration, and backtesting analysis across market regimes.

## Key Features
- **Data Pipeline:**
  - Pulls and cleans SPY OHLCV data from Yahoo Finance (`data.ipynb`).
  - Computes log returns, rolling volatility, and normalized price series.
- **Feature Engineering:**
  - Constructs technical indicators — SMA, EMA, RSI-14, ATR, log returns, momentum differentials (`features.ipynb`).
  - Aligns lagged indicators to avoid look-ahead bias; verifies through index shifting checks.
- **Modeling & Training:**
  - Trains a calibrated Logistic Regression classifier using TimeSeriesSplit for proper temporal validation (`model.ipynb`).
  - Calibrates probabilistic thresholds to optimize precision-recall balance.
- **Backtesting Framework:**
  - Implements a transaction-cost-aware backtester with slippage + fees and configurable signal thresholds (`backtesting5.ipynb`).
  - Produces performance diagnostics — equity curve, drawdown, trade count, and Sharpe ratio.
- **Evaluation Metrics:**
  - Computes AUC, precision, recall, confusion matrices, Sharpe, and cumulative returns (`metrics2.ipynb`).
- **Execution Runner:**
  - `run.ipynb` executes the entire pipeline end-to-end, generating reproducible results and plots.

## Repository Structure
```
SPY-ML-Strategy/
│
├── data.ipynb           # Data loading and preprocessing
├── features.ipynb       # Feature engineering
├── model.ipynb          # Training and calibration
├── backtesting5.ipynb   # Transaction-cost-aware backtester
├── metrics2.ipynb       # Evaluation metrics and diagnostics
├── run.ipynb            # Master notebook for full pipeline
└── README.md            # Project overview (this file)
```

## Example Outputs
- Equity curve vs. SPY benchmark
- Drawdown plot highlighting low-volatility capital preservation periods
- Model confidence distribution showing probabilistic signal calibration
- Achieved average Sharpe ≈ 0.45 – 0.52, maintaining stable returns across volatile market segments.

## Tech Stack
| Category | Tools / Libraries |
|-----------|------------------|
| Data | pandas, numpy, yfinance |
| Modeling | scikit-learn (LogisticRegression, calibration), joblib |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter Notebook |
| Validation | TimeSeriesSplit, custom backtesting functions |

## Future Extensions
- Add XGBoost / Random Forest classifiers for nonlinear regime modeling.
- Deploy a Streamlit dashboard for live backtest visualization.
- Integrate real-time API to update signals daily.

## How to Run
1. Clone or download this repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch Jupyter and open `run.ipynb`.
4. Execute cells in order to reproduce preprocessing → feature creation → modeling → backtesting → metrics.

## Requirements (minimal)
```
pandas
numpy
scikit-learn
matplotlib
seaborn
yfinance
```
