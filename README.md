# Aseras Macro Regime Engine

A systematic weekly bias engine for NQ/ES equity index futures. 
Combines Federal Reserve balance sheet data, macroeconomic 
indicators, and institutional futures positioning to generate 
a structured weekly outlook every Thursday.

## What It Does

Pulls and processes data from multiple free sources, scores 
the macro environment using a rules-based percentile regime 
classifier, runs an XGBoost price direction model, and outputs 
a complete weekly bias report including key levels, volatility 
forecast, and COT positioning context.

## Results

Blind out-of-sample test (Feb–May 2026, 16 weeks):
- 76.9% directional accuracy on 13 tradeable calls
- RISK-ON regime: 100% accurate
- LEAN BULL calls: 5/5 perfect
- 2 misses attributable to April 2026 tariff deal catalyst

## Data Sources

- Federal Reserve H.4.1 (WALCL, RRP, TGA) via FRED API
- US 10-Year Treasury Yield via FRED
- DXY and VIX via yfinance
- CFTC Commitment of Traders (Leveraged Funds positioning) 
  for NQ and ES futures — 12 years of history
- NQ/ES OHLCV price data 2014–2026

## Architecture

Six scripts running in sequence:

- 01_data_prep.py — back-adjusts raw futures contracts 
  into continuous price series
- 02_macro_features.py — pulls and engineers macro features 
  from FRED and yfinance
- 03_feature_engineering.py — builds weekly labels and 
  price-based features
- 04_train_model.py — trains XGBoost models with Optuna 
  hyperparameter tuning
- 07_cot_features.py — downloads and processes CFTC COT data
- 08_integrate_cot.py — merges COT features into model dataset
- 05_weekly_report.py — live weekly bias report (run Thursdays)
- 06_check_accuracy.py — blind prediction accuracy checker

## Weekly Usage

Run every Thursday after 4:30 PM ET once the H.4.1 is published:

    python scripts/05_weekly_report.py --live

## Setup

    pip install -r requirements.txt

Create a .env file in the project root:

    FRED_API_KEY=your_key_here

Get a free FRED API key at fred.stlouisfed.org

## Current Limitations

- Weekly timeframe only — not an intraday signal
- 1–2 week lag on regime change detection
- Cannot catch catalyst-driven reversals (policy announcements, 
  geopolitical events)
- Designed as Layer 1 of a multi-layer bias framework

## Planned Additions

- VIX term structure (VIX/VIX3M ratio)
- Yield curve slope (2Y10Y spread)
- Credit spread proxy (HYG/LQD)
- Market breadth internals
- Regime classification layer

## Tech Stack

Python 3.13, XGBoost, Optuna, pandas, scikit-learn, 
FRED API, yfinance, CFTC public data
