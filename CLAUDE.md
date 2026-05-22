# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**VANTA** — a systematic weekly bias engine for NQ/ES equity index futures. Combines Fed balance sheet data, macro indicators, and CFTC COT positioning to generate a weekly directional outlook every Thursday after 4:30 PM ET (when the H.4.1 is published).

Blind out-of-sample accuracy (Feb–May 2026, 16 weeks): 76.9% on 13 tradeable calls.

## Environment Setup

```bash
pip install -r requirements.txt       # main pipeline (XGBoost, Optuna, FRED, yfinance)
pip install -r requirements_levels.txt  # freeflow_levels.py only (requests, pandas, numpy)
```

Create `.env` in the project root:
```
FRED_API_KEY=your_key_here
FF_SESSION=your_freeflow_session_cookie
```

FRED API key: free at fred.stlouisfed.org  
FF_SESSION: session cookie from free-flow.site (expires periodically; update COOKIES dict in `freeflow_levels.py` if 401/403 errors appear)

## Running the System

**Weekly bias report (run every Thursday after 4:30 PM ET):**
```bash
python scripts/05_weekly_report.py --live
```

**Historical blind predictions (no `--live` flag):**
```bash
python scripts/05_weekly_report.py
python scripts/06_check_accuracy.py   # compare predictions vs actuals afterward
```

**Print formatted outlook from saved predictions:**
```bash
python run_weekly_outlook.py               # most recent week
python run_weekly_outlook.py --date 2026-05-14  # specific week
python run_weekly_outlook.py --pred path/to/predictions.csv
```

**FreeFlow options level calculator:**
```bash
python freeflow_levels.py                         # single expiry snapshot
python freeflow_levels.py --multi                 # aggregate 3 nearest expiries
python freeflow_levels.py --live --interval 60    # continuous refresh
python freeflow_levels.py --exp 2026-05-23        # specific expiry
```

**Historical level logger (runs via Windows Task Scheduler at 08:30/09:35/13:00/15:30 ET):**
```bash
python freeflow_logger.py --force    # log immediately
python freeflow_logger.py --schedule # wait for next snapshot time
```

## Pipeline Scripts (run in order to rebuild from scratch)

| Script | Purpose |
|--------|---------|
| `scripts/01_data_prep.py` | Back-adjusts raw NQ/ES contract CSVs → continuous series in `data/processed/` |
| `scripts/02_macro_features.py` | Pulls FRED + yfinance data → `data/processed/macro_features.csv` |
| `scripts/03_feature_engineering.py` | Builds weekly labels + price features → `data/processed/model_dataset.csv` |
| `scripts/07_cot_features.py` | Downloads CFTC COT history → `data/processed/cot_NQ.csv`, `cot_ES.csv` |
| `scripts/08_integrate_cot.py` | Merges COT → `data/processed/model_dataset_cot.csv` + `model_dataset_enriched.csv` |
| `scripts/04_train_model.py` | Trains XGBoost + vol model with Optuna → `models/*.pkl` |
| `scripts/05_weekly_report.py` | Live report (consumes saved models; does not retrain) |
| `scripts/06_check_accuracy.py` | Scores blind predictions vs actual weekly closes |

Raw price CSVs go in `data/raw/NQ/` and `data/raw/ES/` — scripts 01+ expect them there.

## Architecture

```
data/raw/           ← NQ/ES OHLCV CSVs (daily, 4h, 1h, 15m, 5m, 1m per contract)
data/processed/     ← outputs of scripts 01–08; inputs to weekly report
models/             ← price_model.pkl, vol_model.pkl, price_features.pkl, all_features.pkl
logs/               ← levels_YYYY-MM-DD.csv snapshots written by freeflow_logger.py
```

**Raw CSV format**: semicolon-separated, European number formatting (`.` as thousands separator, `,` as decimal). `scripts/01_data_prep.py` handles this via `sep=";", thousands=".", decimal=","`. First row is a header junk line and is skipped.

**Processed CSV columns** (after `01_data_prep.py`): `date, symbol, open, high, low, close, volume` — date-indexed, one row per bar.

**Macro regime scoring** (rules-based, not ML): each of 11 macro indicators (net Fed liquidity, VIX, DXY, 10Y yield, yield curve slope, VIX/VIX3M ratio) is scored +1/0/-1 by its percentile rank vs a rolling 156-week window. Total score maps to: RISK-ON (≥+3), LEAN RISK-ON (+1/+2), TRANSITION (0), LEAN RISK-OFF (-1/-2), RISK-OFF (≤-3). Historical percentile context is read from `data/processed/model_dataset_enriched.csv`.

**XGBoost models**: `price_model` predicts weekly NQ direction (bull/bear probability); `vol_model` forecasts weekly realized vol. Both use a combined feature set of macro + price features; COT features are injected at prediction time.

**Two-layer system**:
- Layer 1: `scripts/` pipeline — macro regime + XGBoost directional model
- Layer 2: `freeflow_levels.py` — options Greeks (GEX, VEX, CharmEX, DEX) scoring from FreeFlow API to identify NQ reversal levels. Vol regime (CONTRACTION/NEUTRAL/EXPANSION based on IV) adjusts weighting of Greeks.

**`05_weekly_report.py --live` downloads fresh data from yfinance** — it does NOT read `NQ_daily_clean.csv`. The processed CSVs are only used by scripts 03 and 04 (training pipeline).

## Frontend Dashboard (Vanta)

`index.html` — single-file static dashboard deployed on **Netlify** (not GitHub Pages). Both the static HTML and the serverless function are served by Netlify. Two tabs:
- **Bias** (`#bias`): fetches `bias_output.json` as a static file (relative path)
- **Levels** (`#levels`): fetches live data via `/.netlify/functions/levels`

`netlify/functions/levels.js` — serverless function (zero external deps, Node built-ins only). Calls FreeFlow API using `FF_SESSION` env var set in Netlify dashboard, aggregates 3 nearest expiries, scores levels, returns JSON. Runs on each browser request; `Cache-Control: max-age=240` limits upstream calls. Key constants: `FILTER_PCT = 5.0` (only strikes within ±5% of futures price), `MIN_SCORE = 20.0`, QQQ-to-NQ conversion `ratio ≈ 41.14`.

`netlify.toml` — publishes from `.` (root), functions from `netlify/functions/`, proxies `/api/*` → `/.netlify/functions/:splat`.

**`levels_data.json`** — a static snapshot in the repo root used for local development and debugging only. It is NOT auto-updated by any process. The live dashboard always calls the Netlify function.

**`bias_output.json`** — updated manually: run the weekly report with `--live`, then copy the JSON output into this file and commit. Schema: `meta`, `confluence`, `macro_regime` (name, score, factor_scores), `price_model` (direction, probability), `vol_forecast`, `cot`, `vix_term_structure`, `key_levels`.

**Dashboard JS architecture** (`index.html` script section):
- All DOM construction uses the `el(tag, attrs, ...children)` helper — no framework, no template strings.
- `kv(key, val, valStyle)` builds a labeled key-value row; `card(titleText, ...children)` builds a card with staggered fade-in animation.
- `render(d)` consumes `bias_output.json`; `renderLevels(d)` consumes the levels API response.
- The `HIST` array near the top of the script is **hardcoded** weekly prediction history (not read from a file) — update it manually when adding new weeks.
- Tab routing uses URL hash (`#bias`, `#levels`); switching tabs calls `switchTab(name)`.
- Color helpers: `cc(v)` → `"bull"/"bear"/"mixed"` CSS class from a regime/confluence string; `dc(v)` → `"bull"/"bear"` from a direction string.

## Key Constraints

- **No intraday signals** — weekly timeframe only; runs once per week on Thursday.
- **COT data lag** — CFTC releases Tuesday data on Friday; `nq_lev_pctile` is sourced from `cot_NQ.csv` (historical), not the live COT feed.
- **FreeFlow session expires** — if the Levels tab shows auth errors, update `FF_SESSION` in both `.env` (local) and the Netlify environment variable.
- **VIX term structure display** — in `--live` mode, `vix_ratio` and `vvix` use the current unshifted values; historical weeks use the model-input (shifted-by-1-week) values to avoid lookahead.
- **Model artifacts must exist** — `05_weekly_report.py` immediately loads `models/*.pkl` on import; run `04_train_model.py` first if models are missing.
