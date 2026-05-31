# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**VANTA** — a three-layer NQ/ES bias engine. Each layer is independent and has its own
data, math, and surface in the dashboard.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — MACRO BIAS (weekly)                                          │
│  scripts/05_weekly_report.py → bias_output.json                         │
│  Fed H.4.1 + macro indicators + CFTC COT → XGBoost direction + regime  │
│  Updated: Friday 5 PM EDT via .github/workflows/weekly-macro-update.yml │
│  Surface: dashboard #bias tab, /.netlify/functions/bias                 │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — RTH BIAS (live, every ~5 min)                                │
│  netlify/functions/intraday.js (canonical, server-side)                 │
│  scripts/09_intraday_bias.py (Python backtest baseline, NOT production) │
│  RTH macro 4-factor bias + open archetype scoring (4 types)            │
│  Options flow + entropy gate + PCA as secondary context                 │
│  Surface: dashboard #intraday tab ("RTH Bias"), /.netlify/functions/intraday │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — LEVELS SCORING (live, every ~5 min)                          │
│  netlify/functions/levels.js (handler) + lib/options.js (math)          │
│  Per-strike GEX/VEX/CharmEX/DAG/OI scoring, vol×gamma regime weights    │
│  hold_prob: data-driven wall reliability (489K OPRA touch events)       │
│  Surface: dashboard #levels tab, /.netlify/functions/levels             │
└─────────────────────────────────────────────────────────────────────────┘
```

**Honest edge claims by layer:**

| Layer | Backtest result | Statistical strength | Trading-grade? |
|---|---|---|---|
| 1 — Macro weekly | 10/13 = 76.9% (Feb-May 2026) | Significant vs 50% random (p=0.026); NOT significant vs 56.2% bull base rate (p=0.067) | Borderline; n=13 too small |
| 2 — Intraday | Untested on outcomes | Theory-grounded (gamma regime: Dim 2025; entropy gate: regime-switching lit) | No — needs ≥30 days of labeled snapshots |
| 3 — Levels score | Composite score AUC 0.53 (near-random) | Arbitrary regime-weight table, empirically unvalidated | No |
| 3 — hold_prob | LR CV AUC 0.617 ± 0.001; XGBoost CV AUC 0.782 ± 0.001 | 489K OPRA touch events, Feb-Dec 2025 0DTE QQQ | Directional signal; single-regime (2025 bull) |

---

## Quick Navigation

| If you want to … | Read this file |
|---|---|
| Run the weekly macro report | `scripts/05_weekly_report.py` |
| Score blind historical predictions | `scripts/06_check_accuracy.py` → `data/processed/weekly_accuracy_log.csv` |
| Understand intraday classification logic | `netlify/functions/intraday.js` (canonical) |
| Understand per-level scoring math | `netlify/functions/lib/options.js` |
| See private methodology source PDFs | `clauderesources/` (gitignored) |
| Trigger a snapshot manually | Actions tab → "GEX 15-min snapshot" → Run workflow |
| Run the local logger daemon | `python schedule_freeflow_logger.py` |
| Look at intraday backtest | `scripts/09_intraday_bias.py` + `scripts/10_validate_intraday.py` |
| Add a new macro feature | `scripts/02_macro_features.py` (data) + `04_train_model.py` (model) + `05_weekly_report.py` (live) |
| Rebuild 0DTE OPRA profiles + snapshots | `scripts/batch_decode_opra.py` |
| Retrain hold_prob model | `scripts/label_intraday_touches.py` → `scripts/fit_intraday_wall_model.py` |
| Validate wall scoring against outcomes | `scripts/calibration_summary.py` |
| Run reversal trade parameter search | `scripts/reversal_backtest.py` |

---

## Environment Setup

```bash
pip install -r requirements.txt         # macro pipeline (XGBoost, Optuna, FRED, yfinance)
pip install -r requirements_levels.txt  # levels logger only (requests, pandas, numpy, pytz)
pip install databento scipy             # OPRA decoder (decode_opra_day.py)
```

`.env` in project root:
```
FRED_API_KEY=your_key_here
FF_SESSION=your_freeflow_session_cookie
```

FRED API key: free at fred.stlouisfed.org.
FF_SESSION: session cookie from free-flow.site. Expires periodically — if Levels tab shows 401/403,
update both `.env` and the GitHub repo secret `FF_SESSION`.

---

## Running the System

**Weekly bias report (Thursdays after 4:30 PM ET):**
```bash
python scripts/05_weekly_report.py --live    # writes bias_output.json
```
Auto-runs every Friday 10 PM UTC via GitHub Actions.

**Historical blind predictions + accuracy log:**
```bash
python scripts/05_weekly_report.py           # writes data/processed/blind_predictions_*.csv
python scripts/06_check_accuracy.py          # builds/updates weekly_accuracy_log.csv
```

**Local intraday backtest:**
```bash
python scripts/09_intraday_bias.py --date 2026-05-22
python scripts/10_validate_intraday.py       # walk-forward when logs/ has data
```

**Manual one-shot snapshot:**
```bash
python freeflow_levels.py --multi            # aggregate 3 nearest expiries
python schedule_freeflow_logger.py --once    # one tick (ET-aware no-op outside window)
```

**Rebuild OPRA 0DTE pipeline from scratch:**
```bash
python scripts/batch_decode_opra.py --workers 4          # generates gex_snapshots_0dte/ + gex_profiles_0dte/
python scripts/label_intraday_touches.py                  # builds intraday_touches.csv (~489K events)
python scripts/fit_intraday_wall_model.py                 # trains LR + XGBoost, saves models/wall_score_intraday.json
# Then manually update _LR coefficients in netlify/functions/lib/options.js from the JSON
```

---

## Layer 1 — Macro Bias Pipeline (scripts/)

Run scripts 01-08 in order to rebuild from scratch:

| Script | Purpose |
|--------|---------|
| `01_data_prep.py` | Back-adjusts raw NQ/ES contract CSVs → continuous series in `data/processed/` |
| `02_macro_features.py` | Pulls FRED + yfinance → `data/processed/macro_features.csv` |
| `03_feature_engineering.py` | Builds weekly labels + price features → `model_dataset.csv` |
| `07_cot_features.py` | Downloads CFTC COT history → `cot_NQ.csv`, `cot_ES.csv` |
| `08_integrate_cot.py` | Merges COT → `model_dataset_cot.csv` + `model_dataset_enriched.csv` |
| `04_train_model.py` | Trains XGBoost + vol model with Optuna → `models/*.pkl` |
| `05_weekly_report.py` | Live report (consumes saved models; does not retrain) |
| `06_check_accuracy.py` | Open-to-close labeled accuracy log + binomial tests |

**Raw CSV format**: semicolon-separated, European number formatting (`.` thousands sep, `,` decimal).
Handled by `01_data_prep.py` via `sep=";", thousands=".", decimal=","`. First row is junk and skipped.

**Macro regime scoring** (rules-based, not ML): each of 11 indicators (net Fed liquidity, VIX,
DXY, 10Y yield, yield curve slope, VIX/VIX3M ratio — both WoW and 4w deltas) is scored +1/0/-1
by its percentile rank vs a rolling 156-week window from `model_dataset_enriched.csv`. Total
score maps to:

```
RISK-ON       ≥ +3    →  STRONG BULL
LEAN RISK-ON  +1, +2  →  LEAN BULL
TRANSITION    0       →  MIXED
LEAN RISK-OFF -1, -2  →  LEAN BEAR
RISK-OFF      ≤ -3    →  STRONG BEAR
```

**XGBoost layer**: `price_model` predicts weekly NQ bull/bear probability; `vol_model` forecasts
weekly realized vol. Both use macro + price features; COT features are injected at prediction time.

**COT staleness check** (`05_weekly_report.py` head): if `cot_NQ.csv`'s most recent row is >9 days
old, the script auto-runs `07_cot_features.py` before proceeding.

**`--live` mode** downloads fresh data from yfinance — does NOT read the processed CSVs. The
processed CSVs only feed the training pipeline (scripts 03 and 04).

---

## Layer 2 — RTH Bias (netlify/functions/intraday.js)

**Canonical implementation is the JS function.** Dashboard tab is labelled "RTH Bias".
The Python `09_intraday_bias.py` is a simplified backtest baseline; it does NOT implement
the RTH archetype scoring.

**Inputs per request:**
1. FreeFlow API: per-strike GEX/VEX/CharmEX/DEX/DAG/OI for 3 nearest expiries
2. FreeFlow vol endpoint: current_iv, rv_iv_ratio, hv21
3. Yahoo Finance: 2 years of daily NQ=F (fallback QQQ) — feeds entropy and PCA
4. Yahoo Finance: SHY (1-3Y Treasury ETF, 1 month) — 2Y yield direction proxy
5. Yahoo Finance: USDJPY=X (1 month) — BOJ/carry unwind signal
6. `bias_output.json` (bundled via `require()` at build time): macro confluence + COT + liquidity

**Computed inside the handler, in order (primary path):**
1–12. GEX/entropy/PCA pipeline
13. `classify2YSignal(shyBars)` → RISING_FAST / RISING / STABLE / FALLING / UNAVAILABLE
14. `classifyBOJSignal(bars)` → CARRY_UNWIND / YEN_STABLE / YEN_WEAKENING / UNAVAILABLE
15. `getLiquidityTrend(macroBiasData)` → IMPROVING / STABLE / DETERIORATING
16. `getCotLabel(macroBiasData)` → FUMES_LONG / NEUTRAL / EXTREME_SHORT
17. `classifyRTHBias({…})` → BULLISH / BEARISH / NEUTRAL / UNKNOWN
18. `classifyOpenArchetype({…})` → TYPE_A / TYPE_B / TYPE_C / TYPE_D + confidence 0-5
19. `getConfig()` → loads archetype names/descriptions from gitignored methodology_config or env var

**Open Archetype scoring** (`classifyOpenArchetype`):
- Scores all 4 types based on GEX structure signals (ivBand, flipDiff, dex_sign, vex_sign, charm_sign, wall proximity)
- Each condition adds 1 point; max score 5; winner displayed in the big action panel
- TYPE_A and TYPE_C are bull-resolving (`dir: 'bull'`); TYPE_B and TYPE_D are bear-resolving
- Actual archetype names/descriptions are in the gitignored `methodology_config.js`

**RTH Bias verdict** (`classifyRTHBias`):
- Weighted signals: 2Y yield (1-2 pts), liquidity (1), COT (1), BOJ (2 for CARRY_UNWIND), weekly macro (1)
- BEARISH when bear score ≥ 2.5; BULLISH when bull score ≥ 2.5; else NEUTRAL

**Hard gate**: CRITICAL entropy → `NO_BIAS, AVOID` on the options classifier only; RTH Bias still shows.

---

## Layer 3 — Levels Scoring (netlify/functions/lib/options.js)

Per-strike composite score (unchanged, kept for ranking):
```
score = (gex_norm * w_gex + vex_norm * w_vex + charmex_norm * w_charmex
       + oi_norm * w_oi + dag_norm * w_dag) × 100
```
Each component min-max normalized across nearby strikes. Weights come from a 3×3 table
indexed by (volRegime, gammaRegime). **This score has AUC 0.53 — near-random. Use `hold_prob` instead.**

**`hold_prob`** — data-driven wall reliability, attached to every scored level:
- Logistic regression on 489K intraday 0DTE touch events (Feb-Dec 2025 OPRA CBBO-1m)
- CV AUC 0.617 ± 0.001; XGBoost CV AUC 0.782 ± 0.001
- Top features by importance: `time_of_day`, `is_high_vol`, `confluence`, `is_put`, `charmex_norm`
- Coefficients in `models/wall_score_intraday.json` — hardcoded in `_LR` object in `lib/options.js`
- To retrain: run OPRA pipeline (see Running the System), then update `_LR` manually from the JSON
- **`vex_over_gex` and `charmex_over_gex` default to training means** in production because
  FreeFlow data has a different unit scale than the OPRA training data — do not reintroduce
  live computation of these two features

**`confluence`** — 1 when GEX_norm + VEX_norm + CharmEX_norm all ≥ 40 at the same strike.
These are the highest-quality walls; `hold_prob` is meaningfully higher on confluence=1 walls.

**Per-level fields from `scoreLevels`:**
- `score` — composite magnitude rank (0-100)
- `hold_prob` — data-driven reliability (0-1 sigmoid)
- `type` — `"CALL WALL"` or `"PUT WALL"` + ` + VOL SENSITIVE` if `|VEX| / |GEX| > 2.0`
- `wall_reaction` — tag from `classifyWallReaction(level)` (private reaction table)
- `confluence` — boolean int (from FreeFlow data, not always populated)

**`scoreLevels(strikes, weights, futuresPrice, volRegime, gammaFlip)`** — signature takes
`volRegime` and `gammaFlip` to pass through to `computeHoldProb`.

**Constants:**
- `FILTER_PCT = 5.0` — strikes within ±5% of futures price
- `MIN_SCORE = 20.0` — discard scored strikes below this
- `PROXIMITY_EFOLD = 200.0` — in `intraday.js` for top-wall ranking (true halflife ≈ 139pts)
- `REGIME_WEIGHTS` — 3×3 (vol × gamma) → {gex, vex, charmex, oi, dag}; **empirically unvalidated**

---

## OPRA Calibration Pipeline (scripts/)

Processes historical 0DTE QQQ options data (Databento CBBO-1m format) to train `hold_prob`.
Source files live in `dataidk/` (gitignored — too large to commit).

| Script | Purpose |
|--------|---------|
| `decode_opra_day.py` | Decode one OPRA day → 0DTE GEX profile + 13 intraday snapshots |
| `batch_decode_opra.py` | Parallel batch across all days → `gex_snapshots_0dte/` + `gex_profiles_0dte/` |
| `label_wall_outcomes.py` | Daily OHLC-based wall hold/break labeling (coarse, used for calibration_summary) |
| `label_intraday_touches.py` | Minute-by-minute touch detection using nearest prior snapshot → `intraday_touches.csv` |
| `fit_intraday_wall_model.py` | Train LR + XGBoost on touch events → `models/wall_score_intraday.json` |
| `fit_wall_score_model.py` | Validates composite score vs outcomes (shows AUC 0.53 = near-random) |
| `calibration_summary.py` | Statistical report on wall hold rates by regime/type/threshold |
| `reversal_backtest.py` | Vectorized parameter search: stop/target/filter combos → `reversal_backtest.csv` |

**Key design facts:**
- OPRA CBBO-1m provides bid/ask quotes, not true OI. Quote size (bid_sz + ask_sz) is used as
  an OI proxy — underestimates true dealer exposure but preserves relative strike ranking.
- `decode_opra_day.py` filters to expiry == trade_date (0DTE only) and takes 13 snapshots per
  day at 30-min intervals (9:31, 10:00, 10:30 … 15:30 ET).
- Intraday spot price is adjusted per-snapshot: `spot_t = spot_open × (nq_t / nq_open)` using
  NQ 1-minute bars from `data/processed/NQ_1m_clean.csv`.
- The `vex_over_gex` ratio has mean ≈ 0.0003 in OPRA data (tiny vega relative to gamma at 0DTE).
  FreeFlow live data has vex/gex ≈ 0.2+. Do not use live ratio features in `computeHoldProb`.

**Backtest findings (Feb-Dec 2025 0DTE QQQ, using NQ 1m for forward P&L):**
- Best setup at 100 NQ pt target: 71% win rate, 4:1 RR, N=17 — requires confluence + midday
  (12-14 ET) + GEX≥60 + wall ≥1.5% from spot + slow approach (vel ≤ 5)
- Strongest empirical signals: time_of_day (afternoon > morning by 17pp), is_high_vol (EXPANSION
  vol degrades wall reliability), confluence walls outperform non-confluence
- FROM_ABOVE (longs at PUT walls) outperforms FROM_BELOW (shorts at CALL walls) in bull year
- Stop definition matters: 0DTE walls require 2-bar confirmed break, not single-bar, due to
  brief overshoots before reversal

---

## Methodology Sources (clauderesources/, gitignored)

Private source PDFs informing Layer 2 (RTH) and Layer 3 (Levels) rules. Owner-only.

Published empirical references that inform generic constants in `lib/options.js`:
- Elms 2026 — SPX amplification vs pinning regime
- Dim/Eraker/Vilkov 2025 — MM 0DTE gamma asymmetry (source of `GAMMA_ASYMMETRY_RATIO = 0.344`)
- Garmash 2024 — gamma regime ↔ mean-reversion/momentum mapping

**Helpers in `lib/options.js`:**
- `computeHoldProb(level, volRegime, timeOfDayET, gammaFlip, futuresPrice)` — logistic regression
  wall reliability (0-1); coefficients from `models/wall_score_intraday.json`
- `currentHourET()` — returns current hour + minute/60 in ET; feeds `time_of_day` feature live
- `classifyWallReaction(level)` — private reaction table → `wall_reaction` tag
- `computeAggregateGreeks(strikeArr)` — sign triple + totals across the passed set
- `applyBiasTable(aggregates, volRegime, priceVsFlip)` — private bias table → primary bias tag
- `WALL_REACTION_DIR` / `BIAS_TAG_DIR` — tag → `{dir: BULL/BEAR/NEUTRAL, strength: 0/1/2}` maps
- `GAMMA_ASYMMETRY_RATIO = 0.344` — from Dim et al. Table 3
- `PINNING_REGIME_ACTIVE = false` — from Elms 2026; exported but not yet consumed

**Helpers in `netlify/functions/intraday.js` (RTH layer):**
- `classify2YSignal(bars)`, `classifyBOJSignal(bars)` — macro proxies
- `getLiquidityTrend(data)` / `getCotLabel(data)` — derive from `bias_output.json`
- `classifyRTHBias(inputs)`, `classifyOpenArchetype(inputs)` — verdicts
- `getMacroBiasData()` — live-fetches `bias_output.json` from GitHub (10-min cache)
- `getConfig()` — lazy-loads private label config (env var → gitignored file → empty fallback)

**Privacy model:**
- `netlify/functions/lib/methodology_config.js` — **gitignored**; real content is owner-only
- `netlify/functions/lib/methodology_config.example.js` — **committed** skeleton (empty strings)
- Production: `METHODOLOGY_CONFIG` env var on Netlify holds base64-encoded config
- Committed code uses abstract codes (`TYPE_A/B/C/D`, `FUMES_LONG`, etc.)
- `clauderesources/` — **gitignored**; all source PDFs live here

---

## Frontend (index.html + auth.js)

Single-file static dashboard served by Netlify. Three tabs (`#bias`, `#intraday`="RTH Bias", `#levels`)
routed via URL hash. All DOM construction uses an `el(tag, attrs, ...children)` helper — no
framework, no template strings.

**Levels tab display:**
- Walls sorted by `hold_prob` descending by default (not `score`)
- `Hold %` column color-coded: green ≥55%, amber 40-54%, red <40%
- HIGH VOL banner appears when `vol_regime === "EXPANSION"` (walls hold only 28% historically)
- Alert panel (Watching/Target) shows `XX% hold` instead of raw score

**Auth**: `login.html` + `auth.js` gate the dashboard. Token is base64-encoded `{user, ts}` in
`localStorage` as `vanta_session`. Not server-signed — adequate for small trusted user base only.

**`netlify.toml`** publishes from `.` (root), functions from `netlify/functions/`, proxies
`/api/*` → `/.netlify/functions/:splat`.

---

## Data & Persistence

```
data/raw/                    ← NQ/ES OHLCV (daily, 4h, 1h, 15m, 5m, 1m per contract)
data/processed/
  NQ_1m_clean.csv            ← 12 years of NQ 1-min bars; used by OPRA pipeline for intraday spot
  model_dataset_enriched.csv ← rolling-window percentile context for live regime scoring
  gex_profiles_0dte/         ← opening 0DTE GEX profile per day (from batch_decode_opra.py)
  gex_snapshots_0dte/        ← 13 intraday snapshots per day; gex_snapshot_YYYYMMDD_HHMM.csv
  intraday_touches.csv       ← 489K labeled touch events from gex_snapshots_0dte + NQ_1m
  reversal_backtest.csv      ← output of reversal_backtest.py parameter search
  wall_outcomes.csv          ← daily OHLC-based wall outcomes (coarser than intraday_touches)
  weekly_accuracy_log.csv    ← cumulative O→C labeled track record
models/
  price_model.pkl, vol_model.pkl   ← Layer 1 XGBoost models
  wall_score_intraday.json         ← hold_prob LR coefficients + scaler params (source of truth)
  wall_score_intraday.pkl          ← trained LR + XGBoost objects (Python only)
dataidk/                     ← raw OPRA CBBO-1m .dbn.zst files (gitignored, large)
logs/                        ← levels_YYYY-MM-DD.csv snapshots (gex-snapshots branch)
bias_output.json             ← Layer 1 live output (bundled at build time)
```

**`bias_output.json`** auto-updated by `.github/workflows/weekly-macro-update.yml` every Friday
at 10 PM UTC. Requires `FRED_API_KEY` repo secret.

**`gex-snapshots` branch** holds accumulated level/snapshot CSVs. Populated by
`.github/workflows/gex-snapshot.yml` every 15 min during 03:00-17:00 ET, Mon-Fri. Requires
`FF_SESSION` repo secret.

---

## Edge & Calibration Status

**Empirically validated:**
- `hold_prob` model: LR CV AUC 0.617, XGBoost CV AUC 0.782 on 489K 0DTE touch events
- Strongest signals: time_of_day (afternoon walls hold 56% vs open 39%), is_high_vol, confluence
- PUT walls outperform CALL walls in bull-year data (50% vs 43% intraday hold rate)
- Gamma regime → vol regime mapping (Dim/Eraker/Vilkov 2025)
- Modern SPX in amplification not pinning regime (Elms 2026)
- Macro weekly hit rate 76.9% (13 weeks; significant vs random, NOT vs bull base rate)

**Theory-grounded but uncalibrated:**
- Composite `score` (REGIME_WEIGHTS 3×3 grid) — AUC 0.53, near-random on 0DTE data
- Primary-bias rules (private methodology, untested on NQ outcomes)
- Per-wall reaction tags (same)
- Shannon entropy threshold at 75th percentile of rolling 252-day window
- PCA pc1_momentum_valid threshold at 0.3 combined |loadings|
- `STRONG_WALL = 60`, `EXCEPTIONAL_WALL = 75`, `H_GEX_CONFIDENCE_CUT = 0.6`
- `PROXIMITY_EFOLD = 200pts`
- `GAMMA_ASYMMETRY_RATIO = 0.344` (SPX-derived; NQ-specific value unknown)

**Known data limitations:**
- OPRA CBBO uses quote size as OI proxy — underestimates true dealer exposure vs FreeFlow live data
- hold_prob trained on 2025 bull year only; will need retraining for bear/volatile regimes
- `vex_over_gex` ratio has different scale in OPRA vs FreeFlow — excluded from live `computeHoldProb`

---

## Key Constraints & Gotchas

- **Three independent layers** — don't conflate. Layer 1 is weekly XGBoost; Layer 2 is live
  options-flow + entropy/PCA; Layer 3 is per-strike scoring. Each has its own update cadence.
- **Python intraday script is NOT canonical** — `intraday.js` is the production path. Python is
  for hindcasting only.
- **`hold_prob` coefficients must not be hand-tuned** — update only by retraining via the OPRA
  pipeline. Source of truth is `models/wall_score_intraday.json`; hardcode into `_LR` in `options.js`.
- **COT data lag** — CFTC releases Tuesday data on Friday. Staleness check auto-refreshes if >9 days.
- **All time arithmetic must be ET** — `freeflow_logger.py` and CI runners use ET explicitly.
- **`--live` mode bypasses processed CSVs** — `05_weekly_report.py --live` downloads fresh from yfinance.
- **Model artifacts must exist** — `05_weekly_report.py` loads `models/*.pkl` on import. Run
  `04_train_model.py` first if missing.
- **FF_SESSION expires periodically** — update `.env`, GitHub repo secret, AND Netlify env var.
- **`scoreLevels` signature** — takes `(strikes, weights, futuresPrice, volRegime, gammaFlip)`;
  both callers (`levels.js` and `intraday.js`) must pass all five args.

---

## Autonomy & Operations (for Claude Code itself)

- **Don't tune uncalibrated thresholds without data** — constants in "theory-grounded but
  uncalibrated" above stay frozen until labeled outcomes provide stratified hit rates.
- **Don't change `_LR` coefficients by hand** — retrain via the OPRA pipeline.
- **Don't conflate layers** — a fix to bias-table logic doesn't touch the XGBoost pipeline.
- **Update CLAUDE.md when adding helpers** — this file is the single source of truth for
  navigating the codebase. New functions in `lib/options.js` belong in the helpers section.
- **Three layers, three commits** — when changes span layers, prefer separate commits per layer.
- **The Python intraday script is intentionally simpler than the JS** — port new logic to it
  only when specifically needed for backtesting.
