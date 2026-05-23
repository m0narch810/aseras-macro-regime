# CLAUDE.md

Guide for Claude Code when working in this repository.

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
│  LAYER 2 — INTRADAY BIAS (live, every ~5 min)                           │
│  netlify/functions/intraday.js (canonical, server-side)                 │
│  scripts/09_intraday_bias.py (Python backtest baseline, NOT production) │
│  Options flow + entropy gate + PCA + macro confluence + walls.pdf tags  │
│  Surface: dashboard #intraday tab, /.netlify/functions/intraday         │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — LEVELS SCORING (live, every ~5 min)                          │
│  netlify/functions/levels.js (handler) + lib/options.js (math)          │
│  Per-strike GEX/VEX/CharmEX/DAG/OI scoring, vol×gamma regime weights    │
│  walls.pdf reaction tag per level                                       │
│  Surface: dashboard #levels tab, /.netlify/functions/levels             │
└─────────────────────────────────────────────────────────────────────────┘
```

**Honest edge claims by layer** (see "Edge & Calibration Status" below for detail):

| Layer | Backtest result | Statistical strength | Trading-grade? |
|---|---|---|---|
| 1 — Macro weekly | 10/13 = 76.9% (Feb-May 2026) | Significant vs 50% random (p=0.026); NOT significant vs 56.2% bull base rate (p=0.067) | Borderline; n=13 too small |
| 2 — Intraday | Untested on outcomes | Theory-grounded (gamma regime: Dim 2025; entropy gate: regime-switching lit) | No — needs ≥30 days of labeled snapshots |
| 3 — Levels | Untested on outcomes | walls.pdf table is theory; modern SPX shows amplification not pinning (Elms 2026) | No — same data gap |

---

## Quick Navigation

| If you want to … | Read this file |
|---|---|
| Run the weekly macro report | `scripts/05_weekly_report.py` |
| Score blind historical predictions | `scripts/06_check_accuracy.py` → `data/processed/weekly_accuracy_log.csv` |
| Understand intraday classification logic | `netlify/functions/intraday.js` (canonical) |
| Understand per-level scoring math | `netlify/functions/lib/options.js` |
| See the walls.pdf / bias.pdf source rules | `clauderesources/` |
| Trigger a snapshot manually | Actions tab → "GEX 15-min snapshot" → Run workflow |
| Run the local logger daemon | `python schedule_freeflow_logger.py` |
| Look at intraday backtest | `scripts/09_intraday_bias.py` + `scripts/10_validate_intraday.py` |
| Add a new macro feature | `scripts/02_macro_features.py` (data) + `04_train_model.py` (model) + `05_weekly_report.py` (live) |

---

## Environment Setup

```bash
pip install -r requirements.txt         # macro pipeline (XGBoost, Optuna, FRED, yfinance)
pip install -r requirements_levels.txt  # levels logger only (requests, pandas, numpy, pytz)
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
old, the script auto-runs `07_cot_features.py` before proceeding. Prevents the script silently
emitting a 17-week-stale `nq_lev_pctile` that the previous version was doing.

**`--live` mode** downloads fresh data from yfinance — does NOT read the processed CSVs. The
processed CSVs only feed the training pipeline (scripts 03 and 04).

---

## Layer 2 — Intraday Bias (netlify/functions/intraday.js)

**Canonical implementation is the JS function.** The Python `09_intraday_bias.py` is a simplified
backtest baseline used by `10_validate_intraday.py` for walk-forward validation. The dashboard
never calls Python.

**Inputs per request:**
1. FreeFlow API: per-strike GEX/VEX/CharmEX/DEX/DAG/OI for 3 nearest expiries
2. FreeFlow vol endpoint: current_iv, rv_iv_ratio, hv21
3. Yahoo Finance: 2 years of daily OHLC for NQ=F (fallback QQQ) — feeds entropy and PCA
4. `bias_output.json` (bundled via `require()` at build time): macro confluence

**Computed inside the handler, in order:**
1. `aggregateDataset(data)` → per-strike Greeks
2. `computeGammaFlip(strikes, futuresPrice)` → linear-interpolated GEX zero crossing
3. **Vol-scaled gamma regime band**: `_ivBand = max(30, 0.5 × futuresPrice × IV/100 / sqrt(252))`.
   `POSITIVE` if `(price - flip) > _ivBand`; `NEGATIVE` if `< -_ivBand`; else `NEAR_FLIP`.
   This is the 1σ half-day move scaled by half — adaptive across vol regimes.
4. `classifyVolRegime(iv, rvIvRatio)` → EXPANSION / NEUTRAL / CONTRACTION
5. `getWeights(volRegime, gammaRegime)` → 2D regime weight lookup (3×3 grid in lib/options.js)
6. `scoreLevels(strikes, weights, futuresPrice)` → filtered + scored levels with `wall_reaction` tag
7. `computeHGEXNorm(levels)` → normalized Shannon entropy of |GEX| distribution
8. `computeTopWall(levels, futuresPrice)` → proximity-weighted top wall (e-fold = 200pts)
9. `nearbyStrikes(strikes, futuresPrice)` → FILTER_PCT-only set (broader than scored levels)
10. `computeAggregateGreeks(nearby)` + `applyBiasTable(aggregates, volRegime, priceVsFlip)` → bias.pdf tag
11. `computeReturnEntropy(yahoo.bars.map(b => b.close))` → STABLE / CRITICAL / UNKNOWN
12. `computePCA(yahoo.bars)` → PC1/PC2/PC3 + `pc1_momentum_valid` guard
13. `classifyIntradayBias({…})` → bias + confidence + reason, with continuous evidence accumulator

**Confidence is a continuous evidence score, bucketed once at the end** (no ordinal clamping
chain — Phase 4 fix). Each modifier adds/subtracts:
- bias.pdf primary tag confirms (+strength) or conflicts (−strength) with the directional call
- walls.pdf wall_reaction confirms or conflicts (strength-2 conflicts trigger WALL_BREAKDOWN air-pocket flag)
- GEX dispersion (H_GEX_norm > 0.6) → −1
- Macro neutral → −1
- In NEGATIVE gamma regime: all evidence scaled by `GAMMA_ASYMMETRY_RATIO = 0.344` (Dim et al. 2025)

Final mapping: `evidence ≥ 1 → HIGH`, `≤ -1 → LOW`, else `MODERATE`.

**Hard gate**: CRITICAL entropy → `NO_BIAS, AVOID`. Bypasses everything else.

---

## Layer 3 — Levels Scoring (netlify/functions/lib/options.js)

Per-strike score:
```
score = (gex_norm * w_gex + vex_norm * w_vex + charmex_norm * w_charmex
       + oi_norm * w_oi + dag_norm * w_dag) × 100
```
Each component min-max normalized across nearby strikes. Weights come from a 3×3 table
indexed by (volRegime, gammaRegime).

**Per-level extras attached by `scoreLevels`:**
- `type`: `"CALL WALL"` or `"PUT WALL"` + ` + VOL SENSITIVE` if `|VEX| / |GEX| > 2.0`
- `wall_reaction`: walls.pdf tag from `classifyWallReaction(level)` — one of
  `CALL_WALL_{BEARISH_REJECT,BEARISH_BREAKDOWN,BULLISH_SQUEEZE,BULLISH_GRIND,MIXED}` or
  `PUT_WALL_{BULLISH_SUPPORT,VULNERABLE,BULLISH_REVERSAL,WEAK_BOUNCE_FADE,MIXED}`

**Constants:**
- `FILTER_PCT = 5.0` — strikes within ±5% of futures price
- `MIN_SCORE = 20.0` — discard scored strikes below this
- `PROXIMITY_EFOLD = 200.0` — in `intraday.js` for top-wall ranking (true halflife = 200·ln2 ≈ 139pts)
- `REGIME_WEIGHTS` — 3×3 (vol × gamma) → {gex, vex, charmex, oi, dag}; theoretically motivated but
  **empirically unvalidated** until snapshot logs accumulate

---

## PDF-Derived Methodology (clauderesources/)

Source PDFs that inform Layer 2 and Layer 3:

| PDF | What it provides |
|---|---|
| `bias.pdf` | Aggregate bias rules: (GEX sign × Charm sign × Vanna sign × IV regime × flip side) → primary bias tag |
| `walls.pdf` | Per-wall reaction table: (wall type × DEX × Charm × Vanna) → expected behavior |
| `OPTIONSFLOW.pdf` | Glossary / mechanics: Greek definitions, dealer hedging, GEX/DEX/VEX/TEX explanations |
| `1.pdf` (Elms 2026) | Empirical: modern SPX shows AMPLIFICATION not pinning. High OI ↔ wider ranges (p=0.0003) |
| `2.pdf` (Dim/Eraker/Vilkov 2025) | Empirical: MM 0DTE net gamma positive on avg; positive-gamma attenuation ~3× stronger than negative-gamma amplification |
| `3.pdf` (Garmash 2024) | Confirms gamma-regime → mean-reversion vs momentum mapping |

**Helpers in `lib/options.js`:**
- `classifyWallReaction(level)` — walls.pdf table → wall_reaction tag
- `computeAggregateGreeks(strikeArr)` — sign triple + totals across the passed set
- `applyBiasTable(aggregates, volRegime, priceVsFlip)` — bias.pdf table → primary bias tag
- `WALL_REACTION_DIR` / `BIAS_TAG_DIR` — tag → `{dir: BULL/BEAR/NEUTRAL, strength: 0/1/2}` maps
- `GAMMA_ASYMMETRY_RATIO = 0.344` — from Dim et al. Table 3 (-0.022 / -0.064 = 0.344)
- `PINNING_REGIME_ACTIVE = false` — from Elms 2026; exported but not yet consumed by logic

---

## Frontend (index.html + auth.js)

Single-file static dashboard served by Netlify. Three tabs (`#bias`, `#intraday`, `#levels`)
routed via URL hash. All DOM construction uses an `el(tag, attrs, ...children)` helper — no
framework, no template strings.

**Auth**: `login.html` + `auth.js` gate the dashboard (user `aseras`, SHA-256 password hash in `auth.js`).
Login stores a base64-encoded `{user, ts}` token in `localStorage` as `vanta_session`. All three
data functions require an `Authorization: Bearer <token>` header and return 401 without a valid,
unexpired token. The token is not server-signed → forgeable by anyone who reads `auth.js`. Adequate
for a small trusted user base; not for sensitive data.

**`netlify.toml`** publishes from `.` (root), functions from `netlify/functions/`, proxies
`/api/*` → `/.netlify/functions/:splat`.

---

## Data & Persistence

```
data/raw/           ← NQ/ES OHLCV (daily, 4h, 1h, 15m, 5m, 1m per contract)
data/processed/     ← outputs of scripts 01–08; inputs to weekly report
  weekly_accuracy_log.csv   ← cumulative O→C labeled track record (06_check_accuracy.py)
  blind_predictions_*.csv   ← model predictions for backtest weeks
  cot_NQ.csv, cot_ES.csv    ← CFTC COT history (refreshed by 07_cot_features.py)
  NQ_daily_clean.csv, etc.  ← back-adjusted continuous price series
  model_dataset_enriched.csv ← rolling-window percentile context for live regime scoring
models/             ← price_model.pkl, vol_model.pkl, *_features.pkl
logs/               ← levels_YYYY-MM-DD.csv snapshots (gex-snapshots branch)
                    + intraday_inputs_log.jsonl + snapshot_index.csv
bias_output.json    ← Layer 1 live output (bundled into bias.js at esbuild time)
levels_data.json    ← LOCAL DEV ONLY static snapshot (live dashboard never reads it)
```

**`bias_output.json`** is auto-updated by `.github/workflows/weekly-macro-update.yml` every Friday
at 10 PM UTC. Schema: `meta`, `confluence`, `macro_regime`, `price_model`, `vol_forecast`, `cot`,
`vix_term_structure`, `key_levels`. Requires `FRED_API_KEY` repo secret.

**`gex-snapshots` branch** holds the accumulated level/snapshot CSVs. Populated by
`.github/workflows/gex-snapshot.yml` every 15 min during 03:00-17:00 ET, Mon-Fri. Requires
`FF_SESSION` repo secret.

---

## Edge & Calibration Status

**What's empirically validated:**
- Gamma regime → vol regime mapping (Dim/Eraker/Vilkov 2025 on SPX, generalizes to NQ)
- Modern SPX is in amplification regime not pinning (Elms 2026)
- Macro weekly hit rate of 76.9% over 13 tradeable weeks (statistically significant vs random,
  NOT significant vs the 56.2% bull base rate — equities drift up)

**What's theory-grounded but uncalibrated on NQ:**
- bias.pdf primary-bias rules (rules from vendor methodology, untested on NQ outcomes)
- walls.pdf per-wall reaction tags (same)
- Shannon entropy threshold at 75th percentile of rolling 252-day window
- PCA pc1_momentum_valid threshold at 0.3 combined |loadings|
- `STRONG_WALL = 60`, `EXCEPTIONAL_WALL = 75`, `H_GEX_CONFIDENCE_CUT = 0.6`
- `PROXIMITY_EFOLD = 200pts`
- `REGIME_WEIGHTS` 3×3 grid in `lib/options.js`

**What needs labeled-outcome data to fix:**
- All of the above thresholds
- Whether walls actually predict reactions at the specific tag levels
- Whether the bias-table conflicts/confirms add or subtract from accuracy
- Whether the negative-gamma scaling ratio of 0.344 is right for NQ (was derived on SPX)

**Path to calibration:**
1. `gex-snapshots` branch accumulates 15-min snapshots automatically
2. Manual trading journal joins outcomes to snapshots by timestamp
3. After ~30-60 trading days and ~30-50 labeled trades, stratify hit rate by:
   - `top_wall.score` deciles (verifies STRONG_WALL threshold)
   - `(gammaRegime, volRegime)` 9-cell grid (verifies regime weights)
   - `H_GEX_norm` deciles (verifies dispersion penalty)
   - `wall_reaction` tag (verifies walls.pdf table)
4. Re-fit thresholds and (eventually) `REGIME_WEIGHTS` against the labeled set

---

## Key Constraints & Gotchas

- **Three independent layers** — don't conflate. Layer 1 is weekly XGBoost; Layer 2 is live
  options-flow + entropy/PCA; Layer 3 is per-strike scoring. Each has its own update cadence,
  its own JSON output, and its own dashboard tab.
- **Python intraday script is NOT canonical** — `intraday.js` is the production path. Python is
  for hindcasting only and intentionally lags the JS (no walls.pdf tags, no vol-scaled band).
- **COT data lag** — CFTC releases Tuesday data on Friday. The COT staleness check refreshes
  automatically if >9 days old. `nq_lev_pctile` comes from `cot_NQ.csv` (historical), not the
  live feed.
- **All time arithmetic must be ET** — `freeflow_logger.py` and `schedule_freeflow_logger.py`
  use ET explicitly so they're correct on UTC-based CI runners.
- **`--live` mode bypasses processed CSVs** — `05_weekly_report.py --live` downloads fresh
  data from yfinance. The processed CSVs feed scripts 03-04 (training) only.
- **VIX term structure display sign convention** — in `--live` mode `vix_ratio` and `vvix` are
  current unshifted values; historical weeks use the shifted-by-1-week values to avoid lookahead.
- **Model artifacts must exist** — `05_weekly_report.py` loads `models/*.pkl` on import. Run
  `04_train_model.py` first if missing.
- **FF_SESSION expires periodically** — if Levels tab shows auth errors, update both `.env` (local)
  and the GitHub repo secret. The Netlify env var is separate again.

---

## Autonomy & Operations (for Claude Code itself)

When making changes:
- **Don't tune uncalibrated thresholds without data** — every constant in "Edge & Calibration
  Status > theory-grounded but uncalibrated" should remain frozen until snapshot logs + trade
  journal provide stratified hit rates.
- **Don't conflate layers** — a fix to bias-table logic doesn't touch the XGBoost pipeline.
- **Update CLAUDE.md when adding helpers** — this file is the single source of truth for
  navigating the codebase. If a new function is added to `lib/options.js`, document it here.
- **Three layers, three commits** — when changes span layers, prefer separate commits per layer.
- **The Python intraday script is intentionally simpler than the JS** — port new logic to it
  only when it's specifically needed for backtesting.
