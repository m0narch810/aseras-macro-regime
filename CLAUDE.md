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
1–12. (same GEX/entropy/PCA pipeline as before — unchanged)
13. `classify2YSignal(shyBars)` → RISING_FAST / RISING / STABLE / FALLING / UNAVAILABLE
14. `classifyBOJSignal(usdjpyBars)` → CARRY_UNWIND / YEN_STABLE / YEN_WEAKENING / UNAVAILABLE
15. `getLiquidityTrend(macroBiasData)` → IMPROVING / STABLE / DETERIORATING (from net_liq_wow score)
16. `getCotLabel(macroBiasData)` → FUMES_LONG / NEUTRAL / EXTREME_SHORT (from nq_lev_pctile)
17. `classifyRTHBias({…})` → BULLISH / BEARISH / NEUTRAL / UNKNOWN with bull/bear counts
18. `classifyOpenArchetype({…})` → TYPE_A / TYPE_B / TYPE_C / TYPE_D + confidence 0-5 + signals[]
19. `getConfig()` → loads archetype names/descriptions from gitignored methodology_config or env var

**Open Archetype scoring** (`classifyOpenArchetype`):
- Scores all 4 types based on GEX structure signals (ivBand, flipDiff, dex_sign, vex_sign, charm_sign, wall proximity)
- Each condition adds 1 point; max score 5; winner displayed in the big action panel
- TYPE_A and TYPE_C are bull-resolving (`dir: 'bull'`); TYPE_B and TYPE_D are bear-resolving
- Actual archetype names/descriptions are in the gitignored `methodology_config.js`

**RTH Bias verdict** (`classifyRTHBias`):
- Weighted signals: 2Y yield (1-2 pts), liquidity (1), COT (1), BOJ (2 for CARRY_UNWIND), weekly macro (1)
- BEARISH when bear score ≥ 2.5; BULLISH when bull score ≥ 2.5; else NEUTRAL

**Options-flow bias** (`classifyIntradayBias`) is still computed but shown as secondary context ("Options Flow Bias" section) below the main RTH panel. CRITICAL entropy gate still bypasses this classifier.

**Hard gate**: CRITICAL entropy → `NO_BIAS, AVOID` on the options classifier only; RTH Bias still shows.

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
| `daily macro bias by dxrk.pdf` | RTH macro 5-factor framework: 2Y yield, TGA/RRP, COT crowding, BOJ signal, surprise mechanism |
| `weekly macro bias by dxrk.pdf` | Weekly HTF methodology: fed stance, COT extremes, net liquidity, yield curve, cross-asset divergence |
| `how to predict market open by dxrk.pdf` | 4 open archetypes (TYPE_A/B/C/D), manipulation tells, DEX-vs-open-move framework |
| `market_definitive_research.pdf` | Chapter 10: NY AM Daily Prediction Protocol (-9 to +9 scoring); 5-phase framework |

**Helpers in `lib/options.js`:**
- `classifyWallReaction(level)` — walls.pdf table → wall_reaction tag
- `computeAggregateGreeks(strikeArr)` — sign triple + totals across the passed set
- `applyBiasTable(aggregates, volRegime, priceVsFlip)` — bias.pdf table → primary bias tag
- `WALL_REACTION_DIR` / `BIAS_TAG_DIR` — tag → `{dir: BULL/BEAR/NEUTRAL, strength: 0/1/2}` maps
- `GAMMA_ASYMMETRY_RATIO = 0.344` — from Dim et al. Table 3 (-0.022 / -0.064 = 0.344)
- `PINNING_REGIME_ACTIVE = false` — from Elms 2026; exported but not yet consumed by logic

**New helpers in `netlify/functions/intraday.js` (RTH Bias layer):**
- `classify2YSignal(bars)` — 5-day ROC of SHY ETF → RISING_FAST / RISING / STABLE / FALLING / UNAVAILABLE
- `classifyBOJSignal(bars)` — 3-day ROC of USDJPY=X → CARRY_UNWIND / YEN_STABLE / YEN_WEAKENING / UNAVAILABLE
- `getLiquidityTrend(macroBiasData)` — reads `net_liq_wow` factor score from bias_output.json
- `getCotLabel(macroBiasData)` — reads `nq_lev_pctile` from bias_output.json → FUMES_LONG / NEUTRAL / EXTREME_SHORT
- `classifyRTHBias({yieldSignal, liquidityTrend, cotLabel, bojSignal, macroConfluence})` — 5-factor RTH verdict
- `classifyOpenArchetype({gammaFlip, futuresPrice, aggregateGreeks, levels, ivBand})` — scores 4 open types (TYPE_A/B/C/D), returns winner + confidence 0-5
- `getConfig()` — lazy-loads methodology_config.js (gitignored) or METHODOLOGY_CONFIG env var or empty fallback

**Methodology config privacy model:**
- `netlify/functions/lib/methodology_config.js` — **gitignored**; contains archetype names, descriptions, action text, RTH factor labels (dxrk framework specifics)
- `netlify/functions/lib/methodology_config.example.js` — **committed** skeleton with empty strings; shows structure only
- Production Netlify: set `METHODOLOGY_CONFIG` env var to base64-encoded JSON of the real config
- Committed intraday.js code uses abstract `TYPE_A/B/C/D` labels; display text injected at runtime
- `scripts/methodology_config.py` — **gitignored**; Python counterpart (weekly report narrative labels, not yet implemented)

---

## Frontend (index.html + auth.js)

Single-file static dashboard served by Netlify. Three tabs (`#bias`, `#intraday`="RTH Bias", `#levels`)
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
