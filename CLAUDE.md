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
│  hold_prob: mechanical wall reliability (dealer-mechanics formula, no fitting) │
│  Surface: dashboard #levels tab, /.netlify/functions/levels             │
└─────────────────────────────────────────────────────────────────────────┘
```

**Honest edge claims by layer:**

| Layer | Backtest result | Statistical strength | Trading-grade? |
|---|---|---|---|
| 1 — Macro weekly | 10/13 = 76.9% (Feb-May 2026) | Significant vs 50% random (p=0.026); NOT significant vs 56.2% bull base rate (p=0.067) | Borderline; n=13 too small |
| 2 — Intraday | Untested on outcomes | Theory-grounded (gamma regime: Dim 2025; entropy gate: regime-switching lit) | No — needs ≥30 days of labeled snapshots |
| 3 — Levels score | Composite score AUC 0.53 (near-random) | Arbitrary regime-weight table, empirically unvalidated | No |
| 3 — hold_prob | Not backtested (by design) | Theory-grounded: dealer-mechanics reliability formula (gamma regime, GTBR inelasticity, hedge polarity, skew, pinning) | No — mechanistic prior, awaiting forward-test validation |

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
| Retrain hold_prob (QQQ only) | `scripts/label_intraday_touches.py` → `scripts/fit_intraday_wall_model.py` |
| Retrain hold_prob (QQQ + SPY combined) | above + SPY pipeline → `fit_intraday_wall_model.py --combined` |
| Rebuild SPY 2020-2022 GEX profiles | `scripts/decode_spy_eod.py` |
| Label SPY touch events | `scripts/label_spy_touches.py` |
| Validate wall scoring against outcomes | `scripts/calibration_summary.py` |
| Validate scoring + hold_prob vs realized NQ outcomes (self-supervised) | `scripts/validate_walls.py` |
| Map accurate %-swing reversals to projected strikes (smoothed ratio) | `scripts/map_reversals.py` |
| Run reversal trade parameter search | `scripts/reversal_backtest.py` |
| Run limit-order simulation | `scripts/limit_order_backtest.py` |
| VP utilities (POC, VAH/VAL, HVN, LVN) | `scripts/compute_volume_profile.py` |

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
Auto-runs Mon-Fri 9 AM ET (pre-market refresh) AND Friday 10 PM UTC (full COT update) via GitHub Actions.
On Windows, prefix with `$env:PYTHONIOENCODING="utf-8";` to avoid cp1252 Unicode errors.

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
python scripts/label_intraday_touches.py                  # builds intraday_touches.csv (~489K events, includes VP features)
python scripts/fit_intraday_wall_model.py                 # QQQ-only model
python scripts/fit_intraday_wall_model.py --combined      # QQQ + SPY combined (requires spy_touches.csv)
# Then manually update _LR coefficients in netlify/functions/lib/options.js from the JSON
```

**Rebuild SPY 2020-2022 pipeline:**
```bash
python scripts/decode_spy_eod.py             # SPY EOD options → gex_profiles_spy/ (756 days)
python scripts/label_spy_touches.py          # ES 1m touch labeling → spy_touches.csv (~22K events)
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

Per-strike composite score:
```
raw   = (gex_norm * w_gex + vex_norm * w_vex + charmex_norm * w_charmex
       + oi_norm * w_oi + dag_norm * w_dag) × 100
score = raw × (0.5 + 0.5 × protrusion) × regimeRelevance
```
Weights come from a 3×3 table indexed by (volRegime, gammaRegime). **Empirically unvalidated
magnitude rank — use `hold_prob` for reliability.**

**Normalization (rewritten 2026-06-04 — was crushing real walls):** components are
normalized with `normalizeRobust` (scale by the 90th-percentile magnitude, clamped to [0,1])
instead of min-max — a single monster wall no longer defines the ceiling and zero-out every
mid-tier intraday level. `net_gex` additionally uses `normalizeGexPerSide` (call walls scaled
among calls, put walls among puts) so a giant put wall doesn't suppress every call wall.

**Protrusion multiplier softened** `0.25+0.75×prot` → `0.5+0.5×prot`: shoulder/ramp walls
(the rungs price reverses at on a gamma ladder) keep ≥50% credit instead of being knocked
below `MIN_SCORE`.

**`regimeRelevance` (NEW — regime + flip position enters the surface, not just hold_prob):**
computed per wall from spot vs `gammaFlip` using the same vol-scaled band as `levels.js`.
Reversal geometry = resistance above spot (call wall) or support below spot (put wall).
POSITIVE gamma → reversal walls ×1.15 / acceleration-geom ×0.90; NEGATIVE → reversal ×0.85;
NEAR_FLIP → reversal ×1.00 / other ×0.95; UNKNOWN → ×1.0.

**Guaranteed surface (NEW):** the top-5 gross-gamma walls per side within ±2.5% are ALWAYS
kept even if the composite filters them out — a genuine rank-5 reversal wall can never be
silently dropped. Each level carries `surfaced_by: 'score' | 'gamma_rank'`.

**NMS tightened** `0.006` (≈4.4 QQQ pts) → `0.0025` (≈1.8 QQQ pts): the old separation merged
distinct 0DTE dollar strikes (e.g. 733 vs 735) into one zone, demoting a strike you reverse at
to CONTEXT under its neighbour. Now only immediate $1 neighbours / futures-rounding dupes collapse.

**`hold_prob`** — mechanical Wall-Hold Reliability (R), attached to every scored level.
Derived purely from options-dealer mechanics — **no historical backtest fitting** (the prior
QQQ+SPY 511K-touch-event LR/XGBoost model was removed; that data was not predictive):
```
R = cap( B_regime × P × O × A_dex × PCR × S_skew × F_term × F_vrp × F_gtbr × G_pin , 1.0)
```
- `B_regime` — global gamma baseline (the ONLY place long/short gamma enters): smooth ramp across
  the vol-scaled NEAR_FLIP band — `diff ≥ band → 0.5`, `diff ≤ −band → 0.1`, linear in between
  (`0.3 + 0.2·diff/band`, = 0.3 exactly at flip). Replaces the old hard 0.1/0.3/0.5 step so a
  flip estimate a strike or two off nudges B by a few % instead of swinging the wall 5×.
- `P` — protrusion 0–1 mapped to 0.5–1.5 (dominant local node)
- `O` — one-sidedness `|GEX|/ag` mapped to 0.85–1.15 (decisive vs offsetting dealer gamma)
- `A_dex` — hedge polarity: counter-trend DEX (above spot & DEX>0, or below spot & DEX<0) ×1.25,
  pro-trend ×0.5 (acceleration zone)
- `PCR` — dominant-side OI asymmetry `max(call_oi,put_oi)/(call_oi+put_oi)` → 0.9–1.1
- `S_skew` — `|strike_iv − ATM| / ATM > 0.20` → ×1.15 (side-aware skew intensity)
- `F_term` — `hv5/hv63 > 1.25` → ×0.85 (short-vol spike weakens walls); no-op if hv absent
- `F_vrp` — `rv_iv_ratio < 0.5` → ×1.15 (rich VRP favors mean-reversion); no-op if absent
- `F_gtbr` — `|dist| > GTBR` → ×0.2 (inelastic momentum breaks through)
- `G_pin` — time > 14:00 ET & DAG in top decile of nearby strikes → ×1.25 (late pinning magnet)
- Sign convention: `net_gex > 0` = call-dominated (call wall), `< 0` = put-dominated (put wall) —
  NOT dealer long/short gamma. Long/short gamma is captured only by `B_regime`.
- All inputs come live from FreeFlow; the formula is in `computeHoldProb(level, ctx)` in `lib/options.js`.
  No model artifact required.

**`confluence`** — 1 when GEX_norm + VEX_norm + CharmEX_norm all ≥ 40 at the same strike.
These are the highest-quality walls; `hold_prob` is meaningfully higher on confluence=1 walls.

**Per-level fields from `scoreLevels`:**
- `score` — composite magnitude rank (0-100)
- `hold_prob` — mechanical Wall-Hold Reliability R (0-1); pure dealer-mechanics formula, no fitting
- `hps_score` — mechanistic checklist count (0-5)
- `hps_label` — `"HIGH"` (≥4) / `"MEDIUM"` (3) / `"LOW"` (0-2)
- `hps_conditions` — `{regime_positive, gtbr_inside, dex_aligned, charm_vanna, magnitude_outlier}`
- `net_tex` — theta exposure ($ time-decay/day). FreeFlow returns no theta, so it's
  derived per row via the driftless gamma-theta identity `θ = -½·Γ·S²·σ²` (T cancels;
  reuses FreeFlow's gamma + per-strike iv_pct), summed × OI × 100 / 365. Always negative
  (book bleeds decay); tracks gross gamma. Not consumed by hold_prob — available only.
  Recoverable on logged snapshots from `net_gex` + `strike_iv` via the same identity.
- `type` — `"CALL WALL"` or `"PUT WALL"` + ` + VOL SENSITIVE` if `|VEX| / |GEX| > 2.0`
- `wall_reaction` — tag from `classifyWallReaction(level)` (private reaction table)
- `confluence` — boolean int (from FreeFlow data, not always populated)
- `surfaced_by` — `'score'` (cleared MIN_SCORE) or `'gamma_rank'` (kept by the guaranteed
  top-5-per-side gross-gamma backstop despite a sub-threshold composite)
- `is_dominant` — survived per-side NMS (the local leader for its price zone)
- `conviction` — `'STANDALONE'` (dominant + hps≥4) / `'CONFIRM'` (dominant + hps=3) / `'CONTEXT'`
- `watch_suppressed` — `true` when spot is BELOW `gammaFlip` (negative gamma) and this is a
  below-spot put wall that is NOT squeeze-grade (`is_dominant && hold_prob ≥ SQUEEZE_HOLD_PROB`).
  Below-spot put walls in negative gamma are acceleration rungs, not support (2026-06-05: a −4.8%
  liquidation sliced every one, incl. a −0.51B wall, without a pause). The dashboard's
  `buildBiasLevels` drops suppressed walls from the "Long at"/Watching panel — if all are
  suppressed the long side goes empty with a "negative gamma, don't fade" note (`longSuppressed`).
  Trigger is raw spot-below-flip, NOT the band-gated NEGATIVE regime: the vol-scaled band reads
  NEAR_FLIP too long (at the 06-05 open spot was 225pts below flip but still NEAR_FLIP), so the
  band would make suppression late exactly when it matters. The squeeze exception self-tightens
  with depth: `hold_prob`'s `B_regime` ramps 0.5→0.1 below flip, so clearing 0.55 is feasible near
  the flip and mechanically impossible deep in negative gamma — only a near-flip squeeze qualifies.

**Top-level fields added to `levels.js` response:**
- `gtbr_pts` — expected remaining NQ range in points at time of request (same formula as `computeGTBR`)

**`scoreLevels(strikes, weights, futuresPrice, volRegime, gammaFlip)`** — signature takes
`volRegime` and `gammaFlip` to pass through to `computeHoldProb`.

**Constants:**
- `FILTER_PCT = 5.0` — strikes within ±5% of futures price
- `MIN_SCORE = 20.0` — discard scored strikes below this
- `SQUEEZE_HOLD_PROB = 0.55` — in negative gamma, a below-spot put wall must clear this
  `hold_prob` (and be `is_dominant`) to escape `watch_suppressed` (squeeze-grade exception)
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
| `label_intraday_touches.py` | Minute-by-minute touch detection using nearest prior snapshot → `intraday_touches.csv` (includes VP features) |
| `fit_intraday_wall_model.py` | Train LR + XGBoost on touch events → `models/wall_score_intraday.json`; `--combined` merges QQQ + SPY |
| `fit_wall_score_model.py` | Validates composite score vs outcomes (shows AUC 0.53 = near-random) |
| `calibration_summary.py` | Statistical report on wall hold rates by regime/type/threshold |
| `reversal_backtest.py` | Vectorized parameter search: stop/target/filter combos → `reversal_backtest.csv` |
| `compute_volume_profile.py` | VP utilities: `build_vp()`, `get_poc_vah_val()`, `get_hvns()`, `get_lvns()`, `build_vp_cache()`, `build_week_vp_cache()` |
| `decode_spy_eod.py` | SPY EOD options chain (OptionsDX format) → per-day ES GEX profiles in `gex_profiles_spy/` |
| `label_spy_touches.py` | ES 1m touch labeling from SPY GEX profiles → `spy_touches.csv` |
| `limit_order_backtest.py` | Pre-loaded limit-order simulation: identifies qualifying walls per snapshot, simulates fills and outcomes |

**Key design facts:**
- OPRA CBBO-1m provides bid/ask quotes, not true OI. Quote size (bid_sz + ask_sz) is used as
  an OI proxy — underestimates true dealer exposure but preserves relative strike ranking.
- `decode_opra_day.py` filters to expiry == trade_date (0DTE only) and takes 13 snapshots per
  day at 30-min intervals (9:31, 10:00, 10:30 … 15:30 ET).
- Intraday spot price is adjusted per-snapshot: `spot_t = spot_open × (nq_t / nq_open)` using
  NQ 1-minute bars from `data/processed/NQ_1m_clean.csv`.
- The `vex_over_gex` ratio has mean ≈ 0.0003 in OPRA data (tiny vega relative to gamma at 0DTE).
  FreeFlow live data has vex/gex ≈ 0.2+. Do not use live ratio features in `computeHoldProb`.

**SPY EOD pipeline facts (`decode_spy_eod.py` + `label_spy_touches.py`):**
- Source: `archive (2)/spy_2020_2022.csv` — OptionsDX/CBOE format, brackets in column names (`[QUOTE_DATE]`)
  that must be stripped. Parse with `pd.to_numeric(..., errors='coerce')` — columns arrive as strings.
- Uses EOD snapshot (4pm) as next trading day's opening wall map. DTE==0 preferred; DTE==1 fallback.
- SPY→ES conversion: ES 9:31 AM open / SPY UNDERLYING_LAST, computed daily from `data/raw/ES/1Min_ES.csv`.
- ES 1m file is semicolon-separated, European numbers: read with `sep=";", thousands=".", decimal=","`.
- VP features are zeroed in `spy_touches.csv` — only QQQ OPRA data has intraday snapshots for VP computation.
- `label_spy_touches.py` uses prior-day profile as wall map, not same-day snapshots.

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
- `computeHoldProb(level, ctx)` — mechanical Wall-Hold Reliability R (0-1); pure dealer-mechanics
  formula (see Layer 3 section), no fitting. `ctx = {gammaFlip, futuresPrice, timeOfDayET, atmIv,
  rvIvRatio, hvTermRatio, gtbr, dagDecileThr}`, assembled once per request by `scoreLevels`
- `computeGTBR(futuresPrice, iv, timeOfDayET)` — expected remaining NQ range in points;
  full-day 1σ scaled by √(T_remaining/6.5); collapses toward expiry — key 0DTE insight
- `computeHPS(level, futuresPrice, iv, gammaFlip, volRegime, protrusion, timeOfDayET)` — 5-condition
  mechanistic checklist: `{score: 0-5, label: HIGH/MEDIUM/LOW, conditions: {regime_positive,
  gtbr_inside, dex_aligned, charm_vanna, magnitude_outlier}}`; complements hold_prob
- `scoreLevels(strikes, weights, futuresPrice, volRegime, gammaFlip, iv)` — now takes optional
  `iv` (6th arg); each level now includes `hps_score`, `hps_label`, `hps_conditions` fields
- `normalizeRobust(values)` — scale by 90th-percentile magnitude (clamped 0–1); outlier-resistant
  replacement for `normalizeAbs` in `scoreLevels` so one monster wall doesn't zero-out the rest
- `normalizeGexPerSide(netGex)` — robust-normalizes call walls and put walls separately
  (index-aligned); prevents a giant put wall from suppressing all call walls
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
- Watching panel shows a confirmation hint below the level rows:
  - `confluence=1` → green "Multi-Greek confluence ✓"
  - `confluence=0, hold_prob ≥ 0.55` → muted "Single-Greek lead — LVN or OTE adds conviction"
  - `confluence=0, hold_prob < 0.55` → red "Greeks partial — look for LVN or OTE entry confirmation"

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
  gex_profiles_spy/          ← SPY EOD GEX profiles (from decode_spy_eod.py); 756 days 2020-2022
  intraday_touches.csv       ← ~489K labeled QQQ touch events (gitignored — >100MB)
  spy_touches.csv            ← ~22K labeled SPY/ES touch events (gitignored)
  reversal_backtest.csv      ← output of reversal_backtest.py (gitignored)
  limit_order_backtest.csv   ← output of limit_order_backtest.py (gitignored)
  wall_outcomes.csv          ← daily OHLC-based wall outcomes (coarser than intraday_touches)
  weekly_accuracy_log.csv    ← cumulative O→C labeled track record
models/
  price_model.pkl, vol_model.pkl   ← Layer 1 XGBoost models
  wall_score_intraday.json         ← hold_prob LR coefficients + scaler params (source of truth)
  wall_score_intraday.pkl          ← trained LR + XGBoost objects (Python only)
dataidk/                     ← raw OPRA CBBO-1m .dbn.zst files (gitignored, large)
archive (2)/                 ← spy_2020_2022.csv — SPY EOD options chain 2020-2022 (gitignored, 1.28GB)
logs/                        ← levels_YYYY-MM-DD.csv snapshots (gex-snapshots branch)
bias_output.json             ← Layer 1 live output (bundled at build time)
```

**`bias_output.json`** auto-updated by `.github/workflows/weekly-macro-update.yml`:
- Mon-Fri 9 AM ET (pre-market VIX/DXY/yield refresh)
- Friday 10 PM UTC (full COT + weekly model run)
Requires `FRED_API_KEY` repo secret.

**`gex-snapshots` branch** holds accumulated level/snapshot CSVs. Populated by
`.github/workflows/gex-snapshot.yml` every 15 min during 03:00-17:00 ET, Mon-Fri. Requires
`FF_SESSION` repo secret.

---

## Edge & Calibration Status

**Empirically validated:**
- Gamma regime → vol regime mapping (Dim/Eraker/Vilkov 2025)
- Modern SPX in amplification not pinning regime (Elms 2026)
- Macro weekly hit rate 76.9% (13 weeks; significant vs random, NOT vs bull base rate)

**Validation harness** (`scripts/validate_walls.py`): replays OLD vs NEW scoring + hold_prob for
every logged 0DTE wall and labels HELD/BROKE against yfinance NQ=F 1m bars (touch + 30-min
2-bar-confirmed break). Strikes are placed via the smoothed-ratio projection (see strike-conversion
note below), NOT FreeFlow's strike_futures. Gated to `FIRST_GOOD_DAY = 2026-06-04` — earlier days
had only ~3-5 sparse snapshots and were deleted from the gex-snapshots branch.

**Validation run 2026-06-04 (today-only, n=589 touched 0DTE walls, base hold rate 56.2%):**
- **Surfacing fix is real (not overfit):** held-wall recall OLD 35% → NEW 96%; per-wall precision
  OLD 61% → NEW 57%; DOMINANT precision OLD 52% → NEW 54%. The old composite dropped ~65% of walls
  that actually held — quantifies the "733 wasn't flagged" complaint. NEW surfaces ~95% of touched
  walls, so its precision ≈ base rate; the DOMINANT tag (54%) does NOT beat the 56.2% base — we can
  surface candidates but **cannot yet rank/select which hold.**
- **`hold_prob` is confirmed broken (the strike correction clarified, did not rescue it):**
  rank AUC 0.444 (old) / 0.426 (new) — both <0.5, and the smooth-B change made it slightly worse.
  Calibration is starkly non-monotonic: the 0.4-0.6 bucket holds 82-93% but **every wall with
  R>0.6 broke (0/43 held).** The factors clearly carry information (buckets are far from random),
  but the formula combines them wrong — an AUC of 0.43 is an exploitable anti-signal, not noise.
  hold_prob needs to be REFIT on the labeled touches, not hand-tweaked.
- **Label caveat:** the touch+2-bar-break definition is noisy for near-spot walls (price
  oscillates → auto-"break"), which likely inflates the R>0.6 break rate (high R = close +
  aligned walls). The %-swing-reversal-off-strike definition (`scripts/map_reversals.py`,
  default 0.33% ≈ 100 NQ pts, scale-invariant) is a cleaner label and should replace
  touch-break before refitting.

**`hold_prob` (mechanical R) — theory-grounded, NOT yet validated on outcomes:**
- Pure dealer-mechanics formula (gamma regime, GTBR inelasticity, hedge polarity, one-sidedness,
  skew, late pinning). No backtest fitting — the prior QQQ+SPY touch-event model was removed.
- Each factor is mechanically motivated but the multipliers/thresholds are reasoned priors, not
  calibrated against realized hold rates. Needs forward-tested labeled snapshots to validate.
- Known soft spot: `B_regime` is a hard step at the gamma flip (0.5 vs 0.1) — near-flip snapshots
  swing ~5×. Consider a NEAR_FLIP midpoint (0.3) using the `gammaRegime` band from `levels.js`.

**Theory-grounded but uncalibrated:**
- Composite `score` (REGIME_WEIGHTS 3×3 grid) — AUC 0.53, near-random on 0DTE data
- Primary-bias rules (private methodology, untested on NQ outcomes)
- Per-wall reaction tags (same)
- Shannon entropy threshold at 75th percentile of rolling 252-day window
- PCA pc1_momentum_valid threshold at 0.3 combined |loadings|
- `STRONG_WALL = 60`, `EXCEPTIONAL_WALL = 75`, `H_GEX_CONFIDENCE_CUT = 0.6`
- `PROXIMITY_EFOLD = 200pts`
- `GAMMA_ASYMMETRY_RATIO = 0.344` (SPX-derived; NQ-specific value unknown)

**Strike→futures conversion accuracy (found 2026-06-04):** FreeFlow's `strike_futures`
(used by `levels.js` for `dist_nq`/GTBR and by `validate_walls.py` for touch detection) sits
~36 NQ pts ABOVE the smoothed-ratio projection. The user's TradingView method —
`ratio = SMA(NQ_close/QQQ_close, 100)` on extended-hours data, `nq_level = strike × ratio` —
matches actual reversal wicks far better (733 → 30166 vs FreeFlow 30201; price reversed at
30151-30164). `scripts/map_reversals.py` implements the smoothed-ratio method. **Implication:**
production wall NQ prices are biased high by ~30-50 pts, and the earlier validation's touch
labels were detected at mislocated strikes — so the AUC<0.5 result is partly confounded by
strike misplacement and must be re-run with smoothed-ratio strike locations before trusting it.

**ATM IV capture (fixed 2026-06-04):** `/vol/realized`'s `current_iv` (the snapshot `iv` column)
is intermittently null AND a different/larger tenor than the 0DTE option smile (≈35 vs ≈21). The
per-strike `iv_pct` smile is clean and always present (monotonic put-skew, e.g. 21.7→19.5 across
730→741). So `freeflow_logger.py` now also logs **`atm_iv`** = the at-the-money strike's own
`iv_pct` — smile-consistent, vol-endpoint-independent. Use `atm_iv` (not `iv`) for skew + IV_norm
features so both come from the same smile. (Live `levels.js`/`options.js` still use the
vol-endpoint `iv` for GTBR/regime — switching those to the smile ATM is a separate change that
rescales GTBR and needs verification.)

**Known data limitations:**
- OPRA CBBO uses quote size as OI proxy — underestimates true dealer exposure vs FreeFlow live data
- SPY EOD pipeline uses volume as OI proxy (resets daily) — weaker proxy than OPRA quote size
- The OPRA/SPY backtest pipeline (`fit_intraday_wall_model.py`, `models/xgb_wall.json`,
  `wall_score_intraday.*`) is **no longer wired into production** — `hold_prob` is now the mechanical
  formula in `computeHoldProb`. The scripts/artifacts remain on disk but are vestigial.

---

## Key Constraints & Gotchas

- **Three independent layers** — don't conflate. Layer 1 is weekly XGBoost; Layer 2 is live
  options-flow + entropy/PCA; Layer 3 is per-strike scoring. Each has its own update cadence.
- **Python intraday script is NOT canonical** — `intraday.js` is the production path. Python is
  for hindcasting only.
- **`hold_prob` is a mechanical formula, not a fitted model** — edit the multipliers/thresholds in
  `computeHoldProb` directly (they're reasoned dealer-mechanics priors). There is no `_LR`/XGBoost
  model in the production path anymore. Keep long/short-gamma logic confined to `B_regime`.
- **Windows encoding** — scripts with Unicode characters (→ ✓ ⚠) will crash on cp1252. Prefix with
  `$env:PYTHONIOENCODING="utf-8";` in PowerShell or run with `python -u`.
- **Large CSVs are gitignored** — `intraday_touches.csv`, `spy_touches.csv`, `reversal_backtest.csv`,
  `limit_order_backtest.csv` exceed 100MB and are excluded from git. Regenerate locally from pipeline.
- **COT data lag** — CFTC releases Tuesday data on Friday. Staleness check auto-refreshes if >9 days.
- **All time arithmetic must be ET** — `freeflow_logger.py` and CI runners use ET explicitly.
- **`--live` mode bypasses processed CSVs** — `05_weekly_report.py --live` downloads fresh from yfinance.
- **Model artifacts must exist** — `05_weekly_report.py` loads `models/*.pkl` on import. Run
  `04_train_model.py` first if missing.
- **FF_SESSION expires periodically** — update `.env`, GitHub repo secret, AND Netlify env var.
- **`scoreLevels` signature** — `(strikes, weights, futuresPrice, volRegime, gammaFlip, iv, volCtx)`.
  `iv` (6th) feeds GTBR + skew; `volCtx` (7th, optional) = `{rvIvRatio, hv5, hv63}` enables the VRP +
  term-structure filters in `computeHoldProb`. `levels.js` passes all seven; `intraday.js` omits
  `volCtx` (those filters no-op).

---

## Autonomy & Operations (for Claude Code itself)

- **Don't tune uncalibrated thresholds without data** — constants in "theory-grounded but
  uncalibrated" above stay frozen until labeled outcomes provide stratified hit rates.
- **`hold_prob` is a mechanical formula** — tune its multipliers in `computeHoldProb` directly; there
  is no fitted model to retrain. Keep long/short-gamma logic only in `B_regime`.
- **Don't conflate layers** — a fix to bias-table logic doesn't touch the XGBoost pipeline.
- **Update CLAUDE.md when adding helpers** — this file is the single source of truth for
  navigating the codebase. New functions in `lib/options.js` belong in the helpers section.
- **Three layers, three commits** — when changes span layers, prefer separate commits per layer.
- **The Python intraday script is intentionally simpler than the JS** — port new logic to it
  only when specifically needed for backtesting.
