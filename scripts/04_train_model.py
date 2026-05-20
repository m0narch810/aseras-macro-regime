import os
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import pickle  # noqa: S403 — loading trusted internal model artifacts only

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CONFIG ────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── LOAD ALL DATA ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(os.path.join(PROC_DIR, "model_dataset_cot.csv"),
                 index_col=0, parse_dates=True)
nq_daily = pd.read_csv(os.path.join(PROC_DIR, "NQ_daily_clean.csv"),
                       parse_dates=["date"]).set_index("date")
nq_4h = pd.read_csv(os.path.join(PROC_DIR, "NQ_4h_clean.csv"),
                    parse_dates=["date"]).set_index("date")
nq_1h = pd.read_csv(os.path.join(PROC_DIR, "NQ_1h_clean.csv"),
                    parse_dates=["date"]).set_index("date")

print(f"  Dataset:  {len(df)} weeks")
print(f"  NQ daily: {len(nq_daily)} bars")
print(f"  NQ 4h:    {len(nq_4h)} bars")
print(f"  NQ 1h:    {len(nq_1h)} bars")

# ══════════════════════════════════════════════════════════════
# PART 1: MACRO REGIME CLASSIFIER (RULES-BASED)
# ══════════════════════════════════════════════════════════════
# What this does:
# Instead of using ML (which failed at 50%), we classify the
# macro environment using percentile ranks of each indicator.
# Each input gets scored: bullish, bearish, or neutral.
# The combined score defines the regime.
#
# Why rules-based is better here:
# - Transparent: you can see exactly why it says risk-on/off
# - No overfitting: no model to memorize noise
# - The relationships (net liq up = risk on) are known and stable,
#   they just aren't precise enough for weekly direction calls

print("\n" + "="*60)
print("  PART 1: MACRO REGIME ENGINE")
print("="*60)

MACRO_COLS = ["net_liq_wow", "net_liq_4w", "vix_wow", "vix_4w",
              "us10y_wow", "us10y_4w", "dxy_wow", "dxy_4w"]

# Compute percentile ranks over expanding window
# (only uses data available up to that point — no leakage)
macro_pctls = df[MACRO_COLS].expanding(min_periods=20).rank(pct=True)

# Score each indicator for equity-bullish or equity-bearish
# Net liq rising = bullish -> high percentile = bullish
# VIX falling = bullish -> LOW percentile = bullish (so we invert)
# DXY falling = bullish -> LOW percentile = bullish (invert)
# US10Y: complex — sharp rise = bearish, mild fall = bullish

def score_macro(row):
    scores = {}

    # Net liquidity: high percentile = rising fast = bullish
    scores["net_liq_wow"] = 1 if row["net_liq_wow"] > 0.65 else (-1 if row["net_liq_wow"] < 0.35 else 0)
    scores["net_liq_4w"]  = 1 if row["net_liq_4w"]  > 0.65 else (-1 if row["net_liq_4w"]  < 0.35 else 0)

    # VIX: LOW percentile = falling = bullish (inverted)
    scores["vix_wow"]     = 1 if row["vix_wow"] < 0.35 else (-1 if row["vix_wow"] > 0.65 else 0)
    scores["vix_4w"]      = 1 if row["vix_4w"]  < 0.35 else (-1 if row["vix_4w"]  > 0.65 else 0)

    # DXY: LOW percentile = falling = bullish (inverted)
    scores["dxy_wow"]     = 1 if row["dxy_wow"] < 0.35 else (-1 if row["dxy_wow"] > 0.65 else 0)
    scores["dxy_4w"]      = 1 if row["dxy_4w"]  < 0.35 else (-1 if row["dxy_4w"]  > 0.65 else 0)

    # US10Y: sharp rise = bearish for equities
    scores["us10y_wow"]   = 1 if row["us10y_wow"] < 0.35 else (-1 if row["us10y_wow"] > 0.70 else 0)
    scores["us10y_4w"]    = 1 if row["us10y_4w"]  < 0.35 else (-1 if row["us10y_4w"]  > 0.70 else 0)

    total = sum(scores.values())
    return total, scores

# Apply to full dataset
macro_scores = []
macro_details = []
for idx in macro_pctls.index:
    row = macro_pctls.loc[idx]
    if row.isna().any():
        macro_scores.append(np.nan)
        macro_details.append({})
    else:
        total, details = score_macro(row)
        macro_scores.append(total)
        macro_details.append(details)

df["macro_score"] = macro_scores

def classify_regime(score):
    if pd.isna(score):
        return "UNKNOWN"
    if score >= 3:
        return "RISK-ON"
    elif score <= -3:
        return "RISK-OFF"
    elif score >= 1:
        return "LEAN RISK-ON"
    elif score <= -1:
        return "LEAN RISK-OFF"
    else:
        return "TRANSITION"

df["regime"] = df["macro_score"].apply(classify_regime)

# How well does regime predict direction?
regime_valid = df.dropna(subset=["label", "macro_score"])
regime_vs_actual = pd.crosstab(regime_valid["regime"],
                                regime_valid["label"].map(
                                    {-1: "Bear", 0: "Neutral", 1: "Bull"}))
print("\nRegime vs actual weekly direction:")
print(regime_vs_actual)

# Win rate per regime
for regime in ["RISK-ON", "LEAN RISK-ON", "TRANSITION",
               "LEAN RISK-OFF", "RISK-OFF"]:
    subset = regime_valid[regime_valid["regime"] == regime]
    if len(subset) > 0:
        bull_pct = (subset["label"] == 1).mean()
        bear_pct = (subset["label"] == -1).mean()
        print(f"  {regime:<16}  n={len(subset):>3}  "
              f"Bull: {bull_pct:.0%}  Bear: {bear_pct:.0%}")

# ══════════════════════════════════════════════════════════════
# PART 2: PRICE DIRECTION MODEL (ML)
# ══════════════════════════════════════════════════════════════
# This model works (64.5% OOS). Keep it as-is.

print("\n" + "="*60)
print("  PART 2: PRICE DIRECTION MODEL (XGBoost)")
print("="*60)

# Binary only — drop neutral
df_binary = df[df["label"] != 0].copy()
df_binary["label_binary"] = (df_binary["label"] == 1).astype(int)

PRICE_FEATURES = [
    "nq_week_ret", "nq_close_pos", "nq_week_range",
    "nq_mom_4w", "nq_mom_12w", "nq_vol_regime",
    "es_nq_spread", "realized_vol",
    "nq_1h_drift", "nq_1h_vol",
    # Top-4 COT features by importance (leakage-safe: shifted +1 week)
    "nq_lev_wow", "es_lev_wow",
    "nq_lev_pctile", "es_asset_mgr_net_pct",
]
PRICE_FEATURES = [c for c in PRICE_FEATURES if c in df_binary.columns]

TRAIN_END = "2021-12-31"
VAL_END   = "2023-12-31"

train_mask = df_binary.index <= TRAIN_END
val_mask   = (df_binary.index > TRAIN_END) & (df_binary.index <= VAL_END)
test_mask  = df_binary.index > VAL_END

X_train = df_binary.loc[train_mask, PRICE_FEATURES]
X_val   = df_binary.loc[val_mask,   PRICE_FEATURES]
X_test  = df_binary.loc[test_mask,  PRICE_FEATURES]
y_train = df_binary.loc[train_mask, "label_binary"]
y_val   = df_binary.loc[val_mask,   "label_binary"]
y_test  = df_binary.loc[test_mask,  "label_binary"]

weights = compute_sample_weight("balanced", y_train)

# ── OPTUNA TUNING ─────────────────────────────────────────────
print("  Running Optuna (100 trials)...")

def objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
        max_depth         = trial.suggest_int("max_depth", 2, 5),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
        min_child_weight  = trial.suggest_int("min_child_weight", 2, 10),
        objective         = "binary:logistic",
        eval_metric       = "error",  # classification error drives early stopping, not logloss
        early_stopping_rounds = 30,
        random_state      = 42,
        verbosity         = 0,
    )
    m = xgb.XGBClassifier(**params)
    m.fit(X_train, y_train, sample_weight=weights,
          eval_set=[(X_val, y_val)], verbose=False)
    return balanced_accuracy_score(y_val, m.predict(X_val))

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=False)

best = study.best_params
print(f"  Best val bal-accuracy: {study.best_value:.1%}")
print(f"  Best params: {best}")

# ── FINAL MODEL WITH BEST PARAMS ──────────────────────────────
price_model = xgb.XGBClassifier(
    **best,
    objective="binary:logistic",
    eval_metric="error",
    early_stopping_rounds=30,
    random_state=42,
    verbosity=0,
)
price_model.fit(X_train, y_train, sample_weight=weights,
                eval_set=[(X_val, y_val)], verbose=False)

DECISION_THRESHOLD = 0.50

print(f"  Early stop at tree: {price_model.best_iteration}")

for name, Xs, ys in [("Train", X_train, y_train),
                      ("Val", X_val, y_val),
                      ("Test", X_test, y_test)]:
    probs = price_model.predict_proba(Xs)[:, 1]
    preds = (probs >= DECISION_THRESHOLD).astype(int)
    acc  = accuracy_score(ys, preds)
    bacc = balanced_accuracy_score(ys, preds)
    baseline = max(ys.mean(), 1 - ys.mean())
    beat = "OK" if bacc > 0.52 else "--"
    print(f"  {name:<8} acc: {acc:.1%}  bal-acc: {bacc:.1%}  (baseline: {baseline:.1%})  {beat}")

y_test_preds = (price_model.predict_proba(X_test)[:, 1] >= DECISION_THRESHOLD).astype(int)
report = classification_report(y_test, y_test_preds,
         target_names=["BEARISH", "BULLISH"])
for line in report.split("\n"):
    print(f"    {line}")

importance = pd.Series(price_model.feature_importances_,
                       index=PRICE_FEATURES).sort_values(ascending=False)
print("  Top 10 features by importance:")
for feat, score in importance.head(10).items():
    bar = "#" * int(score * 200)
    print(f"    {feat:<26} {score:.4f}  {bar}")

# ══════════════════════════════════════════════════════════════
# PART 3: VOLATILITY MODEL
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 3: VOLATILITY FORECAST MODEL")
print("="*60)

ALL_FEATURES = list(set(PRICE_FEATURES + [
    "net_liq", "net_liq_wow", "net_liq_4w",
    "vix", "vix_wow", "vix_4w",
    "us10y", "us10y_wow", "us10y_4w",
    "dxy", "dxy_wow", "dxy_4w",
]))
ALL_FEATURES = [c for c in ALL_FEATURES if c in df_binary.columns]

vol_reg = xgb.XGBRegressor(
    n_estimators=500, max_depth=2, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=3.0, min_child_weight=4,
    early_stopping_rounds=50, random_state=42, verbosity=0,
)
vol_reg.fit(df_binary.loc[train_mask, ALL_FEATURES],
            df_binary.loc[train_mask, "fwd_vol"],
            eval_set=[(df_binary.loc[val_mask, ALL_FEATURES],
                       df_binary.loc[val_mask, "fwd_vol"])],
            verbose=False)

# Classify vol forecast into buckets
vol_preds  = vol_reg.predict(df_binary.loc[test_mask, ALL_FEATURES])
vol_actual = df_binary.loc[test_mask, "fwd_vol"].values
valid_mask = ~np.isnan(vol_preds) & ~np.isnan(vol_actual)
vol_corr   = np.corrcoef(vol_preds[valid_mask], vol_actual[valid_mask])[0, 1]
print(f"  Vol forecast correlation: {vol_corr:.3f}")
print(f"  (above 0.3 is useful, above 0.5 is good)")

# ══════════════════════════════════════════════════════════════
# PART 4: KEY LEVELS AND INTRADAY STRUCTURE
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 4: KEY LEVELS & STRUCTURE")
print("="*60)

# Weekly key levels from daily data
nq_weekly = nq_daily.resample("W-THU").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum"
})

# Prior week levels
nq_weekly["pw_high"]  = nq_weekly["high"].shift(1)
nq_weekly["pw_low"]   = nq_weekly["low"].shift(1)
nq_weekly["pw_close"] = nq_weekly["close"].shift(1)
nq_weekly["pw_mid"]   = (nq_weekly["pw_high"] + nq_weekly["pw_low"]) / 2

# 4-week high/low
nq_weekly["4w_high"]  = nq_weekly["high"].rolling(4).max().shift(1)
nq_weekly["4w_low"]   = nq_weekly["low"].rolling(4).min().shift(1)

# Weekly ATR (average true range of daily bars, aggregated weekly)
nq_daily["tr"] = np.maximum(
    nq_daily["high"] - nq_daily["low"],
    np.maximum(
        abs(nq_daily["high"] - nq_daily["close"].shift(1)),
        abs(nq_daily["low"]  - nq_daily["close"].shift(1))
    )
)
nq_weekly["atr_5d"]  = nq_daily["tr"].resample("W-THU").mean()
nq_weekly["atr_20d"] = nq_daily["tr"].rolling(20).mean().resample("W-THU").last()

# 4H trend: count of higher highs / higher lows in last 5 bars
def trend_score_4h(group):
    if len(group) < 5:
        return 0
    last5 = group.tail(5)
    hh = (last5["high"].diff() > 0).sum()
    hl = (last5["low"].diff() > 0).sum()
    ll = (last5["low"].diff() < 0).sum()
    lh = (last5["high"].diff() < 0).sum()
    return (hh + hl) - (ll + lh)  # positive = uptrend, negative = downtrend

nq_4h_trend = nq_4h.resample("W-THU").apply(trend_score_4h)
nq_4h_trend.name = "trend_4h"

# 1H momentum: last 10 bars average return
nq_1h["ret"] = nq_1h["close"].pct_change()
nq_1h_mom = nq_1h["ret"].resample("W-THU").apply(
    lambda x: x.tail(10).mean() if len(x) >= 10 else 0
)
nq_1h_mom.name = "momentum_1h"

# Merge levels into main dataset
levels = nq_weekly[["pw_high", "pw_low", "pw_close", "pw_mid",
                     "4w_high", "4w_low", "atr_5d", "atr_20d"]]
levels = levels.join(nq_4h_trend, how="left")
levels = levels.join(nq_1h_mom,   how="left")
levels.to_csv(os.path.join(PROC_DIR, "weekly_levels.csv"))
print("  Saved: data/processed/weekly_levels.csv")
print(f"  {len(levels)} weeks of level data")

# ══════════════════════════════════════════════════════════════
# PART 5: COMPREHENSIVE WEEKLY REPORT
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  COMPREHENSIVE BIAS REPORT — LAST 8 WEEKS")
print("="*60)

# Get last 8 weeks that exist in all datasets
last_dates = df.index[-8:]

for date in last_dates:
    print(f"\n  +--- Week of {date.date()} {'-'*35}")

    # 1. MACRO REGIME
    regime = df.loc[date, "regime"] if date in df.index else "N/A"
    m_score = df.loc[date, "macro_score"] if date in df.index else 0
    detail = macro_details[df.index.get_loc(date)] if date in df.index else {}
    print(f"  | MACRO REGIME:  {regime}  (score: {m_score:+.0f}/8)")
    if detail:
        bull_factors = [k for k, v in detail.items() if v > 0]
        bear_factors = [k for k, v in detail.items() if v < 0]
        if bull_factors:
            print(f"  |   Bullish: {', '.join(bull_factors)}")
        if bear_factors:
            print(f"  |   Bearish: {', '.join(bear_factors)}")

    # 2. PRICE DIRECTION
    if date in df_binary.index:
        prob = price_model.predict_proba(
            df_binary.loc[[date], PRICE_FEATURES])[:, 1][0]
        direction = "BULLISH" if prob >= DECISION_THRESHOLD else "BEARISH"
        confidence = max(prob, 1 - prob)
        print(f"  | PRICE BIAS:   {direction}  ({confidence:.0%} confidence)")
    else:
        print(f"  | PRICE BIAS:   NEUTRAL WEEK (skipped)")
        prob = 0.5

    # 3. VOLATILITY FORECAST
    if date in df_binary.index:
        vol_f = vol_reg.predict(
            df_binary.loc[[date], ALL_FEATURES])[0]
        vol_pctl = (df_binary["fwd_vol"].dropna() < vol_f).mean()
        if vol_pctl > 0.7:
            vol_label = "HIGH VOL"
        elif vol_pctl < 0.3:
            vol_label = "LOW VOL"
        else:
            vol_label = "NORMAL VOL"
        print(f"  | VOL FORECAST: {vol_label}  "
              f"(est. {vol_f:.1%} weekly, {vol_pctl:.0%} percentile)")

    # 4. KEY LEVELS
    if date in levels.index:
        lv = levels.loc[date]
        print(f"  | KEY LEVELS:")
        print(f"  |   Prior Week High:  {lv['pw_high']:.2f}")
        print(f"  |   Prior Week Low:   {lv['pw_low']:.2f}")
        print(f"  |   Prior Week Close: {lv['pw_close']:.2f}")
        print(f"  |   Prior Week Mid:   {lv['pw_mid']:.2f}")
        print(f"  |   4-Week High:      {lv['4w_high']:.2f}")
        print(f"  |   4-Week Low:       {lv['4w_low']:.2f}")
        print(f"  |   ATR (5d):         {lv['atr_5d']:.2f} pts")
        print(f"  |   ATR (20d):        {lv['atr_20d']:.2f} pts")

    # 5. INTRADAY STRUCTURE
    if date in levels.index:
        t4h = lv.get("trend_4h", 0)
        m1h = lv.get("momentum_1h", 0)
        trend_label = "UPTREND" if t4h > 2 else "DOWNTREND" if t4h < -2 else "CHOPPY"
        mom_label   = "BULLISH" if m1h > 0.0002 else "BEARISH" if m1h < -0.0002 else "FLAT"
        print(f"  | INTRADAY:")
        print(f"  |   4H Trend:         {trend_label} (score: {t4h:+.0f})")
        print(f"  |   1H Momentum:      {mom_label} ({m1h:+.6f})")

    # 6. CONFLUENCE
    signals = []
    if regime in ["RISK-ON", "LEAN RISK-ON"]:
        signals.append(1)
    elif regime in ["RISK-OFF", "LEAN RISK-OFF"]:
        signals.append(-1)
    else:
        signals.append(0)

    if prob > 0.55:
        signals.append(1)
    elif prob < 0.45:
        signals.append(-1)
    else:
        signals.append(0)

    if date in levels.index:
        if t4h > 2:
            signals.append(1)
        elif t4h < -2:
            signals.append(-1)
        else:
            signals.append(0)

    bull_count = signals.count(1)
    bear_count = signals.count(-1)
    if bull_count >= 2 and bear_count == 0:
        confluence = "STRONG BULL"
    elif bear_count >= 2 and bull_count == 0:
        confluence = "STRONG BEAR"
    elif bull_count > bear_count:
        confluence = "LEAN BULL"
    elif bear_count > bull_count:
        confluence = "LEAN BEAR"
    else:
        confluence = "MIXED — REDUCE SIZE"

    # Actual outcome
    actual_label = df.loc[date, "label"] if date in df.index else "?"
    actual_name  = {1: "BULL OK", -1: "BEAR OK", 0: "FLAT"}.get(actual_label, "?")

    print(f"  |")
    print(f"  | >CONFLUENCE:   {confluence}")
    print(f"  | >ACTUAL:       {actual_name}")
    print(f"  +{'-'*55}")

# ══════════════════════════════════════════════════════════════
# SAVE EVERYTHING
# ══════════════════════════════════════════════════════════════
print("\nSaving models and data...")
for fname, obj in [
    ("price_model.pkl",    price_model),
    ("vol_model.pkl",      vol_reg),
    ("price_features.pkl", PRICE_FEATURES),
    ("all_features.pkl",   ALL_FEATURES),
]:
    with open(os.path.join(MODEL_DIR, fname), "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: models/{fname}")

# Save the full enriched dataset
df.to_csv(os.path.join(PROC_DIR, "model_dataset_enriched.csv"))
print(f"  Saved: data/processed/model_dataset_enriched.csv")

print("\nAll done.")