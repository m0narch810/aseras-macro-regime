import os
import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# ── LOAD DATA ─────────────────────────────────────────────────
print("Loading data...")
nq   = pd.read_csv(os.path.join(PROC_DIR, "NQ_daily_clean.csv"),
                   parse_dates=["date"]).set_index("date")
es   = pd.read_csv(os.path.join(PROC_DIR, "ES_daily_clean.csv"),
                   parse_dates=["date"]).set_index("date")
nq1h = pd.read_csv(os.path.join(PROC_DIR, "NQ_1h_clean.csv"),
                   parse_dates=["date"]).set_index("date")
macro = pd.read_csv(os.path.join(PROC_DIR, "macro_weekly.csv"),
                    index_col=0, parse_dates=True)

print(f"  NQ daily:  {len(nq)} rows")
print(f"  ES daily:  {len(es)} rows")
print(f"  NQ 1h:     {len(nq1h)} rows")
print(f"  Macro:     {len(macro)} weeks")

# ── WEEKLY AGGREGATION ────────────────────────────────────────
# What this does:
# Resample daily bars into weekly bars (week ending Thursday
# to match the macro data). Each week gets: open, high, low,
# close, volume, and realized volatility.

def make_weekly_bars(df, name):
    weekly = df["close"].resample("W-THU").agg(
        weekly_open  = "first",
        weekly_high  = "max",
        weekly_low   = "min",
        weekly_close = "last",
        weekly_vol   = "sum"
    )
    # Realized vol = std of daily returns within the week
    daily_ret = df["close"].pct_change()
    weekly["realized_vol"] = daily_ret.resample("W-THU").std() * np.sqrt(5)
    print(f"  {name} weekly bars: {len(weekly)}")
    return weekly

print("\nBuilding weekly bars...")
nq_w = make_weekly_bars(nq, "NQ")
es_w = make_weekly_bars(es, "ES")

# ── BUILD LABELS ──────────────────────────────────────────────
# What this does:
# For each week T, we look at what happens in week T+1.
# Label 1 (direction): did NQ go up or down next week?
#   +1 = bullish (close higher than open)
#   -1 = bearish (close lower than open)
#    0 = neutral (move < 0.3%, too small to trade)
# Label 2 (volatility): realized vol of next week
# Label 3 (magnitude): point move of next week

nq_w["fwd_return"]  = nq_w["weekly_close"].pct_change().shift(-1)
nq_w["fwd_vol"]     = nq_w["realized_vol"].shift(-1)
nq_w["fwd_points"]  = (nq_w["weekly_close"].shift(-1)
                       - nq_w["weekly_close"])

NEUTRAL_THRESHOLD = 0.003  # 0.3% — moves smaller than this = neutral

def label_direction(ret):
    if pd.isna(ret):
        return np.nan
    if ret >  NEUTRAL_THRESHOLD:
        return 1    # bullish
    elif ret < -NEUTRAL_THRESHOLD:
        return -1   # bearish
    else:
        return 0    # neutral

nq_w["label"] = nq_w["fwd_return"].apply(label_direction)

# ── PRICE FEATURES ────────────────────────────────────────────
# What this does:
# Builds features FROM the current week's price action.
# These tell the model things like: is momentum up or down,
# how volatile has it been, where did price close in the range.
# All computed at week T, used to predict week T+1.

# 1. Weekly return (current week momentum)
nq_w["nq_week_ret"]    = nq_w["weekly_close"].pct_change()

# 2. Where did price close within the week's range (0=low, 1=high)
nq_w["nq_close_pos"]   = ((nq_w["weekly_close"] - nq_w["weekly_low"]) /
                           (nq_w["weekly_high"]  - nq_w["weekly_low"] + 1e-9))

# 3. ATR proxy — weekly high-low range as % of close
nq_w["nq_week_range"]  = ((nq_w["weekly_high"] - nq_w["weekly_low"]) /
                            nq_w["weekly_close"])

# 4. 4-week and 12-week momentum
nq_w["nq_mom_4w"]      = nq_w["weekly_close"].pct_change(4)
nq_w["nq_mom_12w"]     = nq_w["weekly_close"].pct_change(12)

# 5. Vol regime — is current vol high or low vs recent average
nq_w["nq_vol_regime"]  = (nq_w["realized_vol"] /
                           nq_w["realized_vol"].rolling(12).mean())

# 6. ES-NQ spread (ES and NQ diverging can signal reversals)
es_w["es_week_ret"]    = es_w["weekly_close"].pct_change()
nq_w["es_nq_spread"]   = nq_w["nq_week_ret"] - es_w["es_week_ret"]

# ── 1H FEATURES ───────────────────────────────────────────────
# What this does:
# Looks at the last 5 trading days of 1h bars to compute
# intraday momentum and structure going into the next week

print("Building 1h features...")
nq1h["ret_1h"] = nq1h["close"].pct_change()

# Average hourly return over the week (directional drift)
nq1h_weekly = nq1h["ret_1h"].resample("W-THU").mean().rename("nq_1h_drift")

# Intraday vol (std of hourly returns)
nq1h_vol = nq1h["ret_1h"].resample("W-THU").std().rename("nq_1h_vol")

# ── MERGE EVERYTHING ──────────────────────────────────────────
print("Merging features with macro data...")

price_features = nq_w[[
    "nq_week_ret", "nq_close_pos", "nq_week_range",
    "nq_mom_4w", "nq_mom_12w", "nq_vol_regime",
    "es_nq_spread", "realized_vol",
    "fwd_return", "fwd_vol", "fwd_points", "label"
]].copy()

price_features = price_features.join(nq1h_weekly, how="left")
price_features = price_features.join(nq1h_vol,    how="left")

# Merge with macro — align on Thursday week-end date
dataset = macro.join(price_features, how="inner")
dataset = dataset.dropna(subset=["label"])

print(f"\nFinal dataset shape: {dataset.shape}")
print(f"Date range: {dataset.index.min().date()} to {dataset.index.max().date()}")
print(f"\nLabel distribution:")
print(dataset["label"].value_counts().sort_index().rename(
    {-1: "Bearish", 0: "Neutral", 1: "Bullish"}))

# ── RETURN DISTRIBUTION ANALYSIS ──────────────────────────────
print("\n" + "="*60)
print("  RETURN DISTRIBUTION ANALYSIS")
print("="*60)
returns = dataset["fwd_return"].dropna()
print(f"\nWeekly NQ fwd return stats  ({len(returns)} weeks):")
print(f"  Mean:   {returns.mean():+.3%}   Median: {returns.median():+.3%}")
print(f"  Std:    {returns.std():.3%}   Skew:   {returns.skew():+.3f}")
print(f"  Min:    {returns.min():+.3%}   Max:    {returns.max():+.3%}")
print(f"\n  {'Threshold':<11} {'Bear':>6} {'Neutral':>8} {'Bull':>6}  {'Neutral%':>9}")
print(f"  {'-'*46}")
for thresh in [0.003, 0.005, 0.007]:
    bear = (returns < -thresh).sum()
    bull = (returns >  thresh).sum()
    neut = len(returns) - bear - bull
    print(f"  {thresh:.1%}       {bear:>6} {neut:>8} {bull:>6}  {neut/len(returns):>8.1%}")

# ── SAVE ──────────────────────────────────────────────────────
out_path = os.path.join(PROC_DIR, "model_dataset.csv")
dataset.to_csv(out_path)
print(f"\nDataset saved: {out_path}")