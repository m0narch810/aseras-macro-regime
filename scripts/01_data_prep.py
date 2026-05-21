import pandas as pd
import os

# ── CONFIG ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR    = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "NQ": {
        "daily": "NQ/D_NQ.csv",
        "4h":    "NQ/4H_NQ.csv",
        "1h":    "NQ/1H_NQ.csv",
        "15m":   "NQ/15Min_NQ.csv",
        "5m":    "NQ/NQ_5Min.csv",
        "1m":    "NQ/1Min_NQ.csv",
    },
    "ES": {
        "daily": "ES/D_ES.csv",
        "4h":    "ES/4H_ES.csv",
        "1h":    "ES/1H_ES.csv",
        "15m":   "ES/15Min_ES.csv",
        "5m":    "ES/5Min_ES.csv",
        "1m":    "ES/1Min_ES.csv",
    },
}

# ── PARSE FUNCTION ────────────────────────────────────────────
# What this does:
# - Skips the first row (the "Time Series;NQH26" junk)
# - Uses semicolons as separator
# - Converts European numbers: removes "." thousands sep, swaps "," for "."
def load_csv(filepath):
    df = pd.read_csv(
        filepath,
        sep=";",
        skiprows=1,
        thousands=".",
        decimal=",",
        low_memory=False
    )
    df.columns = ["date", "symbol", "open", "high", "low", "close", "volume"]

    # Parse dates — handles both "1/30/2026" and "1/30/2026 3:00 PM"
    df["date"] = pd.to_datetime(df["date"])

    # Force numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    return df

# ── BACK-ADJUSTMENT ───────────────────────────────────────────
# What this does:
# - Finds every point where the contract symbol changes
# - At that point, calculates the gap between the old contract's
#   last close and the new contract's first open
# - Shifts all prices BEFORE that point by that gap
# - Result: one smooth continuous price series with no fake jumps
def back_adjust(df):
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Find rollover points
    rollovers = df[df["symbol"] != df["symbol"].shift(1)].index.tolist()
    if 0 in rollovers:
        rollovers.remove(0)

    # Apply adjustment backwards from each rollover
    for idx in reversed(rollovers):
        prev_close = df.loc[idx - 1, "close"]
        new_open   = df.loc[idx, "open"]
        gap        = new_open - prev_close

        price_cols = ["open", "high", "low", "close"]
        df.loc[:idx-1, price_cols] += gap

    return df

# ── QUALITY CHECKS ────────────────────────────────────────────
# What this does:
# - Reports how many rows have issues without deleting anything silently
# - Removes duplicates
# - Flags and removes bars where high < low (corrupted)
# - Reports any NaN values
def quality_check(df, name):
    issues = 0

    dupes = df.duplicated(subset=["date"]).sum()
    if dupes > 0:
        print(f"  [{name}] Removing {dupes} duplicate timestamps")
        df = df.drop_duplicates(subset=["date"])
        issues += dupes

    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        print(f"  [{name}] Removing {bad_hl} bars where high < low")
        df = df[df["high"] >= df["low"]]
        issues += bad_hl

    nans = df[["open","high","low","close","volume"]].isna().sum().sum()
    if nans > 0:
        print(f"  [{name}] Found {nans} NaN values in OHLCV — dropping those rows")
        df = df.dropna(subset=["open","high","low","close"])
        issues += nans

    if issues == 0:
        print(f"  [{name}] Clean — no issues found")

    return df.reset_index(drop=True)

# ── MAIN LOOP ─────────────────────────────────────────────────
for instrument, timeframes in FILES.items():
    print(f"\n{'='*40}")
    print(f"Processing {instrument}")
    print(f"{'='*40}")

    for tf, rel_path in timeframes.items():
        full_path = os.path.join(RAW_DIR, rel_path)
        print(f"\n  Timeframe: {tf}")

        df = load_csv(full_path)
        print(f"  Loaded {len(df):,} rows | {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"  Contracts found: {df['symbol'].unique()}")

        df = back_adjust(df)
        print(f"  Back-adjustment complete — {df['symbol'].nunique()} contracts merged")

        df = quality_check(df, f"{instrument}_{tf}")

        out_path = os.path.join(OUT_DIR, f"{instrument}_{tf}_clean.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

print("\n\nAll done. Check data/processed/ for output files.")