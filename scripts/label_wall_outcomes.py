"""
Label GEX wall outcomes for each trading day.

For each day in gex_profiles/:
  1. Load the GEX profile (top walls by gex_norm)
  2. Fetch QQQ daily OHLC + intraday (5-min where available, else daily)
  3. For each wall: classify as HELD / BROKE / NOT_REACHED
  4. Write one row per wall to data/processed/wall_outcomes.csv

Wall outcome definitions:
  NOT_REACHED  — price never came within REACH_PCT of the wall
  HELD         — price reached within REACH_PCT AND reversed (close moved away)
  BROKE        — price reached within REACH_PCT AND crossed the wall level

Usage:
  python scripts/label_wall_outcomes.py
  python scripts/label_wall_outcomes.py --top-n 5   # walls per day to label
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROFILES_DIR = Path(__file__).parent.parent / "data" / "processed" / "gex_profiles"
OUT_PATH     = Path(__file__).parent.parent / "data" / "processed" / "wall_outcomes.csv"

REACH_PCT    = 0.005   # wall is "reached" if price comes within 0.5% of the strike
MIN_NORM     = 20.0    # only label walls with gex_norm >= this
TOP_N        = 10      # top walls per day to label (by gex_norm)
UNDERLYING   = "QQQ"


def classify_wall(strike: float, gex: float, open_: float,
                  high: float, low: float, close: float) -> str:
    reach_band = strike * REACH_PCT

    above_spot = strike > open_
    reached = (high >= strike - reach_band) if above_spot else (low <= strike + reach_band)

    if not reached:
        return "NOT_REACHED"

    # Broke: price crossed to the other side of the wall
    if above_spot:
        broke = close > strike + reach_band
    else:
        broke = close < strike - reach_band

    return "BROKE" if broke else "HELD"


def fetch_daily_ohlc(start: date, end: date) -> pd.DataFrame:
    """Fetch QQQ daily OHLC for a date range. Returns df with date index."""
    hist = yf.download(
        UNDERLYING,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if hist.empty:
        return hist

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    hist.index = pd.to_datetime(hist.index).date
    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    profile_files = sorted(PROFILES_DIR.glob("gex_profile_*.csv"))
    if not profile_files:
        print("No profile files found.")
        return

    # Collect all dates and fetch price data in one bulk call
    dates = []
    for f in profile_files:
        stem = f.stem  # gex_profile_YYYYMMDD
        d = date(int(stem[-8:-4]), int(stem[-4:-2]), int(stem[-2:]))
        dates.append(d)

    min_date = min(dates)
    max_date = max(dates) + timedelta(days=1)
    print(f"Fetching QQQ daily OHLC {min_date} to {max_date} ...")
    price_df = fetch_daily_ohlc(min_date, max_date)
    if price_df.empty:
        print("ERROR: no price data returned")
        return
    print(f"  Got {len(price_df)} days of OHLC")

    rows = []
    for f, d in zip(profile_files, dates):
        if d not in price_df.index:
            print(f"  [skip] {d} not in price data")
            continue

        profile = pd.read_csv(f)
        profile = profile[profile["gex_norm"] >= MIN_NORM].copy()
        if profile.empty:
            continue

        top_walls = profile.nlargest(args.top_n, "gex_norm")

        row_price = price_df.loc[d]
        open_  = float(row_price["Open"])
        high   = float(row_price["High"])
        low    = float(row_price["Low"])
        close  = float(row_price["Close"])
        spot   = float(profile["spot"].iloc[0])

        for _, wall in top_walls.iterrows():
            outcome = classify_wall(
                float(wall["strike"]), float(wall["gex"]),
                open_, high, low, close,
            )
            rows.append({
                "date":        d,
                "strike":      wall["strike"],
                "gex":         wall["gex"],
                "gex_norm":    wall["gex_norm"],
                "vex":         wall.get("vex", np.nan),
                "oi":          wall.get("oi", np.nan),
                "iv_mean":     wall.get("iv_mean", np.nan),
                "dist_pct":    wall["dist_pct"],
                "gex_sign":    "PUT_WALL" if wall["gex"] < 0 else "CALL_WALL",
                "spot":        spot,
                "open":        open_,
                "high":        high,
                "low":         low,
                "close":       close,
                "outcome":     outcome,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out_df)} wall observations -> {OUT_PATH}")

    # Quick summary
    print("\nOutcome distribution:")
    print(out_df["outcome"].value_counts().to_string())
    print("\nHeld rate by gex_norm decile:")
    out_df["norm_decile"] = pd.qcut(out_df["gex_norm"], 5, labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    reached = out_df[out_df["outcome"] != "NOT_REACHED"].copy()
    if not reached.empty:
        summary = (
            reached.groupby("norm_decile", observed=True)["outcome"]
            .apply(lambda s: (s == "HELD").mean())
            .rename("held_rate")
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
