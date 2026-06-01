"""
Decode SPY EOD options chain data into daily GEX profiles.

Reads spy_2020_2022.csv (1.28 GB, Jan 2020-Dec 2022) in chunks and produces
one CSV per trading day, in the same format as gex_profiles_0dte/ so that
label_spy_touches.py can consume them.

Key differences from OPRA pipeline:
  - Source: EOD chain snapshot (QUOTE_TIME_HOURS == 16.0), not intraday CBBO
  - Greeks already provided (C_GAMMA, C_VEGA, C_THETA, etc.) — no IV solve needed
  - Volume used as OI proxy (same spirit as OPRA bid_sz + ask_sz proxy)
  - Strikes converted to ES futures using SPY/ES ratio from ES 1m bars
  - DTE == 0 preferred; DTE == 1 fallback when < 5 strikes available
  - T_hours = 6.5 (standard RTH) for the profile — EOD so no intraday decomp

Output:
  data/processed/gex_profiles_spy/gex_profile_spy_YYYYMMDD.csv

Columns (same schema as gex_profiles_0dte):
  strike, gex, vex, charmex, oi, gex_norm, vex_norm, charmex_norm, oi_norm,
  confluence, spot, dist_pct, nq_qqq_ratio, strike_futures, iv_mean, date, T_hours

Usage:
  python scripts/decode_spy_eod.py
  python scripts/decode_spy_eod.py --input "archive (2)/spy_2020_2022.csv"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT         = Path(__file__).parent.parent
DEFAULT_INPUT = ROOT / "archive (2)" / "spy_2020_2022.csv"
ES_1M_PATH   = ROOT / "data" / "raw" / "ES" / "1Min_ES.csv"
OUT_DIR      = ROOT / "data" / "processed" / "gex_profiles_spy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MULT                 = 100       # options contract multiplier (100 shares)
CONFLUENCE_THRESHOLD = 40.0      # same as OPRA pipeline
FILTER_PCT           = 0.10      # ±10% of spot — same as OPRA pipeline
MIN_STRIKES          = 5         # min strikes for DTE==0; else fall back to DTE==1
RTH_OPEN_HOUR        = 9
RTH_OPEN_MINUTE      = 31        # use 9:31 bar as "open" price for ratio
CHUNK_SIZE           = 500_000   # rows per chunk when reading the 1.28 GB file
PROGRESS_EVERY       = 50        # print progress every N days


# ── ES 1-minute loader ────────────────────────────────────────────────────────

def load_es_1m(path: Path) -> pd.DataFrame:
    """
    Load ES 1m bars from the European-formatted semicolon-separated CSV.

    Format:
      Date;Symbol;Open;High;Low;Close;Volume
      1/2/2014 5:00 AM;ESH14;1.837,75;1.838,00;...

    European numbers: period = thousands separator, comma = decimal.
    First row may be junk — detect and skip if the Date field is not parseable.
    """
    print(f"Loading ES 1m bars from {path} ...")

    # Read raw to check first row
    raw_head = pd.read_csv(path, sep=";", nrows=2, header=0, dtype=str)
    first_date_val = raw_head.iloc[0]["Date"] if "Date" in raw_head.columns else ""

    # If the first data row can't be parsed as a date it's junk — skip it
    skip = 0
    try:
        pd.to_datetime(first_date_val, format="%m/%d/%Y %I:%M %p")
    except Exception:
        skip = 1
        print(f"  First row appears to be junk ('{first_date_val}') — skipping")

    es = pd.read_csv(
        path,
        sep=";",
        skiprows=range(1, 1 + skip),   # skip header row 0 is kept; skip data rows after it
        thousands=".",
        decimal=",",
        parse_dates=False,             # we'll parse manually for speed
        dtype={"Symbol": str},
    )

    # Some files have no junk row — if column names are wrong, re-read without skiprows
    if "Date" not in es.columns:
        es = pd.read_csv(
            path, sep=";", thousands=".", decimal=",",
            parse_dates=False, dtype={"Symbol": str},
        )

    # Parse Date column: "1/2/2014 5:00 AM"
    es["dt"] = pd.to_datetime(es["Date"], format="%m/%d/%Y %I:%M %p", errors="coerce")
    es = es.dropna(subset=["dt"])
    es["date_only"] = es["dt"].dt.date

    print(f"  Loaded {len(es):,} ES 1m bars  "
          f"({es['date_only'].min()} to {es['date_only'].max()})")
    return es[["dt", "date_only", "Open", "High", "Low", "Close"]].copy()


def get_es_open(es_df: pd.DataFrame, trade_date) -> float:
    """
    Return the ES price at 9:31 AM on trade_date.
    Falls back to 9:30, 9:32, then earliest RTH bar if those are missing.
    Returns NaN if no RTH bar found.
    """
    day = es_df[es_df["date_only"] == trade_date]
    if day.empty:
        return float("nan")

    for h, m in [(9, 31), (9, 30), (9, 32), (9, 33)]:
        bar = day[(day["dt"].dt.hour == h) & (day["dt"].dt.minute == m)]
        if not bar.empty:
            return float(bar["Open"].iloc[0])

    # Fallback: first bar between 9:30 and 10:00
    rth = day[(day["dt"].dt.hour == 9) & (day["dt"].dt.minute >= 30)]
    if not rth.empty:
        return float(rth["Open"].iloc[0])

    return float("nan")


# ── Per-day GEX profile computation ──────────────────────────────────────────

def normalize_col(series: pd.Series) -> pd.Series:
    """Min-max normalize abs values to 0-100."""
    abs_s = series.abs()
    mn, mx = abs_s.min(), abs_s.max()
    if mx > mn:
        return (abs_s - mn) / (mx - mn) * 100.0
    return pd.Series(0.0, index=series.index)


def compute_spy_gex_profile(day_df: pd.DataFrame, spot_spy: float,
                             spy_es_ratio: float, trade_date) -> pd.DataFrame:
    """
    Build a per-strike GEX profile from one day's EOD chain snapshot.

    day_df has already been filtered to the correct DTE and to the ±FILTER_PCT
    window around spot_spy.  Returns a DataFrame with the standard profile schema.
    """
    if day_df.empty:
        return pd.DataFrame()

    # Aggregate call and put greeks per strike
    # GEX = gamma × volume × multiplier; calls positive, puts negative
    # VEX = vega × volume × multiplier; same sign convention as GEX (magnitude)
    # CharmEX = theta × volume × multiplier (theta as charm proxy)
    # OI proxy = (C_VOLUME + P_VOLUME) / 2

    # Coerce all numeric Greek columns to float (may arrive as strings from CSV)
    for col in ["STRIKE", "C_GAMMA", "C_VEGA", "C_THETA", "C_VOLUME",
                "P_GAMMA", "P_VEGA", "P_THETA", "P_VOLUME", "C_IV", "P_IV"]:
        if col in day_df.columns:
            day_df[col] = pd.to_numeric(day_df[col], errors="coerce").fillna(0.0)

    strikes = day_df.groupby("STRIKE").agg(
        c_gamma  = ("C_GAMMA",  "sum"),
        c_vega   = ("C_VEGA",   "sum"),
        c_theta  = ("C_THETA",  "sum"),
        c_vol    = ("C_VOLUME", "sum"),
        p_gamma  = ("P_GAMMA",  "sum"),
        p_vega   = ("P_VEGA",   "sum"),
        p_theta  = ("P_THETA",  "sum"),
        p_vol    = ("P_VOLUME", "sum"),
        c_iv     = ("C_IV",     "mean"),
        p_iv     = ("P_IV",     "mean"),
    ).reset_index()

    # Net GEX: call contribution positive, put contribution negative
    strikes["gex"]     = (strikes["c_gamma"] * strikes["c_vol"] * MULT
                          - strikes["p_gamma"] * strikes["p_vol"] * MULT)

    # VEX: sum of absolute vega exposure across calls and puts
    # Sign: positive at call wall (gex>0), negative at put wall (gex<0)
    strikes["vex_raw"] = (strikes["c_vega"] * strikes["c_vol"] * MULT
                          + strikes["p_vega"] * strikes["p_vol"] * MULT)
    strikes["vex"]     = np.where(strikes["gex"] >= 0,
                                   strikes["vex_raw"], -strikes["vex_raw"])

    # CharmEX: theta × volume × multiplier (theta proxy for charm)
    strikes["charmex_raw"] = (strikes["c_theta"] * strikes["c_vol"] * MULT
                               + strikes["p_theta"] * strikes["p_vol"] * MULT)
    strikes["charmex"]     = np.where(strikes["gex"] >= 0,
                                       strikes["charmex_raw"], -strikes["charmex_raw"])

    # OI proxy
    strikes["oi"] = (strikes["c_vol"] + strikes["p_vol"]) / 2.0

    # IV mean (average of call and put IV per strike)
    strikes["iv_mean"] = (strikes["c_iv"].fillna(0) + strikes["p_iv"].fillna(0)) / 2.0

    # Drop strikes with zero volume (no information)
    strikes = strikes[strikes["oi"] > 0].copy()
    if strikes.empty:
        return pd.DataFrame()

    # Normalize to 0-100 (magnitude rank within day)
    strikes["gex_norm"]     = normalize_col(strikes["gex"])
    strikes["vex_norm"]     = normalize_col(strikes["vex"])
    strikes["charmex_norm"] = normalize_col(strikes["charmex"])
    strikes["oi_norm"]      = normalize_col(strikes["oi"])

    # Confluence: all three greek norms >= threshold at same strike
    strikes["confluence"] = (
        (strikes["gex_norm"]     >= CONFLUENCE_THRESHOLD) &
        (strikes["vex_norm"]     >= CONFLUENCE_THRESHOLD) &
        (strikes["charmex_norm"] >= CONFLUENCE_THRESHOLD)
    ).astype(int)

    # Spot and distance
    strikes["spot"]     = round(spot_spy, 4)
    strikes["dist_pct"] = (strikes["STRIKE"] - spot_spy) / spot_spy * 100.0

    # Convert to ES futures equivalent
    strikes["nq_qqq_ratio"]   = round(spy_es_ratio, 4)   # column name kept for schema compat
    strikes["strike_futures"]  = (strikes["STRIKE"] * spy_es_ratio).round(1)

    # Metadata
    strikes["date"]    = str(trade_date)
    strikes["T_hours"] = 6.5   # standard RTH hours (EOD snapshot = next-day opening walls)

    # Rename STRIKE → strike for schema compatibility
    strikes = strikes.rename(columns={"STRIKE": "strike"})

    # Select and order columns to match gex_profiles_0dte schema
    out_cols = [
        "strike", "gex", "vex", "charmex", "oi",
        "gex_norm", "vex_norm", "charmex_norm", "oi_norm",
        "confluence", "spot", "dist_pct", "nq_qqq_ratio",
        "strike_futures", "iv_mean", "date", "T_hours",
    ]
    return strikes[out_cols].sort_values("strike").reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Decode SPY EOD options → GEX profiles")
    parser.add_argument(
        "--input", default=str(DEFAULT_INPUT),
        help="Path to spy_2020_2022.csv (default: archive (2)/spy_2020_2022.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    # ── 1. Load ES 1m for SPY/ES ratio computation ───────────────────────────
    es_df = load_es_1m(ES_1M_PATH)

    # ── 2. Read SPY options CSV in chunks and bucket by date ─────────────────
    print(f"\nReading SPY options data from {input_path} ...")
    print(f"  File size: {input_path.stat().st_size / 1e9:.2f} GB")

    # Accumulate rows per date across chunks
    date_buckets: dict[str, list] = {}

    dtype_map = {
        "QUOTE_DATE":         str,
        "QUOTE_TIME_HOURS":   float,
        "UNDERLYING_LAST":    float,
        "EXPIRE_DATE":        str,
        "DTE":                float,
        "C_GAMMA":            float,
        "C_VEGA":             float,
        "C_DELTA":            float,
        "C_THETA":            float,
        "C_IV":               float,
        "C_VOLUME":           float,
        "C_BID":              float,
        "C_ASK":              float,
        "STRIKE":             float,
        "P_BID":              float,
        "P_ASK":              float,
        "P_VOLUME":           float,
        "P_GAMMA":            float,
        "P_VEGA":             float,
        "P_DELTA":            float,
        "P_THETA":            float,
        "P_IV":               float,
        "STRIKE_DISTANCE":    float,
        "STRIKE_DISTANCE_PCT":float,
    }

    total_rows_read = 0
    chunk_count = 0

    reader = pd.read_csv(
        input_path,
        dtype=dtype_map,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines="skip",
    )

    for chunk in reader:
        chunk_count += 1
        total_rows_read += len(chunk)

        # Strip square brackets from column names (file has [QUOTE_TIME_HOURS] etc.)
        chunk.columns = [c.strip().strip("[]").strip() for c in chunk.columns]

        # Coerce key filter + numeric columns from string to float/numeric
        for col in ["QUOTE_TIME_HOURS", "UNDERLYING_LAST", "DTE", "STRIKE",
                    "C_GAMMA", "C_VEGA", "C_THETA", "C_VOLUME", "C_IV",
                    "P_GAMMA", "P_VEGA", "P_THETA", "P_VOLUME", "P_IV"]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # Keep only EOD snapshots (QUOTE_TIME_HOURS == 16.0)
        chunk = chunk[chunk["QUOTE_TIME_HOURS"] == 16.0].copy()
        if chunk.empty:
            continue

        # Fill NaN volumes with 0 (some strikes may have no trades)
        for vol_col in ["C_VOLUME", "P_VOLUME"]:
            chunk[vol_col] = chunk[vol_col].fillna(0.0)

        # Group into per-date buckets
        for qdate, grp in chunk.groupby("QUOTE_DATE"):
            key = str(qdate)
            if key not in date_buckets:
                date_buckets[key] = []
            date_buckets[key].append(grp)

        if chunk_count % 10 == 0:
            print(f"  Read {total_rows_read:,} rows so far "
                  f"({chunk_count} chunks, {len(date_buckets)} unique dates)")

    print(f"\nFinished reading: {total_rows_read:,} total rows, "
          f"{len(date_buckets)} trading days")

    # ── 3. Process each day ───────────────────────────────────────────────────
    dates_sorted = sorted(date_buckets.keys())
    print(f"\nProcessing {len(dates_sorted)} trading days ...")

    saved = 0
    skipped_no_es = 0
    skipped_no_strikes = 0
    skipped_exists = 0

    for i, date_str in enumerate(dates_sorted):
        # Parse trade_date (QUOTE_DATE format: YYYY-MM-DD or M/D/YYYY)
        try:
            trade_date = pd.to_datetime(date_str).date()
        except Exception:
            print(f"  [skip] Cannot parse date: {date_str!r}")
            continue

        # Skip if output already exists
        out_path = OUT_DIR / f"gex_profile_spy_{trade_date.strftime('%Y%m%d')}.csv"
        if out_path.exists():
            skipped_exists += 1
            continue

        # Concatenate all chunks for this date
        day_df = pd.concat(date_buckets[date_str], ignore_index=True)

        # spot price: use UNDERLYING_LAST (EOD price for SPY)
        spot_spy = float(day_df["UNDERLYING_LAST"].dropna().iloc[-1]) if not day_df["UNDERLYING_LAST"].dropna().empty else None
        if spot_spy is None or spot_spy <= 0:
            skipped_no_es += 1
            continue

        # ES 9:31 AM open on the SAME day (EOD profile → used for NEXT day, but ratio
        # is computed from same-day open for calibration; label_spy_touches uses it for D+1)
        es_open = get_es_open(es_df, trade_date)
        if np.isnan(es_open) or es_open <= 0:
            # Fallback: try previous trading day
            prev_idx = i - 1
            while prev_idx >= 0:
                try:
                    prev_date = pd.to_datetime(dates_sorted[prev_idx]).date()
                    es_open = get_es_open(es_df, prev_date)
                    if not np.isnan(es_open) and es_open > 0:
                        break
                except Exception:
                    pass
                prev_idx -= 1

        if np.isnan(es_open) or es_open <= 0:
            print(f"  [skip {trade_date}] No ES 1m bar found for ratio")
            skipped_no_es += 1
            continue

        spy_es_ratio = es_open / spot_spy

        # Select DTE==0; fallback to DTE==1 if fewer than MIN_STRIKES
        dte0 = day_df[day_df["DTE"] == 0].copy()
        if len(dte0["STRIKE"].unique()) < MIN_STRIKES:
            dte1 = day_df[day_df["DTE"] == 1].copy()
            if len(dte1["STRIKE"].unique()) >= MIN_STRIKES:
                active = dte1
                dte_used = 1
            else:
                skipped_no_strikes += 1
                continue
        else:
            active = dte0
            dte_used = 0

        # Filter to ±FILTER_PCT of spot
        lo = spot_spy * (1 - FILTER_PCT)
        hi = spot_spy * (1 + FILTER_PCT)
        active = active[(active["STRIKE"] >= lo) & (active["STRIKE"] <= hi)].copy()

        if active.empty or len(active["STRIKE"].unique()) < MIN_STRIKES:
            skipped_no_strikes += 1
            continue

        # Build profile
        profile = compute_spy_gex_profile(active, spot_spy, spy_es_ratio, trade_date)

        if profile.empty:
            skipped_no_strikes += 1
            continue

        profile.to_csv(out_path, index=False)
        saved += 1

        if (i + 1) % PROGRESS_EVERY == 0:
            print(f"  [{i+1}/{len(dates_sorted)}] {trade_date}  "
                  f"saved={saved}  skipped_no_es={skipped_no_es}  "
                  f"skipped_strikes={skipped_no_strikes}  exists={skipped_exists}  "
                  f"dte_used={dte_used}  ratio={spy_es_ratio:.4f}")

    print()
    print("=" * 60)
    print("  SPY EOD DECODE COMPLETE")
    print("=" * 60)
    print(f"  Trading days found:   {len(dates_sorted)}")
    print(f"  Profiles saved:       {saved}")
    print(f"  Skipped (no ES bar):  {skipped_no_es}")
    print(f"  Skipped (< strikes):  {skipped_no_strikes}")
    print(f"  Skipped (exists):     {skipped_exists}")
    print(f"  Output dir:           {OUT_DIR}")


if __name__ == "__main__":
    main()
