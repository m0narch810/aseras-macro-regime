"""
Decode one day of OPRA CBBO-1m data into a per-strike GEX profile.

For each day:
  1. Load the .dbn.zst file
  2. Filter to QQQ options at RTH open (9:31 ET)
  3. Parse OCC symbol → expiry, type, strike
  4. Fetch QQQ underlying price and OI from yfinance
  5. Compute BS IV + Greeks per option
  6. Output per-strike GEX/VEX/CharmEX summary

Usage:
  python scripts/decode_opra_day.py --date 2025-02-10
  python scripts/decode_opra_day.py --date 2025-02-10 --debug
"""

import argparse
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import databento as db
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

ET = ZoneInfo("America/New_York")
DATA_DIR = Path(__file__).parent.parent / "dataidk"
OUT_DIR  = Path(__file__).parent.parent / "data" / "processed" / "gex_profiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNDERLYING   = "QQQ"
RISK_FREE    = 0.045
FILTER_PCT   = 0.10
MIN_OI       = 10
FALLBACK_RATIO   = 41.14
CONFLUENCE_THRESHOLD = 40.0

# Intraday snapshot times: every 30 min through RTH
# Last snapshot at 15:30 to allow entry before close
SNAPSHOT_TIMES = [
    (9, 31), (10, 0), (10, 30), (11, 0), (11, 30),
    (12, 0), (12, 30), (13, 0), (13, 30), (14, 0),
    (14, 30), (15, 0), (15, 30),
]

OUT_DIR_0DTE    = Path(__file__).parent.parent / "data" / "processed" / "gex_profiles_0dte"
OUT_DIR_SNAPS   = Path(__file__).parent.parent / "data" / "processed" / "gex_snapshots_0dte"
OUT_DIR_0DTE.mkdir(parents=True, exist_ok=True)
OUT_DIR_SNAPS.mkdir(parents=True, exist_ok=True)


# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def _d1d2(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_price(S, K, T, r, sigma, flag):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if flag == "C" else max(0.0, K - S)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if flag == "C":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(market_price, S, K, T, r, flag):
    if T <= 0 or market_price <= 0:
        return np.nan
    intrinsic = max(0.0, S - K) if flag == "C" else max(0.0, K - S)
    if market_price <= intrinsic * 1.001:
        return np.nan
    try:
        iv = brentq(
            lambda s: bs_price(S, K, T, r, s, flag) - market_price,
            1e-6, 10.0, xtol=1e-6, maxiter=100
        )
        return iv if 0.001 < iv < 9.9 else np.nan
    except (ValueError, RuntimeError):
        return np.nan

def bs_greeks(S, K, T, r, sigma, flag):
    """Return (delta, gamma, vega, charm) given IV."""
    if T <= 0 or np.isnan(sigma) or sigma <= 0:
        return np.nan, np.nan, np.nan, np.nan
    d1, d2 = _d1d2(S, K, T, r, sigma)
    nd1  = norm.pdf(d1)
    gamma = nd1 / (S * sigma * math.sqrt(T))
    vega  = S * nd1 * math.sqrt(T) / 100          # per 1 vol point
    if flag == "C":
        delta = norm.cdf(d1)
        charm = -(nd1 * (2 * r * T - d2 * sigma * math.sqrt(T))
                  / (2 * T * sigma * math.sqrt(T)))
    else:
        delta = norm.cdf(d1) - 1
        charm = -(nd1 * (2 * r * T - d2 * sigma * math.sqrt(T))
                  / (2 * T * sigma * math.sqrt(T)))
    return delta, gamma, vega, charm


# ── Symbol parsing ────────────────────────────────────────────────────────────

def parse_occ(symbol: str):
    """
    OCC format: 'QQQ   250210C00439000'
    Returns (root, expiry_date, flag, strike_float) or None.
    """
    s = symbol.strip()
    if len(s) < 15:
        return None
    root   = s[:6].strip()
    date_s = s[6:12]
    flag   = s[12]
    strike = int(s[13:]) / 1000.0
    try:
        exp = date(2000 + int(date_s[:2]), int(date_s[2:4]), int(date_s[4:6]))
    except ValueError:
        return None
    if flag not in ("C", "P"):
        return None
    return root, exp, flag, strike



# ── Profile computation (shared by single-snapshot and intraday) ─────────────

def _compute_profile(snap, trade_date, spot, T_year, ratio):
    """
    Given a filtered snap dataframe (already: QQQ, 0DTE, mid > 0),
    compute and return the per-strike GEX profile.
    T_year: time-to-expiry in years for this snapshot.
    """
    lo, hi = spot * (1 - FILTER_PCT), spot * (1 + FILTER_PCT)
    snap = snap[(snap["strike"] >= lo) & (snap["strike"] <= hi)].copy()
    if snap.empty:
        return pd.DataFrame()

    snap["T"] = T_year
    snap["iv"] = snap.apply(
        lambda r: implied_vol(r["mid"], spot, r["strike"], r["T"], RISK_FREE, r["flag"]),
        axis=1,
    )
    snap = snap[snap["iv"].notna()].copy()
    if snap.empty:
        return pd.DataFrame()

    greeks = snap.apply(
        lambda r: bs_greeks(spot, r["strike"], r["T"], RISK_FREE, r["iv"], r["flag"]),
        axis=1,
    )
    snap[["delta", "gamma", "vega", "charm"]] = pd.DataFrame(
        greeks.tolist(), index=snap.index
    )

    snap["oi"] = snap["bid_sz_00"] + snap["ask_sz_00"]
    snap = snap[snap["oi"] >= MIN_OI].copy()
    if snap.empty:
        return pd.DataFrame()

    MULT = 100
    snap["gex_contrib"]     = snap["gamma"] * snap["oi"] * MULT * spot
    snap["vex_contrib"]     = snap["vega"]  * snap["oi"] * MULT
    snap["charmex_contrib"] = snap["charm"] * snap["oi"] * MULT
    snap["gex_signed"] = np.where(snap["flag"] == "C",
                                   snap["gex_contrib"], -snap["gex_contrib"])

    profile = (
        snap.groupby("strike")
        .agg(gex=("gex_signed","sum"), vex=("vex_contrib","sum"),
             charmex=("charmex_contrib","sum"), oi=("oi","sum"),
             iv_mean=("iv","mean"))
        .reset_index().sort_values("strike")
    )

    profile["date"]         = trade_date
    profile["spot"]         = spot
    profile["dist_pct"]     = (profile["strike"] - spot) / spot * 100
    profile["nq_qqq_ratio"] = round(ratio, 4)
    profile["strike_futures"] = (profile["strike"] * ratio).round(1)
    profile["T_hours"]      = round(T_year * 365 * 24, 3)

    for col in ["gex", "vex", "charmex", "oi"]:
        abs_col = profile[col].abs()
        mn, mx  = abs_col.min(), abs_col.max()
        profile[f"{col}_norm"] = ((abs_col - mn) / (mx - mn) * 100) if mx > mn else 0.0

    profile["confluence"] = (
        (profile["gex_norm"]     >= CONFLUENCE_THRESHOLD) &
        (profile["vex_norm"]     >= CONFLUENCE_THRESHOLD) &
        (profile["charmex_norm"] >= CONFLUENCE_THRESHOLD)
    ).astype(int)

    return profile


def _load_day_qqq(trade_date: date):
    """Load OPRA file and return pre-parsed QQQ 0DTE dataframe + index."""
    fname = DATA_DIR / f"opra-pillar-{trade_date.strftime('%Y%m%d')}.cbbo-1m.dbn.zst"
    if not fname.exists():
        raise FileNotFoundError(f"No file for {trade_date}")

    store = db.DBNStore.from_file(str(fname))
    df    = store.to_df()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)

    qqq = df[df["symbol"].str.startswith(UNDERLYING)].copy()
    parsed = qqq["symbol"].map(parse_occ)
    qqq    = qqq[parsed.notna()].copy()
    qqq[["root","expiry","flag","strike"]] = pd.DataFrame(
        parsed[parsed.notna()].tolist(), index=qqq[parsed.notna()].index)
    qqq = qqq[(qqq["root"] == UNDERLYING) & (qqq["expiry"] == trade_date)].copy()
    qqq["bid"] = qqq["bid_px_00"].replace(0, np.nan)
    qqq["ask"] = qqq["ask_px_00"].replace(0, np.nan)
    qqq["mid"] = (qqq["bid"] + qqq["ask"]) / 2.0
    qqq = qqq[qqq["mid"] > 0].copy()
    return qqq


def process_day_snapshots(trade_date: date, nq_prices: dict,
                          debug: bool = False) -> list:
    """
    Generate GEX profiles at every SNAPSHOT_TIMES interval for one day.

    nq_prices : dict mapping (hour, minute) -> NQ futures price.
                Used to adjust QQQ spot intraday: spot_t = spot_open * (nq_t/nq_open).
    Returns   : list of (snapshot_dt_str, profile_df) sorted by time.
    """
    # ── 1. Fetch opening QQQ spot (once per day via yfinance) ────────────────
    hist = yf.download(
        UNDERLYING,
        start=trade_date.strftime("%Y-%m-%d"),
        end=(trade_date + timedelta(days=2)).strftime("%Y-%m-%d"),
        interval="1d", auto_adjust=True, progress=False,
    )
    if hist.empty:
        return []
    open_col = hist["Open"]
    spot_open = float(open_col.iloc[0].iloc[0] if hasattr(open_col.iloc[0], "iloc")
                      else open_col.iloc[0])

    nq_open = nq_prices.get((9, 31)) or nq_prices.get((9, 30))
    ratio   = (nq_open / spot_open) if nq_open else FALLBACK_RATIO

    # ── 2. Load + parse OPRA file once ───────────────────────────────────────
    try:
        qqq = _load_day_qqq(trade_date)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return []

    if qqq.empty:
        return []

    market_close = datetime(trade_date.year, trade_date.month, trade_date.day,
                            16, 0, tzinfo=ET)

    # ── 3. Iterate snapshot times ─────────────────────────────────────────────
    results = []
    for h, m in SNAPSHOT_TIMES:
        snap_dt  = datetime(trade_date.year, trade_date.month, trade_date.day,
                            h, m, tzinfo=ET)
        snap_end = snap_dt + timedelta(minutes=1)

        snap = qqq[(qqq.index >= snap_dt) & (qqq.index < snap_end)].copy()
        if snap.empty:                       # try +1 min (occasional data gap)
            snap_dt2 = snap_dt + timedelta(minutes=1)
            snap     = qqq[(qqq.index >= snap_dt2) &
                           (qqq.index < snap_dt2 + timedelta(minutes=1))].copy()
        if snap.empty:
            continue

        # Intraday spot adjustment
        nq_t  = nq_prices.get((h, m))
        spot  = spot_open * (nq_t / nq_open) if (nq_t and nq_open) else spot_open

        # T in years until 4 PM close (min 5 min to avoid degenerate IV)
        T_hours = max((market_close - snap_dt).total_seconds() / 3600, 5 / 60)
        T_year  = T_hours / (365 * 24)

        profile = _compute_profile(snap, trade_date, spot, T_year, ratio)
        if profile.empty:
            continue

        dt_str = snap_dt.strftime("%Y%m%d_%H%M")
        profile["snapshot_dt"] = snap_dt.strftime("%Y-%m-%d %H:%M")
        results.append((dt_str, profile))

    if debug:
        print(f"  [{trade_date}] {len(results)} snapshots generated")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def process_day(trade_date: date, debug: bool = False, nq_open: float = None) -> pd.DataFrame:
    fname = DATA_DIR / f"opra-pillar-{trade_date.strftime('%Y%m%d')}.cbbo-1m.dbn.zst"
    if not fname.exists():
        raise FileNotFoundError(f"No file for {trade_date}: {fname}")

    # ── 1. Load and filter to QQQ at RTH open (9:31–9:32 ET) ────────────────
    print(f"[{trade_date}] Loading {fname.name} …")
    store = db.DBNStore.from_file(str(fname))
    df    = store.to_df()

    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
    open_ts   = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 31, tzinfo=ET)
    open_end  = open_ts + timedelta(minutes=1)

    snap = df[(df.index >= open_ts) & (df.index < open_end)].copy()
    snap = snap[snap["symbol"].str.startswith(UNDERLYING)].copy()

    if snap.empty:
        print(f"  [warn] No QQQ rows at 9:31 ET — trying 9:32")
        open_ts  = open_ts  + timedelta(minutes=1)
        open_end = open_end + timedelta(minutes=1)
        snap = df[(df.index >= open_ts) & (df.index < open_end)].copy()
        snap = snap[snap["symbol"].str.startswith(UNDERLYING)].copy()

    if snap.empty:
        print(f"  [skip] No QQQ data at open")
        return pd.DataFrame()

    if debug:
        print(f"  Snap rows: {len(snap)}")
        print(snap["symbol"].head(5).tolist())

    # ── 2. Parse symbols ─────────────────────────────────────────────────────
    parsed = snap["symbol"].map(parse_occ)
    snap   = snap[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    snap[["root", "expiry", "flag", "strike"]] = pd.DataFrame(
        parsed.tolist(), index=snap.index
    )
    snap = snap[snap["root"] == UNDERLYING].copy()

    # ── 0DTE filter: keep only options expiring today ────────────────────────
    snap = snap[snap["expiry"] == trade_date].copy()
    if snap.empty:
        print(f"  [skip] No 0DTE QQQ options found for {trade_date}")
        return pd.DataFrame()
    if debug:
        print(f"  0DTE rows: {len(snap)}")

    # ── 3. Compute mid price ─────────────────────────────────────────────────
    snap["bid"] = snap["bid_px_00"].replace(0, np.nan)
    snap["ask"] = snap["ask_px_00"].replace(0, np.nan)
    snap["mid"] = (snap["bid"] + snap["ask"]) / 2.0
    snap         = snap[snap["mid"] > 0].copy()

    # ── 4. Get spot price (daily open; 1-min only available last 30 days) ───
    hist = yf.download(
        UNDERLYING,
        start=trade_date.strftime("%Y-%m-%d"),
        end=(trade_date + timedelta(days=2)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if hist.empty:
        print(f"  [skip] yfinance returned no price data")
        return pd.DataFrame()
    open_col = hist["Open"]
    spot = float(open_col.iloc[0].iloc[0] if hasattr(open_col.iloc[0], "iloc") else open_col.iloc[0])
    print(f"  Spot (daily open): ${spot:.2f}")

    # ── 5. Filter strikes near spot ──────────────────────────────────────────
    lo, hi = spot * (1 - FILTER_PCT), spot * (1 + FILTER_PCT)
    snap   = snap[(snap["strike"] >= lo) & (snap["strike"] <= hi)].copy()

    # ── 6. Compute time-to-expiry and IV ─────────────────────────────────────
    snap["T"] = snap["expiry"].map(
        lambda e: max((e - trade_date).days / 365.0, 1 / 365.0)
    )
    snap["iv"] = snap.apply(
        lambda r: implied_vol(r["mid"], spot, r["strike"], r["T"], RISK_FREE, r["flag"]),
        axis=1,
    )
    snap = snap[snap["iv"].notna()].copy()

    # ── 7. Compute Greeks ────────────────────────────────────────────────────
    greeks = snap.apply(
        lambda r: bs_greeks(spot, r["strike"], r["T"], RISK_FREE, r["iv"], r["flag"]),
        axis=1,
    )
    snap[["delta", "gamma", "vega", "charm"]] = pd.DataFrame(
        greeks.tolist(), index=snap.index
    )

    # ── 8. OI proxy (CBBO has no OI; use consolidated quote size) ───────────
    # bid_sz_00 + ask_sz_00 = total contracts quoted at best bid/ask across
    # all OPRA exchanges. Underestimates true OI but preserves strike ranking.
    snap["oi"] = snap["bid_sz_00"] + snap["ask_sz_00"]
    snap = snap[snap["oi"] >= MIN_OI].copy()

    # ── 9. Aggregate per strike (sum across expiries) ────────────────────────
    MULT = 100  # options multiplier
    snap["gex_contrib"]    = snap["gamma"] * snap["oi"] * MULT * spot
    snap["vex_contrib"]    = snap["vega"]  * snap["oi"] * MULT
    snap["charmex_contrib"] = snap["charm"] * snap["oi"] * MULT

    # Net GEX: dealers short calls (positive gamma) and long puts (positive gamma)
    # Standard convention: call GEX positive, put GEX negative
    snap["gex_signed"] = np.where(
        snap["flag"] == "C",
         snap["gex_contrib"],
        -snap["gex_contrib"],
    )

    profile = (
        snap.groupby("strike")
        .agg(
            gex=("gex_signed", "sum"),
            vex=("vex_contrib", "sum"),
            charmex=("charmex_contrib", "sum"),
            oi=("oi", "sum"),
            iv_mean=("iv", "mean"),
        )
        .reset_index()
        .sort_values("strike")
    )

    profile["date"]      = trade_date
    profile["spot"]      = spot
    profile["dist_pct"]  = (profile["strike"] - spot) / spot * 100

    # Dynamic NQ/QQQ ratio: use actual price if available, else fallback
    ratio = (nq_open / spot) if (nq_open and nq_open > 0) else FALLBACK_RATIO
    profile["nq_qqq_ratio"]   = round(ratio, 4)
    profile["strike_futures"] = (profile["strike"] * ratio).round(1)

    # Normalised scores (0-100) — abs value, for magnitude ranking within day
    for col in ["gex", "vex", "charmex", "oi"]:
        abs_col = profile[col].abs()
        mn, mx  = abs_col.min(), abs_col.max()
        if mx > mn:
            profile[f"{col}_norm"] = (abs_col - mn) / (mx - mn) * 100
        else:
            profile[f"{col}_norm"] = 0.0

    # Multi-Greek confluence flag: GEX + VEX + CharmEX all above threshold
    # These are the walls where all three dealer hedging flows stack at one strike
    profile["confluence"] = (
        (profile["gex_norm"]     >= CONFLUENCE_THRESHOLD) &
        (profile["vex_norm"]     >= CONFLUENCE_THRESHOLD) &
        (profile["charmex_norm"] >= CONFLUENCE_THRESHOLD)
    ).astype(int)

    if debug:
        print(profile[["strike", "dist_pct", "gex", "gex_norm", "vex_norm",
                        "charmex_norm", "confluence"]].head(20).to_string())
        print(f"  Confluence walls: {profile['confluence'].sum()} / {len(profile)}")
        print(f"  NQ/QQQ ratio: {ratio:.4f} (nq_open={nq_open})")

    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    trade_date = date.fromisoformat(args.date)
    profile    = process_day(trade_date, debug=args.debug)

    if profile.empty:
        print("No output produced.")
        sys.exit(1)

    out_path = OUT_DIR_0DTE / f"gex_profile_{trade_date.strftime('%Y%m%d')}.csv"
    profile.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"  {len(profile)} strikes | spot ${profile['spot'].iloc[0]:.2f}")
    top = profile.reindex(profile["gex_norm"].nlargest(5).index)
    print("\nTop 5 GEX strikes:")
    print(top[["strike", "dist_pct", "gex", "gex_norm", "oi"]].to_string(index=False))


if __name__ == "__main__":
    main()
