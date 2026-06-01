"""
Label wall touch events from SPY GEX profiles using ES 1m price action.

Key difference from label_intraday_touches.py:
  - ONE snapshot per day (EOD from day D) used as the wall map for day D+1
  - No volume-profile features (VP columns set to 0)
  - strike_futures column already contains ES-equivalent levels (from decode_spy_eod.py)
  - gamma_flip computed from the EOD profile (same math, no snapshot interpolation needed)

Output:
  data/processed/spy_touches.csv  — same column schema as intraday_touches.csv

Usage:
  python scripts/label_spy_touches.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT           = Path(__file__).parent.parent
PROFILES_DIR   = ROOT / "data" / "processed" / "gex_profiles_spy"
ES_1M_PATH     = ROOT / "data" / "raw" / "ES" / "1Min_ES.csv"
OUT_PATH       = ROOT / "data" / "processed" / "spy_touches.csv"

# Touch detection parameters — same as label_intraday_touches.py
TOUCH_TOLERANCE_ES    = 8.0     # ES points
FORWARD_MINUTES       = 30
BREAK_THRESHOLD_ES    = 5.0     # ES points, >= 3 bars beyond = BROKE
MIN_BARS_BETWEEN      = 10
TOP_N_WALLS           = 10
RTH_START             = 9.5     # 9:30 ET
RTH_END               = 16.0    # 16:00 ET


# ── ES 1m loader (same logic as decode_spy_eod.py) ───────────────────────────

def load_es_1m(path: Path) -> pd.DataFrame:
    """
    Load ES 1m bars from European-formatted semicolon-separated CSV.
    Returns DataFrame with columns: dt, date_only, Open, High, Low, Close.
    """
    print(f"Loading ES 1m bars from {path} ...")

    # Detect junk first row
    raw_head = pd.read_csv(path, sep=";", nrows=2, header=0, dtype=str)
    first_date_val = raw_head.iloc[0]["Date"] if "Date" in raw_head.columns else ""
    skip = 0
    try:
        pd.to_datetime(first_date_val, format="%m/%d/%Y %I:%M %p")
    except Exception:
        skip = 1
        print(f"  First data row appears junk ('{first_date_val}') — skipping")

    es = pd.read_csv(
        path,
        sep=";",
        skiprows=range(1, 1 + skip),
        thousands=".",
        decimal=",",
        parse_dates=False,
        dtype={"Symbol": str},
    )

    if "Date" not in es.columns:
        es = pd.read_csv(
            path, sep=";", thousands=".", decimal=",",
            parse_dates=False, dtype={"Symbol": str},
        )

    es["dt"] = pd.to_datetime(es["Date"], format="%m/%d/%Y %I:%M %p", errors="coerce")
    es = es.dropna(subset=["dt"])
    es["date_only"] = es["dt"].dt.date

    print(f"  Loaded {len(es):,} ES 1m bars  "
          f"({es['date_only'].min()} to {es['date_only'].max()})")
    return es[["dt", "date_only", "Open", "High", "Low", "Close"]].copy()


def get_rth_bars(es_df: pd.DataFrame, trade_date) -> pd.DataFrame:
    """Extract RTH (9:30-16:00) ES 1m bars for a single date."""
    day = es_df[es_df["date_only"] == trade_date].copy()
    if day.empty:
        return day
    hour = day["dt"].dt.hour + day["dt"].dt.minute / 60.0
    rth  = day[(hour >= RTH_START) & (hour < RTH_END)].reset_index(drop=True)
    return rth


# ── GEX profile loader ────────────────────────────────────────────────────────

def load_spy_profile(date_str: str) -> pd.DataFrame:
    """Load a single SPY GEX profile CSV by YYYYMMDD date string."""
    path = PROFILES_DIR / f"gex_profile_spy_{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ── Gamma flip from EOD profile ───────────────────────────────────────────────

def compute_gamma_flip_es(profile: pd.DataFrame) -> float:
    """
    Compute the GEX zero-crossing nearest to spot, returned in ES points.
    Uses strike_futures (already in ES equivalent) directly.
    Falls back to the strike with minimum |GEX| if no crossing found.
    """
    if "strike_futures" not in profile.columns or profile.empty:
        return float("nan")

    spot_es = float(profile["spot"].iloc[0] * profile["nq_qqq_ratio"].iloc[0])
    sorted_p = profile.sort_values("strike_futures")
    levels   = sorted_p["strike_futures"].values
    gex_vals = sorted_p["gex"].values

    best_flip, best_dist = None, float("inf")
    for j in range(len(levels) - 1):
        if (gex_vals[j] > 0) != (gex_vals[j + 1] > 0):
            # Linear interpolation
            flip = (levels[j]
                    + (levels[j + 1] - levels[j])
                    * abs(gex_vals[j])
                    / (abs(gex_vals[j]) + abs(gex_vals[j + 1]) + 1e-9))
            dist = abs(flip - spot_es)
            if dist < best_dist:
                best_dist, best_flip = dist, flip

    if best_flip is None:
        best_flip = levels[np.argmin(np.abs(gex_vals))]

    return float(best_flip)


# ── Touch detection (same logic as label_intraday_touches.py) ─────────────────

def label_touches(bars: pd.DataFrame, wall_es: float, wall_id: str) -> list:
    """
    Find all touch events for a single wall on a single day.
    bars: RTH ES 1m DataFrame with columns dt, Open, High, Low, Close.
    Returns list of dicts (one per touch event).
    """
    if bars.empty or len(bars) < FORWARD_MINUTES:
        return []

    prices = bars["Close"].values
    highs  = bars["High"].values
    lows   = bars["Low"].values
    times  = bars["dt"].values

    touches = []
    last_touch_bar = -MIN_BARS_BETWEEN

    for i in range(len(bars) - FORWARD_MINUTES):
        if i - last_touch_bar < MIN_BARS_BETWEEN:
            continue

        touched_from_below = (highs[i] >= wall_es - TOUCH_TOLERANCE_ES
                              and prices[i] < wall_es)
        touched_from_above = (lows[i]  <= wall_es + TOUCH_TOLERANCE_ES
                              and prices[i] > wall_es)

        if not (touched_from_below or touched_from_above):
            continue

        last_touch_bar = i
        approach_dir   = "FROM_BELOW" if touched_from_below else "FROM_ABOVE"

        fwd_prices = prices[i + 1: i + 1 + FORWARD_MINUTES]
        fwd_highs  = highs[i + 1:  i + 1 + FORWARD_MINUTES]
        fwd_lows   = lows[i + 1:   i + 1 + FORWARD_MINUTES]

        start_idx    = max(0, i - 5)
        raw_vel      = (prices[i] - prices[start_idx]) / max(i - start_idx, 1)
        approach_vel = raw_vel if approach_dir == "FROM_BELOW" else -raw_vel

        prior_touches = len(touches)

        if approach_dir == "FROM_BELOW":
            broke_bars = int(np.sum(fwd_prices > wall_es + BREAK_THRESHOLD_ES))
            outcome    = "BROKE" if broke_bars >= 3 else "HELD"
            max_excursion = max(0.0, float(np.max(fwd_highs)) - wall_es)
        else:
            broke_bars = int(np.sum(fwd_prices < wall_es - BREAK_THRESHOLD_ES))
            outcome    = "BROKE" if broke_bars >= 3 else "HELD"
            max_excursion = max(0.0, wall_es - float(np.min(fwd_lows)))

        ts = pd.Timestamp(times[i])
        touches.append({
            "wall_id":        wall_id,
            "touch_time":     ts,
            "approach_dir":   approach_dir,
            "price_at_touch": round(float(prices[i]), 2),
            "approach_vel":   round(float(approach_vel), 4),
            "prior_touches":  prior_touches,
            "time_of_day":    round(float(ts.hour + ts.minute / 60.0), 3),
            "max_excursion":  round(float(max_excursion), 2),
            "outcome":        outcome,
        })

    return touches


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load ES 1m ──────────────────────────────────────────────────────────
    es_df = load_es_1m(ES_1M_PATH)

    # ── Discover available SPY GEX profile dates ─────────────────────────────
    profile_files = sorted(PROFILES_DIR.glob("gex_profile_spy_*.csv"))
    if not profile_files:
        print(f"ERROR: No profile files found in {PROFILES_DIR}")
        print("  Run scripts/decode_spy_eod.py first.")
        return

    profile_dates = []
    for f in profile_files:
        # filename: gex_profile_spy_YYYYMMDD.csv
        date_str = f.stem.replace("gex_profile_spy_", "")
        try:
            dt = pd.to_datetime(date_str, format="%Y%m%d").date()
            profile_dates.append((date_str, dt))
        except Exception:
            continue

    profile_dates.sort(key=lambda x: x[1])
    print(f"\nFound {len(profile_dates)} SPY GEX profile days  "
          f"({profile_dates[0][1]} to {profile_dates[-1][1]})")

    # Build a date → next trading date index from ES 1m (to find D+1 correctly)
    es_trading_dates = sorted(es_df["date_only"].unique())
    es_date_to_next  = {}
    for idx, d in enumerate(es_trading_dates):
        if idx + 1 < len(es_trading_dates):
            es_date_to_next[d] = es_trading_dates[idx + 1]

    # ── Process each profile day ─────────────────────────────────────────────
    all_touches = []
    skipped_no_bars = 0
    skipped_no_walls = 0

    for i, (date_str, profile_date) in enumerate(profile_dates):
        # The EOD profile for date D is used as wall map for trading day D+1
        next_date = es_date_to_next.get(profile_date)
        if next_date is None:
            skipped_no_bars += 1
            continue

        bars = get_rth_bars(es_df, next_date)
        if bars.empty or len(bars) < FORWARD_MINUTES:
            skipped_no_bars += 1
            continue

        profile = load_spy_profile(date_str)
        if profile.empty:
            skipped_no_walls += 1
            continue

        # Spot (in ES points) for this profile
        spot_spy = float(profile["spot"].iloc[0])
        ratio    = float(profile["nq_qqq_ratio"].iloc[0])
        spot_es  = spot_spy * ratio

        # Gamma flip in ES points
        gamma_flip_es = compute_gamma_flip_es(profile)
        in_neg_gamma  = int(spot_es < gamma_flip_es) if not np.isnan(gamma_flip_es) else 0

        # Vol regime from iv_mean (threshold same as existing pipeline)
        day_iv = float(profile["iv_mean"].median())
        if day_iv >= 0.30:
            vol_regime = "EXPANSION"
        elif day_iv >= 0.20:
            vol_regime = "NEUTRAL"
        else:
            vol_regime = "CONTRACTION"

        # Top N walls by absolute GEX
        profile["abs_gex"] = profile["gex"].abs()
        top_walls = profile.nlargest(TOP_N_WALLS, "abs_gex")

        for _, wall in top_walls.iterrows():
            wall_es = float(wall["strike_futures"])

            # Skip if entire RTH session is nowhere near this wall
            if abs(bars["Close"].mean() - wall_es) > TOUCH_TOLERANCE_ES * 20:
                continue

            gex_raw     = float(wall["gex"])
            vex_raw     = float(wall["vex"])
            charmex_raw = float(wall["charmex"])
            abs_gex     = abs(gex_raw) + 1e-9

            # vex_over_gex and charmex_over_gex computed from raw values
            # (will default to training means in production per CLAUDE.md)
            vex_over_gex     = round(vex_raw / abs_gex, 4)
            charmex_over_gex = round(charmex_raw / abs_gex, 4)

            wall_meta = {
                "date":              str(next_date),         # trading day of the touch
                "profile_date":      str(profile_date),      # EOD profile used
                "strike_etf":        wall["strike"],
                "wall_nq":           round(wall_es, 1),
                "snapshot_time":     "1600",                 # EOD snapshot indicator
                "gex_norm":          round(float(wall["gex_norm"]),     4),
                "vex_norm":          round(float(wall["vex_norm"]),     4),
                "charmex_norm":      round(float(wall["charmex_norm"]), 4),
                "oi_norm":           round(float(wall["oi_norm"]),      4),
                "vex_over_gex":      vex_over_gex,
                "charmex_over_gex":  charmex_over_gex,
                "gamma_flip_nq":     round(float(gamma_flip_es), 1),
                "in_neg_gamma":      in_neg_gamma,
                "wall_above_flip":   int(wall_es > gamma_flip_es) if not np.isnan(gamma_flip_es) else 0,
                "iv_mean":           round(float(wall["iv_mean"]), 6),
                "is_put":            int(gex_raw < 0),
                "dist_pct":          round(float(wall["dist_pct"]), 4),
                "confluence":        int(wall["confluence"]) if "confluence" in wall else 0,
                "T_hours":           round(float(wall["T_hours"]), 3),
                # Volume profile features — not computed for SPY dataset; set to 0/NaN
                "pd_on_hvn":         0,
                "pd_on_lvn":         0,
                "pd_in_value_area":  0,
                "pd_dist_to_poc":    0.0,
                "pd_dist_to_hvn":    0.0,
                "pd_dist_to_lvn":    0.0,
                "pw_on_hvn":         0,
                "pw_on_lvn":         0,
                "pw_in_value_area":  0,
                "pw_dist_to_poc":    0.0,
                "vp_aligned":        0,
                # Source tag (consumed only during combined training diagnostics)
                "source":            "spy",
            }

            wall_id = f"{next_date}_{wall['strike']}_spy"
            events  = label_touches(bars, wall_es, wall_id)

            for ev in events:
                ev.update(wall_meta)
                all_touches.append(ev)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(profile_dates)}] {profile_date} -> {next_date}  "
                  f"touches so far: {len(all_touches)}")

    if not all_touches:
        print("\nNo touch events found — check profile directory and ES 1m date coverage.")
        return

    df = pd.DataFrame(all_touches)

    # Derived columns
    df["vol_regime"]  = np.where(df["iv_mean"] >= 0.30, "EXPANSION",
                        np.where(df["iv_mean"] >= 0.20, "NEUTRAL", "CONTRACTION"))
    df["is_high_vol"] = (df["vol_regime"] == "EXPANSION").astype(int)
    df["is_contraction"] = (df["vol_regime"] == "CONTRACTION").astype(int)
    df["held"]        = (df["outcome"] == "HELD").astype(int)
    df["from_below"]  = (df["approach_dir"] == "FROM_BELOW").astype(int)

    # Ensure column order matches intraday_touches.csv where possible
    lead_cols = [
        "wall_id", "touch_time", "approach_dir", "price_at_touch", "approach_vel",
        "prior_touches", "time_of_day", "max_excursion", "outcome",
        "date", "profile_date", "strike_etf", "wall_nq", "snapshot_time",
        "gex_norm", "vex_norm", "charmex_norm", "oi_norm",
        "vex_over_gex", "charmex_over_gex",
        "gamma_flip_nq", "in_neg_gamma", "wall_above_flip",
        "iv_mean", "is_put", "dist_pct", "confluence", "T_hours",
        "pd_on_hvn", "pd_on_lvn", "pd_in_value_area",
        "pd_dist_to_poc", "pd_dist_to_hvn", "pd_dist_to_lvn",
        "pw_on_hvn", "pw_on_lvn", "pw_in_value_area", "pw_dist_to_poc",
        "vp_aligned",
        "vol_regime", "is_high_vol", "is_contraction", "from_below", "held",
        "source",
    ]
    # Keep any extra columns at the end
    extra = [c for c in df.columns if c not in lead_cols]
    df    = df[lead_cols + extra]

    df.to_csv(OUT_PATH, index=False)

    n_held  = int(df["held"].sum())
    n_broke = len(df) - n_held
    print()
    print("=" * 60)
    print("  SPY TOUCH LABELING COMPLETE")
    print("=" * 60)
    print(f"  Profile days processed:  {len(profile_dates) - skipped_no_bars - skipped_no_walls}")
    print(f"  Skipped (no ES bars):    {skipped_no_bars}")
    print(f"  Skipped (no walls):      {skipped_no_walls}")
    print(f"  Total touches:           {len(df)}")
    print(f"  Held:                    {n_held}  ({n_held/len(df):.1%})")
    print(f"  Broke:                   {n_broke}  ({n_broke/len(df):.1%})")
    print(f"  Saved:                   {OUT_PATH}")
    print()
    print("  By vol regime:")
    print(df.groupby("vol_regime")["held"].agg(["count", "mean"]).rename(
        columns={"count": "N", "mean": "hold_rate"}).to_string())
    print()
    print("  By wall type:")
    df["wall_type"] = np.where(df["is_put"] == 1, "PUT_WALL", "CALL_WALL")
    print(df.groupby("wall_type")["held"].agg(["count", "mean"]).rename(
        columns={"count": "N", "mean": "hold_rate"}).to_string())
    print()
    print("  By year (regime coverage):")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    print(df.groupby("year")["held"].agg(["count", "mean"]).rename(
        columns={"count": "N", "mean": "hold_rate"}).to_string())


if __name__ == "__main__":
    main()
