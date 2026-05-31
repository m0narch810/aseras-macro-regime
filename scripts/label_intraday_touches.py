"""
Intraday wall touch labeling from NQ 1-minute bars.

For each GEX wall, scans every RTH 1-minute bar to detect when NQ price
approaches the wall level. Each approach event is labeled HELD or BROKE
based on the next 30 minutes of price action.

Why this is better than daily OHLC labeling:
- Same wall can be tested multiple times intraday (more samples)
- Captures bounces that reversed within the day (missed by daily close)
- Adds time-of-day, approach velocity, and repeat-test features
- Turns 358 daily labels into thousands of touch events

Output:
  data/processed/intraday_touches.csv

Usage:
  python scripts/label_intraday_touches.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT          = Path(__file__).parent.parent
SNAPSHOTS_DIR = ROOT / "data" / "processed" / "gex_snapshots_0dte"
NQ_1M_PATH    = ROOT / "data" / "processed" / "NQ_1m_clean.csv"
OUT_PATH      = ROOT / "data" / "processed" / "intraday_touches.csv"

FALLBACK_RATIO = 41.14

# Touch tolerance: within this many NQ points counts as approaching the wall
TOUCH_TOLERANCE_NQ = 8.0

# Forward window for labeling: how many minutes after touch to assess outcome
FORWARD_MINUTES = 30

# Minimum price movement through wall to call it BROKE (NQ points)
BREAK_THRESHOLD_NQ = 5.0

# RTH session bounds (ET, as hour floats)
RTH_START = 9.5   # 9:30
RTH_END   = 16.0  # 16:00

# Top N walls per day to track
TOP_N_WALLS = 10

# Minimum bars between touches of the same wall (avoid re-labeling same event)
MIN_BARS_BETWEEN_TOUCHES = 10


def load_nq_day(nq_df, date):
    """Extract RTH 1-minute bars for a single date."""
    day = nq_df[nq_df["date_only"] == date].copy()
    if day.empty:
        return day
    # Filter to RTH
    hour = day["dt"].dt.hour + day["dt"].dt.minute / 60
    return day[(hour >= RTH_START) & (hour < RTH_END)].reset_index(drop=True)


def label_touches(bars, wall_nq, wall_id):
    """
    Find all touch events for a single wall on a single day.
    Returns list of dicts, one per touch event.
    """
    if bars.empty or len(bars) < FORWARD_MINUTES:
        return []

    prices = bars["close"].values
    highs  = bars["high"].values
    lows   = bars["low"].values
    times  = bars["dt"].values

    touches = []
    last_touch_bar = -MIN_BARS_BETWEEN_TOUCHES  # allow first bar

    for i in range(len(bars) - FORWARD_MINUTES):
        if i - last_touch_bar < MIN_BARS_BETWEEN_TOUCHES:
            continue

        # Approach: high or low came within tolerance of wall
        touched_from_below = highs[i] >= wall_nq - TOUCH_TOLERANCE_NQ and prices[i] < wall_nq
        touched_from_above = lows[i]  <= wall_nq + TOUCH_TOLERANCE_NQ and prices[i] > wall_nq

        if not (touched_from_below or touched_from_above):
            continue

        last_touch_bar = i
        approach_dir   = "FROM_BELOW" if touched_from_below else "FROM_ABOVE"

        # Forward window
        fwd_prices = prices[i + 1: i + 1 + FORWARD_MINUTES]
        fwd_highs  = highs[i + 1:  i + 1 + FORWARD_MINUTES]
        fwd_lows   = lows[i + 1:   i + 1 + FORWARD_MINUTES]

        # Approach velocity: price change over last 5 bars, normalized to direction
        # toward wall so positive always means "moving faster toward the wall"
        start_idx  = max(0, i - 5)
        raw_vel    = (prices[i] - prices[start_idx]) / max(i - start_idx, 1)
        approach_vel = raw_vel if approach_dir == "FROM_BELOW" else -raw_vel

        # Number of prior touches of this wall today
        prior_touches = len(touches)

        if approach_dir == "FROM_BELOW":
            # BROKE: closes BREAK_THRESHOLD above wall and stays there for 3+ bars
            broke_bars = np.sum(fwd_prices > wall_nq + BREAK_THRESHOLD_NQ)
            outcome = "BROKE" if broke_bars >= 3 else "HELD"
        else:
            # BROKE: closes BREAK_THRESHOLD below wall and stays there for 3+ bars
            broke_bars = np.sum(fwd_prices < wall_nq - BREAK_THRESHOLD_NQ)
            outcome = "BROKE" if broke_bars >= 3 else "HELD"

        # Max excursion through the wall in the forward window
        if approach_dir == "FROM_BELOW":
            max_excursion = max(0, np.max(fwd_highs) - wall_nq)
        else:
            max_excursion = max(0, wall_nq - np.min(fwd_lows))

        touches.append({
            "wall_id":       wall_id,
            "touch_time":    times[i],
            "approach_dir":  approach_dir,
            "price_at_touch":round(float(prices[i]), 2),
            "approach_vel":  round(float(approach_vel), 4),
            "prior_touches": prior_touches,
            "time_of_day":   round(float(pd.Timestamp(times[i]).hour + pd.Timestamp(times[i]).minute / 60), 3),
            "max_excursion": round(float(max_excursion), 2),
            "outcome":       outcome,
        })

    return touches


def compute_gamma_flip(profile):
    """GEX zero crossing nearest to spot — returns NQ price of flip point."""
    spot = profile["spot"].iloc[0]
    sorted_p = profile.sort_values("strike")
    strikes = sorted_p["strike"].values
    gex     = sorted_p["gex"].values
    best_flip, best_dist = None, float("inf")
    for j in range(len(strikes) - 1):
        if (gex[j] > 0) != (gex[j + 1] > 0):
            # Linear interpolation
            flip = strikes[j] + (strikes[j+1] - strikes[j]) * abs(gex[j]) / (abs(gex[j]) + abs(gex[j+1]) + 1e-9)
            dist = abs(flip - spot)
            if dist < best_dist:
                best_dist, best_flip = dist, flip
    if best_flip is None:
        # Fallback: strike with minimum |GEX|
        best_flip = strikes[np.argmin(np.abs(gex))]
    return best_flip * FALLBACK_RATIO  # convert to NQ (ratio refined per-day later)


def load_day_snapshots(date_str):
    """
    Load all intraday snapshots for a date, sorted by time.
    Returns list of (snapshot_dt_str, profile_df) sorted ascending.
    """
    files = sorted(SNAPSHOTS_DIR.glob(f"gex_snapshot_{date_str}_*.csv"))
    snaps = []
    for f in files:
        p = pd.read_csv(f)
        # Extract time from filename: gex_snapshot_YYYYMMDD_HHMM.csv
        time_str = f.stem.split("_")[-1]  # e.g. "0931"
        snaps.append((time_str, p))
    return snaps


def nearest_prior_snapshot(snaps, touch_hour, touch_minute):
    """
    Return the snapshot taken most recently before the touch time.
    snaps: list of (time_str "HHMM", profile_df) sorted ascending.
    """
    touch_t = touch_hour * 100 + touch_minute
    best = None
    for time_str, profile in snaps:
        snap_t = int(time_str)
        if snap_t <= touch_t:
            best = profile
        else:
            break
    return best


def main():
    print("Loading NQ 1-minute data...")
    nq = pd.read_csv(NQ_1M_PATH, parse_dates=["date"])
    nq = nq.rename(columns={"date": "dt"})
    nq["date_only"] = nq["dt"].dt.date

    # Discover all days that have intraday snapshots
    snap_dates = sorted({
        f.stem[13:21]   # YYYYMMDD from gex_snapshot_YYYYMMDD_HHMM.csv
        for f in SNAPSHOTS_DIR.glob("gex_snapshot_*.csv")
    })
    print(f"Processing {len(snap_dates)} days with intraday snapshots...")

    all_touches = []
    skipped = 0

    for i, date_str in enumerate(snap_dates):
        date = pd.Timestamp(date_str).date()

        bars = load_nq_day(nq, date)
        if bars.empty:
            skipped += 1
            continue

        # Load all intraday snapshots for this day, sorted by time
        day_snaps = load_day_snapshots(date_str)
        if not day_snaps:
            skipped += 1
            continue

        # Scan every NQ bar in RTH; for each bar check all current snapshot walls
        # Key change: for every touch event, look up the most recent snapshot
        # BEFORE the touch time so wall positions reflect live market conditions.
        processed_wall_times = set()  # avoid double-labeling same wall at same minute

        for bar_i, bar in bars.iterrows():
            touch_hour   = bar["dt"].hour
            touch_minute = bar["dt"].minute

            # Get the most recent snapshot prior to this bar
            profile = nearest_prior_snapshot(day_snaps, touch_hour, touch_minute)
            if profile is None:
                continue

            ratio         = float(profile["nq_qqq_ratio"].iloc[0]) if "nq_qqq_ratio" in profile.columns else FALLBACK_RATIO
            gamma_flip_nq = compute_gamma_flip(profile)
            spot_nq       = profile["spot"].iloc[0] * ratio
            in_neg_gamma  = int(spot_nq < gamma_flip_nq)

            profile["abs_gex"] = profile["gex"].abs()
            top = profile.nlargest(TOP_N_WALLS, "abs_gex")

            for _, wall in top.iterrows():
                if "strike_futures" in wall and not pd.isna(wall["strike_futures"]):
                    wall_nq = float(wall["strike_futures"])
                else:
                    wall_nq = wall["strike"] * ratio

                # Skip if price not near this wall
                if abs(bar["close"] - wall_nq) > TOUCH_TOLERANCE_NQ * 3:
                    continue

                wall_id = f"{date}_{wall['strike']}_{touch_hour:02d}{touch_minute:02d}"
                if wall_id in processed_wall_times:
                    continue

                gex_raw     = float(wall["gex"])
                vex_raw     = float(wall["vex"])
                charmex_raw = float(wall["charmex"])
                abs_gex     = abs(gex_raw) + 1e-9

                wall_meta = {
                    "date":             str(date),
                    "strike_etf":       wall["strike"],
                    "wall_nq":          round(wall_nq, 1),
                    "snapshot_time":    f"{touch_hour:02d}{touch_minute:02d}",
                    "gex_norm":         round(float(wall["gex_norm"]), 4),
                    "vex_norm":         round(float(wall["vex_norm"]), 4),
                    "charmex_norm":     round(float(wall["charmex_norm"]), 4),
                    "oi_norm":          round(float(wall["oi_norm"]), 4),
                    "vex_over_gex":     round(vex_raw / abs_gex, 4),
                    "charmex_over_gex": round(charmex_raw / abs_gex, 4),
                    "gamma_flip_nq":    round(gamma_flip_nq, 1),
                    "in_neg_gamma":     in_neg_gamma,
                    "wall_above_flip":  int(wall_nq > gamma_flip_nq),
                    "iv_mean":          round(float(wall["iv_mean"]), 6),
                    "is_put":           int(gex_raw < 0),
                    "dist_pct":         round(float(wall["dist_pct"]), 4),
                    "confluence":       int(wall["confluence"]) if "confluence" in wall else 0,
                    "T_hours":          round(float(wall["T_hours"]), 3) if "T_hours" in wall else 6.0,
                }

                events = label_touches(bars, wall_nq, wall_id)
                if events:
                    processed_wall_times.add(wall_id)
                    for ev in events:
                        ev.update(wall_meta)
                        all_touches.append(ev)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(snap_dates)} days done, {len(all_touches)} touches so far")

    df = pd.DataFrame(all_touches)

    if df.empty:
        print("No touch events found.")
        return

    df["vol_regime"]  = np.where(df["iv_mean"] >= 0.30, "EXPANSION",
                        np.where(df["iv_mean"] >= 0.20, "NEUTRAL", "CONTRACTION"))
    df["is_high_vol"] = (df["vol_regime"] == "EXPANSION").astype(int)
    df["held"]        = (df["outcome"] == "HELD").astype(int)

    df.to_csv(OUT_PATH, index=False)

    n_held  = df["held"].sum()
    n_broke = len(df) - n_held
    print()
    print("=" * 55)
    print("  INTRADAY TOUCH LABELING COMPLETE")
    print("=" * 55)
    print(f"  Days processed:   {len(snap_dates) - skipped}")
    print(f"  Days skipped:     {skipped}")
    print(f"  Total touches:    {len(df)}")
    print(f"  Held:             {n_held}  ({n_held/len(df):.1%})")
    print(f"  Broke:            {n_broke}  ({n_broke/len(df):.1%})")
    print(f"  Saved:            {OUT_PATH}")
    print()
    print("  By vol regime:")
    print(df.groupby("vol_regime")["held"].agg(["count", "mean"]).rename(
        columns={"count": "N", "mean": "hold_rate"}).to_string())
    print()
    print("  By wall type:")
    df["wall_type"] = np.where(df["is_put"] == 1, "PUT_WALL", "CALL_WALL")
    print(df.groupby("wall_type")["held"].agg(["count", "mean"]).rename(
        columns={"count": "N", "mean": "hold_rate"}).to_string())


if __name__ == "__main__":
    main()
