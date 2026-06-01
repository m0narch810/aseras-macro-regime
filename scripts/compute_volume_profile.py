"""
Volume profile utilities for NQ 1-minute bar data.

Computes POC, VAH/VAL (70% value area), HVNs, and LVNs from OHLCV bars
using a fixed-size price bin histogram. No scipy dependency — pure numpy.

Usage (as a module):
    from scripts.compute_volume_profile import build_vp_cache, vp_features_for_wall

Bin size of 5 NQ points matches typical GEX strike granularity (~QQQ $1 × 41 ratio).
Alignment tolerance of 10 pts = 2 bins, generous enough for rounding across ratio.
"""

import numpy as np
import pandas as pd

BIN_SIZE      = 5.0   # NQ points per profile bin
ALIGN_TOL     = 10.0  # NQ pts — "on" a VP level if within this distance
VALUE_AREA    = 0.70  # 70% of session volume defines value area
HVN_THRESHOLD = 0.65  # local max must be >= this fraction of session POC volume
LVN_THRESHOLD = 0.30  # local min must be <= this fraction of mean bin volume
RTH_START     = 9.5
RTH_END       = 16.0


# ── Core VP computation ───────────────────────────────────────────────────────

def _local_extrema(arr):
    """
    Returns (maxima_idx, minima_idx) as index arrays.
    A point is a local max if strictly greater than both neighbours.
    """
    n = len(arr)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    maxima = [i for i in range(1, n - 1) if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]]
    minima = [i for i in range(1, n - 1) if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]]
    return np.array(maxima, dtype=int), np.array(minima, dtype=int)


def build_vp(bars, bin_size=BIN_SIZE):
    """
    Build a volume profile from RTH 1-minute bars.

    Uses typical price (H+L+C)/3 weighted by bar volume — fully vectorized
    via np.histogram.  ~100x faster than per-bar/per-bin Python loops at the
    cost of intra-bar price distribution (irrelevant at 5-pt bin resolution).
    Returns DataFrame with columns [price, volume] where price is bin centre.
    """
    if bars.empty:
        return pd.DataFrame(columns=["price", "volume"])

    typical = (bars["high"].values + bars["low"].values + bars["close"].values) / 3.0
    vols    = bars["volume"].values.astype(float)

    lo_global = np.floor(typical.min() / bin_size) * bin_size - bin_size
    hi_global = np.ceil(typical.max()  / bin_size) * bin_size + bin_size
    edges     = np.arange(lo_global, hi_global + bin_size, bin_size)

    bin_vol, _ = np.histogram(typical, bins=edges, weights=vols)

    centres = (edges[:-1] + edges[1:]) / 2
    vp = pd.DataFrame({"price": centres, "volume": bin_vol})
    return vp[vp["volume"] > 0].reset_index(drop=True)


def get_poc_vah_val(vp):
    """
    Returns (poc, vah, val) from a VP DataFrame.
    Value area is the smallest contiguous range around POC containing ≥70% of volume.
    """
    if vp.empty:
        return None, None, None

    vols   = vp["volume"].values
    prices = vp["price"].values
    poc_i  = int(np.argmax(vols))

    total   = vols.sum()
    target  = total * VALUE_AREA
    lo_i, hi_i = poc_i, poc_i
    accum   = vols[poc_i]
    n       = len(vols)

    while accum < target:
        up_v = vols[hi_i + 1] if hi_i < n - 1 else 0.0
        dn_v = vols[lo_i - 1] if lo_i > 0     else 0.0
        if up_v == 0 and dn_v == 0:
            break
        if up_v >= dn_v:
            hi_i  += 1
            accum += up_v
        else:
            lo_i  -= 1
            accum += dn_v

    return float(prices[poc_i]), float(prices[hi_i]), float(prices[lo_i])


def get_hvns(vp):
    """
    High-volume nodes: local maxima whose volume >= HVN_THRESHOLD × POC volume.
    Returns array of NQ prices.
    """
    if len(vp) < 3:
        return np.array([])
    vols   = vp["volume"].values
    prices = vp["price"].values
    poc_v  = vols.max()
    cutoff = poc_v * HVN_THRESHOLD
    maxima, _ = _local_extrema(vols)
    return prices[maxima[vols[maxima] >= cutoff]]


def get_lvns(vp):
    """
    Low-volume nodes: local minima whose volume <= LVN_THRESHOLD × mean bin volume.
    Returns array of NQ prices.
    """
    if len(vp) < 3:
        return np.array([])
    vols   = vp["volume"].values
    prices = vp["price"].values
    cutoff = vols.mean() * LVN_THRESHOLD
    _, minima = _local_extrema(vols)
    return prices[minima[vols[minima] <= cutoff]]


# ── Feature computation for a single wall ────────────────────────────────────

def _nearest(price, levels):
    if len(levels) == 0:
        return np.nan
    return float(np.min(np.abs(levels - price)))


def vp_features_for_wall(wall_nq, vp, prefix="pd"):
    """
    Given a wall price and a volume profile, return a flat dict of VP features.

    prefix — "pd" for previous-day, "pw" for previous-week.

    Features:
      {prefix}_on_hvn         — 1 if wall within ALIGN_TOL of an HVN
      {prefix}_on_lvn         — 1 if wall within ALIGN_TOL of an LVN
      {prefix}_in_value_area  — 1 if wall between VAL and VAH
      {prefix}_dist_to_poc    — NQ pts to previous-session POC
      {prefix}_dist_to_hvn    — NQ pts to nearest HVN
      {prefix}_dist_to_lvn    — NQ pts to nearest LVN
    """
    prefix = prefix + "_"
    empty = {
        f"{prefix}on_hvn":        0,
        f"{prefix}on_lvn":        0,
        f"{prefix}in_value_area": 0,
        f"{prefix}dist_to_poc":   np.nan,
        f"{prefix}dist_to_hvn":   np.nan,
        f"{prefix}dist_to_lvn":   np.nan,
    }
    if vp is None or vp.empty:
        return empty

    poc, vah, val = get_poc_vah_val(vp)
    hvns          = get_hvns(vp)
    lvns          = get_lvns(vp)

    dist_poc = abs(wall_nq - poc) if poc is not None else np.nan
    dist_hvn = _nearest(wall_nq, hvns)
    dist_lvn = _nearest(wall_nq, lvns)
    in_va    = int(val is not None and vah is not None and val <= wall_nq <= vah)

    return {
        f"{prefix}on_hvn":        int(not np.isnan(dist_hvn) and dist_hvn <= ALIGN_TOL),
        f"{prefix}on_lvn":        int(not np.isnan(dist_lvn) and dist_lvn <= ALIGN_TOL),
        f"{prefix}in_value_area": in_va,
        f"{prefix}dist_to_poc":   round(dist_poc, 2) if not np.isnan(dist_poc) else np.nan,
        f"{prefix}dist_to_hvn":   round(dist_hvn, 2) if not np.isnan(dist_hvn) else np.nan,
        f"{prefix}dist_to_lvn":   round(dist_lvn, 2) if not np.isnan(dist_lvn) else np.nan,
    }


# ── Cache builder (call once before the main labeling loop) ──────────────────

def build_vp_cache(nq_df):
    """
    Pre-computes RTH volume profiles for every trading day in nq_df.

    nq_df must have columns: dt (datetime), open, high, low, close, volume
    and a date_only column (date objects).

    Returns dict: {date: vp_dataframe}
    """
    cache = {}
    all_dates = sorted(nq_df["date_only"].unique())
    for d in all_dates:
        day_bars = nq_df[nq_df["date_only"] == d]
        hour = day_bars["dt"].dt.hour + day_bars["dt"].dt.minute / 60
        rth  = day_bars[(hour >= RTH_START) & (hour < RTH_END)]
        cache[d] = build_vp(rth)
    return cache


def build_week_vp_cache(vp_cache, all_dates_sorted):
    """
    Pre-computes 5-day rolling VP for every date by merging the five prior
    daily VP histograms (bin-level addition — no bar re-iteration needed).
    Returns dict: {date: vp_dataframe}
    """
    week_cache = {}
    dates = list(all_dates_sorted)
    for i, d in enumerate(dates):
        prior = dates[max(0, i - 5): i]
        if not prior:
            week_cache[d] = pd.DataFrame(columns=["price", "volume"])
            continue
        # Merge daily VPs by outer-joining on price and summing volume
        combined = None
        for pd_date in prior:
            dv = vp_cache.get(pd_date)
            if dv is None or dv.empty:
                continue
            combined = dv if combined is None else (
                pd.concat([combined, dv])
                  .groupby("price", as_index=False)["volume"].sum()
            )
        week_cache[d] = combined if combined is not None else pd.DataFrame(columns=["price", "volume"])
    return week_cache


def get_prev_day_vp(cache, trade_date, all_dates_sorted):
    """Most recent trading day before trade_date."""
    idx = all_dates_sorted.searchsorted(trade_date, side="left")
    if idx == 0:
        return None
    return cache.get(all_dates_sorted[idx - 1])


def get_prev_week_vp(week_cache, trade_date, all_dates_sorted):
    """Pre-computed 5-day rolling VP for trade_date."""
    idx = all_dates_sorted.searchsorted(trade_date, side="left")
    if idx == 0:
        return None
    return week_cache.get(all_dates_sorted[idx - 1])
