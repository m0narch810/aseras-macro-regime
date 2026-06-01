"""
Reversal backtest — vectorized parameter search.

Precomputes full forward adverse/favorable traces for every touch event,
then evaluates all parameter combinations in pure numpy — no Python loops
in the inner loop.

Usage:
  python scripts/reversal_backtest.py
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT         = Path(__file__).parent.parent
TOUCHES_PATH = ROOT / "data" / "processed" / "intraday_touches.csv"
NQ_1M_PATH   = ROOT / "data" / "processed" / "NQ_1m_clean.csv"
OUT_PATH     = ROOT / "data" / "processed" / "reversal_backtest.csv"

MAX_BARS = 120
STOP_NQS    = [15, 20, 25, 30, 40]
TARGET_NQS  = [100, 125, 150, 175, 200]


def sep(title=""):
    if title:
        print(f"\n{'=' * 65}")
        print(f"  {title}")
        print("=" * 65)
    else:
        print("-" * 65)


# ── Step 1: Build (n_touches × MAX_BARS) favorable/adverse matrices ──────────

def build_traces(touches, nq):
    """
    Returns:
      fav_mat   : (N, MAX_BARS) — favorable move per bar (away from wall)
      adv_mat   : (N, MAX_BARS) — adverse move per bar (through wall)
      valid_mask: (N,) bool — True if we got enough forward bars
    """
    N          = len(touches)
    fav_mat    = np.full((N, MAX_BARS), np.nan)
    adv_mat    = np.full((N, MAX_BARS), np.nan)
    nq_index   = nq.index
    nq_closes  = nq["close"].values
    n_nq       = len(nq_closes)

    wall_arr   = touches["wall_nq"].values
    dt_arr     = touches["touch_time"].values          # numpy datetime64
    dir_arr    = touches["approach_dir"].values

    for i in range(N):
        pos = nq_index.searchsorted(dt_arr[i], side="right")
        end = min(pos + MAX_BARS, n_nq)
        nb  = end - pos
        if nb < 5:
            continue
        closes = nq_closes[pos:end]
        wall   = wall_arr[i]
        if dir_arr[i] == "FROM_BELOW":
            fav_mat[i, :nb] = wall - closes
            adv_mat[i, :nb] = closes - wall
        else:
            fav_mat[i, :nb] = closes - wall
            adv_mat[i, :nb] = wall - closes

    valid_mask = ~np.isnan(fav_mat[:, 0])
    return fav_mat, adv_mat, valid_mask


# ── Step 2: Precompute first_stop_bar and first_target_bar for each combo ────

def precompute_outcomes(fav_mat, adv_mat):
    """
    Fully vectorized: no Python loops over touches or bars.

    Stop requires 2 consecutive bars through the threshold (same logic as before).
    Uses numpy sliding-window AND to find the first such pair per row.
    """
    N  = fav_mat.shape[0]
    ns = len(STOP_NQS)
    nt = len(TARGET_NQS)

    first_stop   = np.full((N, ns), MAX_BARS + 1, dtype=np.int32)
    first_target = np.full((N, nt), MAX_BARS + 1, dtype=np.int32)

    adv_clean = np.nan_to_num(adv_mat, nan=-np.inf)
    fav_clean = np.nan_to_num(fav_mat, nan=-np.inf)

    for si, stop in enumerate(STOP_NQS):
        hit   = adv_clean >= stop                       # (N, MAX_BARS)
        consec = hit[:, :-1] & hit[:, 1:]              # (N, MAX_BARS-1): pair hit
        any_hit = consec.any(axis=1)                    # (N,)
        first_stop[any_hit, si] = np.argmax(consec[any_hit], axis=1).astype(np.int32)

    for ti, tgt in enumerate(TARGET_NQS):
        hit     = fav_clean >= tgt                      # (N, MAX_BARS)
        any_hit = hit.any(axis=1)
        first_target[any_hit, ti] = np.argmax(hit[any_hit], axis=1).astype(np.int32)

    max_fav = np.where(
        np.all(np.isnan(fav_mat), axis=1),
        -np.inf,
        np.nanmax(fav_mat, axis=1),
    )
    return first_stop, first_target, max_fav


# ── Step 3: Simulate one parameter combo (vectorized over all touches) ────────

def simulate_vec(idx, first_stop, first_target, si, ti, max_bars, **_):
    """
    idx         : boolean array of shape (N,) — entry filter mask
    first_stop  : (N, ns)
    first_target: (N, nt)
    si, ti      : indices into STOP_NQS / TARGET_NQS
    max_bars    : int
    Returns (wins, losses, n) or None if n < 15.
    """
    fs  = first_stop[idx, si]    # first stop bar for filtered trades
    ft  = first_target[idx, ti]  # first target bar

    stop_nq   = STOP_NQS[si]
    target_nq = TARGET_NQS[ti]

    # Win: target reached before stop AND within max_bars
    win_mask  = (ft <= max_bars) & (ft < fs)
    # Loss: stop hit first, OR neither reached within max_bars
    loss_mask = ~win_mask

    n = idx.sum()
    if n < 15:
        return None

    wins   = int(win_mask.sum())
    losses = int(loss_mask.sum())
    if wins + losses == 0:
        return None

    win_rate = wins / (wins + losses)
    avg_win  = float(target_nq)
    avg_loss = float(-stop_nq)
    ev       = win_rate * avg_win + (1 - win_rate) * avg_loss

    return wins, losses, n, round(win_rate, 4), round(ev, 2)


# ── Step 4: Grid search over entry filters ────────────────────────────────────

def grid_search(touches_valid, first_stop, first_target):
    param_grid = {
        "vol_regimes":      [
            ["CONTRACTION"],
            ["CONTRACTION", "NEUTRAL"],
            ["CONTRACTION", "NEUTRAL", "EXPANSION"],
        ],
        "min_gex_norm":     [0, 40, 60, 75],
        "min_time":         [9.5, 10.5, 12.0, 13.0],
        "max_time":         [16.0, 15.0, 14.0],
        "max_approach_vel": [999, 5, 2],
        "approach_dirs":    [
            ["FROM_BELOW", "FROM_ABOVE"],
            ["FROM_BELOW"],
            ["FROM_ABOVE"],
        ],
        "min_dist_pct":     [0, 1.5, 3.0],
        # confluence_only: Greek multi-confluence requirement (original filter)
        "confluence_only":  [False, True],
        # vp_aligned_only: wall must sit on a prior-day or prior-week HVN
        # replaces confluence for higher-frequency setups
        "vp_aligned_only":  [False, True],
        "stop_idx":         list(range(len(STOP_NQS))),
        "target_idx":       list(range(len(TARGET_NQS))),
        "max_bars":         [45, 60, 90, 120],
    }

    keys   = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"  {len(combos):,} combinations to evaluate...")

    # Precompute all unique filter masks once — avoids rebuilding 489K arrays 1.5M times
    vr_arr    = touches_valid["vol_regime"].values
    gex_arr   = touches_valid["gex_norm"].values
    time_arr  = touches_valid["time_of_day"].values
    vel_arr   = touches_valid["approach_vel"].values
    dir_arr   = touches_valid["approach_dir"].values
    dist_arr  = touches_valid["dist_pct"].abs().values
    conf_arr  = touches_valid["confluence"].values if "confluence" in touches_valid.columns else np.zeros(len(touches_valid))
    vpal_arr  = touches_valid["vp_aligned"].values if "vp_aligned" in touches_valid.columns else np.zeros(len(touches_valid))

    N = len(touches_valid)

    vr_masks   = {tuple(v): np.isin(vr_arr, v)   for v in param_grid["vol_regimes"]}
    gex_masks  = {g: gex_arr  >= g                for g in param_grid["min_gex_norm"]}
    tlo_masks  = {t: time_arr >= t                for t in param_grid["min_time"]}
    thi_masks  = {t: time_arr <= t                for t in param_grid["max_time"]}
    vel_masks  = {v: vel_arr  <= v                for v in param_grid["max_approach_vel"]}
    dir_masks  = {tuple(d): np.isin(dir_arr, d)   for d in param_grid["approach_dirs"]}
    dist_masks = {d: dist_arr >= d                for d in param_grid["min_dist_pct"]}
    conf_mask  = conf_arr == 1
    vpal_mask  = vpal_arr == 1
    all_true   = np.ones(N, dtype=bool)

    import sys, time
    t0 = time.time()
    results = []
    for ci, combo in enumerate(combos):
        if ci % 100_000 == 0 and ci > 0:
            elapsed = time.time() - t0
            rate = ci / elapsed
            eta = (len(combos) - ci) / rate
            print(f"  {ci:,}/{len(combos):,}  ({ci/len(combos):.0%})  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  results={len(results):,}", flush=True)
        p = dict(zip(keys, combo))

        if p["min_time"] >= p["max_time"]:
            continue

        mask = (
            vr_masks[tuple(p["vol_regimes"])] &
            gex_masks[p["min_gex_norm"]] &
            tlo_masks[p["min_time"]] &
            thi_masks[p["max_time"]] &
            vel_masks[p["max_approach_vel"]] &
            dir_masks[tuple(p["approach_dirs"])] &
            dist_masks[p["min_dist_pct"]] &
            (conf_mask if p["confluence_only"]  else all_true) &
            (vpal_mask if p["vp_aligned_only"]  else all_true)
        )

        out = simulate_vec(mask, first_stop, first_target,
                           p["stop_idx"], p["target_idx"], p["max_bars"])
        if out is None:
            continue

        wins, losses, n, win_rate, ev = out
        results.append({
            "win_rate":        win_rate,
            "ev":              ev,
            "n":               n,
            "wins":            wins,
            "losses":          losses,
            "stop_nq":         STOP_NQS[p["stop_idx"]],
            "target_nq":       TARGET_NQS[p["target_idx"]],
            "max_bars":        p["max_bars"],
            "vol_regimes":     "+".join(p["vol_regimes"]),
            "min_gex_norm":    p["min_gex_norm"],
            "min_time":        p["min_time"],
            "max_time":        p["max_time"],
            "max_approach_vel":p["max_approach_vel"],
            "approach_dirs":   "+".join(p["approach_dirs"]),
            "min_dist_pct":    p["min_dist_pct"],
            "confluence_only": p["confluence_only"],
            "vp_aligned_only": p["vp_aligned_only"],
        })

    return pd.DataFrame(results)


def print_top(res, sort_col, n=15, title=None, extra_filter=None):
    df = res.copy()
    if extra_filter is not None:
        df = df[extra_filter(df)]
    top = df.sort_values(sort_col, ascending=False).head(n)
    if title:
        sep(title)
    cols = ["win_rate","ev","n","stop_nq","target_nq","max_bars",
            "vol_regimes","min_gex_norm","min_time","max_time",
            "max_approach_vel","approach_dirs","min_dist_pct",
            "confluence_only","vp_aligned_only"]
    cols = [c for c in cols if c in top.columns]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(top[cols].to_string(index=False))


def main():
    sep("REVERSAL BACKTEST — VECTORIZED PARAMETER SEARCH")

    print("  Loading touch events...")
    touches = pd.read_csv(TOUCHES_PATH)
    touches["touch_time"] = pd.to_datetime(touches["touch_time"])

    print("  Loading NQ 1-minute bars...")
    nq = pd.read_csv(NQ_1M_PATH, parse_dates=["date"])
    nq = nq.rename(columns={"date": "dt"})
    nq = nq[nq["dt"] >= "2025-02-01"].sort_values("dt").set_index("dt")
    print(f"  NQ bars: {len(nq):,}  |  Touch events: {len(touches)}")

    sep("BUILDING FORWARD TRACES")
    print(f"  Scanning {MAX_BARS} bars ahead for each touch...")
    fav_mat, adv_mat, valid = build_traces(touches, nq)
    touches_v = touches[valid].reset_index(drop=True)
    fav_v     = fav_mat[valid]
    adv_v     = adv_mat[valid]
    print(f"  Valid trades: {valid.sum()} / {len(touches)}")

    sep("PRECOMPUTING STOP / TARGET HITS")
    first_stop, first_target, max_fav = precompute_outcomes(fav_v, adv_v)

    sep("RAW REVERSAL POTENTIAL (no filters)")
    print(f"  Max favorable move in {MAX_BARS}-bar window:")
    for tgt in TARGET_NQS:
        ti  = TARGET_NQS.index(tgt)
        pct = (first_target[:, ti] <= MAX_BARS).mean()
        print(f"    >= {tgt:3d} NQ pts reached: {pct:.1%}  ({int(pct * len(touches_v))} trades)")
    print(f"  Median max favorable: {np.median(max_fav):.1f} NQ pts")
    print(f"  Mean   max favorable: {np.mean(max_fav):.1f} NQ pts")
    print()
    print("  By vol regime:")
    for reg in ["CONTRACTION", "NEUTRAL", "EXPANSION"]:
        m = touches_v["vol_regime"] == reg
        if m.sum():
            print(f"    {reg:<15} N={m.sum():4d}  "
                  f"median_fav={np.median(max_fav[m.values]):.0f}  "
                  f"mean_fav={np.mean(max_fav[m.values]):.0f}")
    print()
    print("  By time of day:")
    for lbl, lo, hi in [("9:30-10:30",9.5,10.5),("10:30-12",10.5,12.0),
                         ("12-14",12.0,14.0),("14-16",14.0,16.0)]:
        m = (touches_v["time_of_day"] >= lo) & (touches_v["time_of_day"] < hi)
        if m.sum():
            print(f"    {lbl:<12} N={m.sum():4d}  "
                  f"median_fav={np.median(max_fav[m.values]):.0f}  "
                  f"mean_fav={np.mean(max_fav[m.values]):.0f}")

    sep("GRID SEARCH")
    results = grid_search(touches_v, first_stop, first_target)
    results.to_csv(OUT_PATH, index=False)
    print(f"  {len(results):,} viable parameter sets found and saved.")

    if results.empty:
        print("  No viable sets found — all filtered below 15 trades.")
        return

    print_top(results, "win_rate", n=20,
              title="TOP 20 BY WIN RATE (all targets)")

    print_top(results, "win_rate", n=15,
              title="TOP 15 WIN RATE — 100 NQ target",
              extra_filter=lambda d: d["target_nq"] == 100)

    print_top(results, "ev", n=15,
              title="TOP 15 EXPECTED VALUE — 100 NQ target",
              extra_filter=lambda d: d["target_nq"] == 100)

    print_top(results, "n", n=10,
              title="MOST TRADES — win_rate >= 60%, target 100+",
              extra_filter=lambda d: (d["win_rate"] >= 0.60) & (d["target_nq"] >= 100))

    sep("RECOMMENDED SETUP")
    cands = results[
        (results["target_nq"] == 100) &
        (results["win_rate"]  >= 0.65) &
        (results["n"]         >= 20)
    ]
    if cands.empty:
        cands = results[
            (results["target_nq"] >= 75) &
            (results["win_rate"]  >= 0.60) &
            (results["n"]         >= 15)
        ]
    if cands.empty:
        cands = results[results["ev"] > 0].sort_values("ev", ascending=False)

    if not cands.empty:
        b = cands.sort_values("ev", ascending=False).iloc[0]
        print(f"  Win rate:         {b['win_rate']:.1%}")
        print(f"  Expected value:   {b['ev']:+.1f} NQ pts / trade")
        print(f"  N trades (sample):{b['n']}")
        print(f"  Stop:             {b['stop_nq']} NQ pts")
        print(f"  Target:           {b['target_nq']} NQ pts")
        print(f"  Max hold:         {b['max_bars']} min")
        print(f"  Vol regime:       {b['vol_regimes']}")
        print(f"  Min GEX norm:     {b['min_gex_norm']}")
        print(f"  Time window (ET): {b['min_time']:.1f} - {b['max_time']:.1f}")
        print(f"  Approach vel:     <= {b['max_approach_vel']}")
        print(f"  Direction:        {b['approach_dirs']}")
        print(f"  Min dist from spot: {b['min_dist_pct']}%")
    else:
        print("  No setup with positive EV and sufficient trades found.")


if __name__ == "__main__":
    main()
