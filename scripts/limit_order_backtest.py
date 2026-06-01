"""
Limit-order GEX wall backtest.

Simulates the actual trading workflow:
  1. Walk through intraday snapshots (9:31, 10:00 ... 15:30)
  2. At each snapshot, score walls by hold_prob + VP alignment
  3. Qualifying walls (hold_prob >= threshold, dist >= min_dist, VP/confluence)
     get a limit order placed at the wall level
  4. Limit fills when NQ 1m price reaches the wall after identification
  5. Track stop/target from fill bar — multi-hour hold supported
  6. One trade per wall per day

Performance: ALL data preloaded into memory before grid search.
Zero disk I/O inside the simulation loop.

Usage:
  python scripts/limit_order_backtest.py
"""

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compute_volume_profile import (
    build_vp_cache, build_week_vp_cache,
    get_prev_day_vp, get_prev_week_vp,
    vp_features_for_wall,
)

ROOT          = Path(__file__).parent.parent
SNAPSHOTS_DIR = ROOT / "data" / "processed" / "gex_snapshots_0dte"
NQ_1M_PATH    = ROOT / "data" / "processed" / "NQ_1m_clean.csv"
MODEL_JSON    = ROOT / "models" / "wall_score_intraday.json"
OUT_PATH      = ROOT / "data" / "processed" / "limit_order_backtest.csv"

FALLBACK_RATIO  = 41.14
TOUCH_TOL       = 8.0   # NQ pts — limit fills when price within this of wall
TOP_N_WALLS     = 10
RTH_START       = 9.5
RTH_END         = 16.0


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(path):
    with open(path) as f:
        d = json.load(f)
    return (d["features"], d["lr_coefficients"],
            d["lr_intercept"], d["scaler_mean"], d["scaler_scale"])


def hold_prob(row_dict, feats, coefs, intercept, means, scales):
    logit = intercept
    for f in feats:
        raw    = float(row_dict.get(f, 0) or 0)
        logit += coefs[f] * (raw - means[f]) / (scales[f] + 1e-9)
    return float(1 / (1 + np.exp(-logit)))


# ── Preload everything ────────────────────────────────────────────────────────

def preload_snapshots(snap_date_strs):
    """
    Load all snapshot CSVs once into memory.
    Returns dict: {date_str -> [(hhmm_int, profile_df), ...]} sorted by hhmm.
    """
    data = {}
    for ds in snap_date_strs:
        files = sorted(SNAPSHOTS_DIR.glob(f"gex_snapshot_{ds}_*.csv"))
        snaps = []
        for f in files:
            hhmm = int(f.stem.split("_")[-1])
            snaps.append((hhmm, pd.read_csv(f)))
        data[ds] = snaps
    return data


def preload_nq_by_date(nq_path):
    """
    Load NQ 1m bars and split into per-date numpy arrays for fast forward scans.
    Returns dict: {date -> DataFrame(dt, open, high, low, close)}
    """
    nq = pd.read_csv(nq_path, parse_dates=["date"])
    nq = nq.rename(columns={"date": "dt"})
    nq["date_only"] = nq["dt"].dt.date
    hour = nq["dt"].dt.hour + nq["dt"].dt.minute / 60
    nq   = nq[(hour >= RTH_START) & (hour < RTH_END)].copy()

    by_date = {}
    for date, grp in nq.groupby("date_only"):
        by_date[date] = grp.reset_index(drop=True)
    return by_date


def precompute_wall_features(snap_data, vp_cache, week_cache, all_dates_sorted,
                              feats, coefs, intercept, means, scales):
    """
    For every (date, snapshot, wall), precompute all features and hold_prob.
    Returns dict:
      {date_str -> [(hhmm, spot_nq, ratio, [wall_dict, ...]), ...]}

    wall_dict keys: wall_nq, wall_key, approach_dir, hold_prob, dist_pct,
                    vp_aligned, confluence, is_put
    """
    precomputed = {}

    for date_str, snaps in snap_data.items():
        date   = pd.Timestamp(date_str).date()
        pd_vp  = get_prev_day_vp(vp_cache, date, all_dates_sorted)
        pw_vp  = get_prev_week_vp(week_cache, date, all_dates_sorted)

        day_snaps = []
        for hhmm, profile in snaps:
            snap_hour = hhmm // 100 + (hhmm % 100) / 60
            if snap_hour >= RTH_END - 0.5:
                continue

            ratio   = float(profile["nq_qqq_ratio"].iloc[0]) if "nq_qqq_ratio" in profile.columns else FALLBACK_RATIO
            spot_nq = float(profile["spot"].iloc[0]) * ratio

            profile["abs_gex"] = profile["gex"].abs()
            profile_top = profile.nlargest(TOP_N_WALLS, "abs_gex")

            walls = []
            for _, wall in profile_top.iterrows():
                if "strike_futures" in wall and not pd.isna(wall.get("strike_futures")):
                    wall_nq = float(wall["strike_futures"])
                else:
                    wall_nq = float(wall["strike"]) * ratio

                wall_key = round(wall_nq / 5) * 5
                dist_pct = abs(wall_nq - spot_nq) / spot_nq * 100

                gex_raw    = float(wall["gex"])
                is_put     = int(gex_raw < 0)
                approach   = "FROM_ABOVE" if is_put else "FROM_BELOW"
                confluence = int(wall["confluence"]) if "confluence" in wall.index else 0
                iv_mean    = float(wall.get("iv_mean", 0.25))
                abs_gex    = abs(gex_raw) + 1e-9

                vpf_pd     = vp_features_for_wall(wall_nq, pd_vp, prefix="pd")
                vpf_pw     = vp_features_for_wall(wall_nq, pw_vp, prefix="pw")
                vp_aligned = int(bool(vpf_pd.get("pd_on_hvn", 0)) or
                                 bool(vpf_pw.get("pw_on_hvn", 0)))
                vp_lvn     = int(bool(vpf_pd.get("pd_on_lvn", 0)) or
                                 bool(vpf_pw.get("pw_on_lvn", 0)))

                row = {
                    "gex_norm":         float(wall.get("gex_norm", 50)),
                    "vex_norm":         float(wall.get("vex_norm", 50)),
                    "charmex_norm":     float(wall.get("charmex_norm", 50)),
                    "oi_norm":          float(wall.get("oi_norm", 50)),
                    "vex_over_gex":     float(wall.get("vex", 0)) / abs_gex,
                    "charmex_over_gex": float(wall.get("charmex", 0)) / abs_gex,
                    "dist_pct":         dist_pct if is_put else -dist_pct,
                    "is_high_vol":      int(iv_mean >= 0.30),
                    "is_contraction":   int(iv_mean < 0.20),
                    "is_put":           is_put,
                    "in_neg_gamma":     0,
                    "wall_above_flip":  0,
                    "confluence":       confluence,
                    "time_of_day":      snap_hour,
                    "approach_vel":     0,
                    "from_below":       int(approach == "FROM_BELOW"),
                    "pd_on_hvn":        int(vpf_pd.get("pd_on_hvn", 0)),
                    "pd_on_lvn":        int(vpf_pd.get("pd_on_lvn", 0)),
                    "pd_in_value_area": int(vpf_pd.get("pd_in_value_area", 0)),
                    "pd_dist_to_poc":   vpf_pd.get("pd_dist_to_poc", 0) or 0,
                    "pd_dist_to_hvn":   vpf_pd.get("pd_dist_to_hvn", 0) or 0,
                    "pd_dist_to_lvn":   vpf_pd.get("pd_dist_to_lvn", 0) or 0,
                    "pw_on_hvn":        int(vpf_pw.get("pw_on_hvn", 0)),
                    "pw_on_lvn":        int(vpf_pw.get("pw_on_lvn", 0)),
                    "pw_in_value_area": int(vpf_pw.get("pw_in_value_area", 0)),
                    "pw_dist_to_poc":   vpf_pw.get("pw_dist_to_poc", 0) or 0,
                    "vp_aligned":       vp_aligned,
                }

                hp = hold_prob(row, feats, coefs, intercept, means, scales)

                walls.append({
                    "wall_nq":    wall_nq,
                    "wall_key":   wall_key,
                    "approach":   approach,
                    "hp":         hp,
                    "dist_pct":   dist_pct,
                    "vp_aligned": vp_aligned,
                    "vp_lvn":     vp_lvn,
                    "confluence": confluence,
                    "is_put":     is_put,
                    "snap_hour":  snap_hour,
                    "snap_hhmm":  hhmm,
                })

            day_snaps.append((hhmm, spot_nq, ratio, walls))

        precomputed[date_str] = day_snaps

    return precomputed


# ── Simulation (pure in-memory, no I/O) ──────────────────────────────────────

def find_fill_idx(bars_close, bars_low, bars_high, from_idx, wall_nq, tol=TOUCH_TOL):
    for i in range(from_idx, len(bars_close)):
        if bars_low[i] <= wall_nq + tol and bars_high[i] >= wall_nq - tol:
            return i
    return -1


def sim_trade(bars_close, fill_idx, wall_nq, approach, stop_nq, target_nq, max_bars):
    end = min(fill_idx + max_bars + 1, len(bars_close))
    for j in range(fill_idx + 1, end):
        c = bars_close[j]
        fav = (c - wall_nq) if approach == "FROM_ABOVE" else (wall_nq - c)
        adv = (wall_nq - c) if approach == "FROM_ABOVE" else (c - wall_nq)
        if fav >= target_nq:
            return 1   # WIN
        if adv >= stop_nq:
            return 0   # LOSS
    return -1  # TIMEOUT


def run_combo(precomputed, nq_by_date, min_hp, min_dist, qualify_mode,
              approach_dirs, stop_nq, target_nq, max_bars):
    trades = []

    for date_str, day_snaps in precomputed.items():
        date     = pd.Timestamp(date_str).date()
        bars_df  = nq_by_date.get(date)
        if bars_df is None or bars_df.empty:
            continue

        bars_dt    = bars_df["dt"].values
        bars_close = bars_df["close"].values
        bars_low   = bars_df["low"].values
        bars_high  = bars_df["high"].values
        n_bars     = len(bars_close)

        traded_keys = set()
        pending     = {}   # wall_key -> {wall_nq, approach, hp, from_bar_idx, ...}

        for hhmm, spot_nq, ratio, walls in day_snaps:
            # Convert snapshot time to bar index
            snap_ts  = pd.Timestamp(date_str).replace(
                hour=hhmm // 100, minute=hhmm % 100, second=0)
            snap_bar = int(np.searchsorted(bars_dt, np.datetime64(snap_ts), side="right"))

            # Check if any pending limits filled since last snapshot
            for wk, info in list(pending.items()):
                fill_idx = find_fill_idx(bars_close, bars_low, bars_high,
                                         info["from_bar"], info["wall_nq"])
                if fill_idx != -1 and fill_idx < snap_bar:
                    result = sim_trade(bars_close, fill_idx, info["wall_nq"],
                                       info["approach"], stop_nq, target_nq, max_bars)
                    trades.append({
                        "date":       date_str,
                        "wall_nq":    info["wall_nq"],
                        "hp":         info["hp"],
                        "vp_aligned": info["vp_aligned"],
                        "vp_lvn":     info["vp_lvn"],
                        "confluence": info["confluence"],
                        "approach":   info["approach"],
                        "dist_at_id": info["dist_pct"],
                        "outcome":    result,
                        "won":        int(result == 1),
                    })
                    traded_keys.add(wk)
                    del pending[wk]

            # Identify new qualifying walls at this snapshot
            for w in walls:
                wk = w["wall_key"]
                if wk in traded_keys or wk in pending:
                    continue
                if w["dist_pct"] < min_dist:
                    continue
                if w["hp"] < min_hp:
                    continue
                if w["approach"] not in approach_dirs:
                    continue

                # Structural qualification
                ok = False
                if qualify_mode == "hp_only":
                    ok = True
                elif qualify_mode == "vp_or_conf":
                    ok = bool(w["vp_aligned"]) or bool(w["confluence"])
                elif qualify_mode == "lvn_or_conf":
                    ok = bool(w["vp_lvn"]) or bool(w["confluence"])
                elif qualify_mode == "lvn_only":
                    ok = bool(w["vp_lvn"])
                elif qualify_mode == "vp_only":
                    ok = bool(w["vp_aligned"])

                if not ok:
                    continue

                pending[wk] = {
                    "wall_nq":  w["wall_nq"],
                    "approach": w["approach"],
                    "hp":       w["hp"],
                    "from_bar": snap_bar,
                    "dist_pct": w["dist_pct"],
                    "vp_aligned": w["vp_aligned"],
                    "vp_lvn":   w["vp_lvn"],
                    "confluence": w["confluence"],
                }

        # End of day: check remaining pending limits
        for wk, info in list(pending.items()):
            fill_idx = find_fill_idx(bars_close, bars_low, bars_high,
                                     info["from_bar"], info["wall_nq"])
            if fill_idx != -1:
                result = sim_trade(bars_close, fill_idx, info["wall_nq"],
                                   info["approach"], stop_nq, target_nq, max_bars)
                trades.append({
                    "date":       date_str,
                    "wall_nq":    info["wall_nq"],
                    "hp":         info["hp"],
                    "vp_aligned": info["vp_aligned"],
                    "vp_lvn":     info["vp_lvn"],
                    "confluence": info["confluence"],
                    "approach":   info["approach"],
                    "dist_at_id": info["dist_pct"],
                    "outcome":    result,
                    "won":        int(result == 1),
                })

    return trades


# ── Grid search ───────────────────────────────────────────────────────────────

def grid_search(precomputed, nq_by_date, n_days):
    param_grid = {
        "min_hold_prob":  [0.45, 0.50, 0.55],
        "min_dist_pct":   [0.3, 0.5, 1.0],
        "qualify_mode":   ["hp_only", "vp_or_conf", "lvn_or_conf", "lvn_only"],
        "approach_dirs":  [["FROM_ABOVE", "FROM_BELOW"], ["FROM_ABOVE"]],
        "stop_nq":        [15, 20, 25],
        "target_nq":      [75, 100, 125, 150],
        "max_hold_bars":  [60, 120, 180, 240],
    }

    keys   = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    total  = len(combos)
    print(f"  {total:,} parameter combinations", flush=True)

    results = []
    import time
    t0 = time.time()

    for ci, combo in enumerate(combos):
        p = dict(zip(keys, combo))

        trades = run_combo(
            precomputed, nq_by_date,
            p["min_hold_prob"], p["min_dist_pct"], p["qualify_mode"],
            p["approach_dirs"], p["stop_nq"], p["target_nq"], p["max_hold_bars"],
        )

        if ci % 200 == 0 and ci > 0:
            el  = time.time() - t0
            eta = (total - ci) / (ci / el)
            print(f"  {ci}/{total} ({ci/total:.0%})  "
                  f"elapsed={el:.0f}s  eta={eta:.0f}s  "
                  f"results={len(results)}", flush=True)

        if len(trades) < 10:
            continue

        df  = pd.DataFrame(trades)
        n   = len(df)
        wr  = df["won"].mean()
        ev  = wr * p["target_nq"] + (1 - wr) * (-p["stop_nq"])
        tpd = n / n_days

        results.append({
            "win_rate":       round(wr, 4),
            "ev":             round(ev, 2),
            "n":              n,
            "trades_per_day": round(tpd, 2),
            "stop_nq":        p["stop_nq"],
            "target_nq":      p["target_nq"],
            "rr":             round(p["target_nq"] / p["stop_nq"], 1),
            "max_hold_bars":  p["max_hold_bars"],
            "min_hold_prob":  p["min_hold_prob"],
            "min_dist_pct":   p["min_dist_pct"],
            "qualify_mode":   p["qualify_mode"],
            "approach_dirs":  "+".join(p["approach_dirs"]),
        })

    return pd.DataFrame(results)


def sep(t=""):
    print(f"\n{'='*65}\n  {t}\n{'='*65}" if t else "-"*65)


def main():
    sep("LIMIT ORDER BACKTEST — GEX WALL REVERSAL")

    print("Loading NQ 1m bars...", flush=True)
    nq_by_date = preload_nq_by_date(NQ_1M_PATH)
    print(f"  {len(nq_by_date)} trading days loaded")

    snap_date_strs = sorted({
        f.stem[13:21] for f in SNAPSHOTS_DIR.glob("gex_snapshot_*.csv")
    })
    print(f"  {len(snap_date_strs)} snapshot days found")

    print("Building VP caches...", flush=True)
    nq_full          = pd.read_csv(NQ_1M_PATH, parse_dates=["date"])
    nq_full          = nq_full.rename(columns={"date": "dt"})
    nq_full["date_only"] = nq_full["dt"].dt.date
    vp_cache         = build_vp_cache(nq_full)
    all_dates_sorted = np.array(sorted(vp_cache.keys()))
    week_cache       = build_week_vp_cache(vp_cache, all_dates_sorted)
    print(f"  {len(vp_cache)} daily VPs cached")

    print("Loading model...", flush=True)
    model_params = load_model(MODEL_JSON)
    feats, coefs, intercept, means, scales = model_params

    print("Preloading all snapshot CSVs...", flush=True)
    snap_data = preload_snapshots(snap_date_strs)
    print(f"  Done — {sum(len(v) for v in snap_data.values())} snapshots in memory")

    print("Precomputing wall features + hold_prob for all snapshots...", flush=True)
    precomputed = precompute_wall_features(
        snap_data, vp_cache, week_cache, all_dates_sorted,
        feats, coefs, intercept, means, scales,
    )
    print(f"  Done")

    sep("GRID SEARCH")
    results = grid_search(precomputed, nq_by_date, len(snap_date_strs))

    if results.empty:
        print("No viable setups found (all < 10 trades).")
        return

    results.to_csv(OUT_PATH, index=False)
    print(f"\n  {len(results):,} viable sets → {OUT_PATH}")

    cols = ["win_rate","ev","n","trades_per_day","rr","stop_nq","target_nq",
            "max_hold_bars","min_hold_prob","min_dist_pct","qualify_mode","approach_dirs"]

    sep("TOP 20 BY WIN RATE  (≥55% wr, ≥0.2 trades/day)")
    top = results[
        (results["win_rate"] >= 0.55) & (results["trades_per_day"] >= 0.2)
    ].sort_values("win_rate", ascending=False).head(20)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(top[cols].to_string(index=False) if not top.empty else "  None")

    sep("TOP 15 BY EV  (≥55% wr, target ≥ 100, ≥0.2 trades/day)")
    top_ev = results[
        (results["win_rate"] >= 0.55) &
        (results["target_nq"] >= 100) &
        (results["trades_per_day"] >= 0.2)
    ].sort_values("ev", ascending=False).head(15)
    print(top_ev[cols].to_string(index=False) if not top_ev.empty else "  None")

    sep("FREQUENCY SWEET SPOT  (0.3–1.0 trades/day, ≥55% wr)")
    freq = results[
        (results["trades_per_day"] >= 0.3) &
        (results["trades_per_day"] <= 1.0) &
        (results["win_rate"] >= 0.55)
    ].sort_values("ev", ascending=False).head(15)
    print(freq[cols].to_string(index=False) if not freq.empty else "  None")

    sep("RECOMMENDED SETUP")
    cands = results[
        (results["win_rate"] >= 0.60) &
        (results["trades_per_day"] >= 0.2) &
        (results["target_nq"] >= 100) &
        (results["ev"] > 0)
    ]
    if cands.empty:
        cands = results[(results["win_rate"] >= 0.55) & (results["ev"] > 0)]
    if not cands.empty:
        b = cands.sort_values("ev", ascending=False).iloc[0]
        print(f"  Win rate:           {b['win_rate']:.1%}")
        print(f"  Expected value:     +{b['ev']:.1f} NQ pts / trade")
        print(f"  R:R                 {b['rr']:.1f}:1")
        print(f"  N trades:           {b['n']}  over {int(b['n']/b['trades_per_day'])} days")
        print(f"  Avg trades/day:     {b['trades_per_day']:.2f}")
        print(f"  Stop:               {b['stop_nq']} NQ pts")
        print(f"  Target:             {b['target_nq']} NQ pts")
        print(f"  Max hold:           {b['max_hold_bars']} min")
        print(f"  Min hold_prob:      {b['min_hold_prob']}")
        print(f"  Min dist from spot: {b['min_dist_pct']}%")
        print(f"  Qualify mode:       {b['qualify_mode']}")
        print(f"  Direction:          {b['approach_dirs']}")
    else:
        print("  No setup with positive EV and ≥55% win rate found.")


if __name__ == "__main__":
    main()
