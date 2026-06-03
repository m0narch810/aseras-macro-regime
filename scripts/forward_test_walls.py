"""
forward_test_walls.py — self-supervised forward test of the mechanical hold_prob.

Idea (per the "forward-test yourself with yfinance" plan): we already log full
GEX snapshots to the gex-snapshots branch, and the new hold_prob is a pure
deterministic function of those logged inputs. So we can:

  1. Replay the mechanical Wall-Hold Reliability R for every logged 0DTE wall.
  2. Pull NQ=F 1-minute bars from yfinance (post-hoc — the ~15-min delay is
     irrelevant once the bars are final; only the 7-day history cap matters).
  3. For each wall price actually TOUCHED intraday, label HELD vs BROKE over the
     next HOLD_WINDOW_MIN minutes (2-bar confirmed break past a buffer).
  4. Report calibration: hold rate by R bucket + rank AUC of R vs outcome.

This is a PILOT-grade harness. Caveats it does not pretend away:
  - Snapshot density is currently ~5/day (GitHub drops the */5 cron), so n is
    small until capture is denser.
  - "HELD in 30 min" validates reliability, not trade profitability.
  - Historical snapshots predate call_oi/put_oi/strike_iv, so the PCR and skew
    factors replay neutral (×1.0) on old data — faithful once the logger adds them.

Usage:
  python scripts/forward_test_walls.py                 # all available days
  python scripts/forward_test_walls.py --csv out.csv   # also write per-wall rows
"""

import argparse, subprocess, io, csv, math, sys
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

FILTER_PCT      = 5.0    # match the function: only score strikes within ±5%
HOLD_WINDOW_MIN = 30     # minutes after touch to judge hold/break
BREAK_BUFFER    = 0.0010 # price must clear the wall by 0.10% to count as progress
BREAK_CONFIRM   = 2      # consecutive 1m closes beyond buffer = confirmed break
TOUCH_TOL       = 0.0003 # bar must come within 0.03% of the strike to "touch"


# ── mechanical hold_prob, ported verbatim from lib/options.js ──────────────────
def compute_gtbr(futures, iv, t_et):
    if not futures or not iv or iv <= 0:
        return None
    daily1sd = futures * (iv / 100.0) / math.sqrt(252)
    t_rem = max(0.0, 16.0 - (t_et if t_et else 9.5)) / 6.5
    return daily1sd * (math.sqrt(t_rem) if t_rem > 0 else 1.0)


def compute_protrusion(values, half=3):
    a = [abs(v) for v in values]
    ratios = []
    for i, v in enumerate(a):
        s, n = 0.0, 0
        for j in range(max(0, i - half), min(len(a), i + half + 1)):
            if j != i:
                s += a[j]; n += 1
        lm = s / n if n else 0.0
        ratios.append(1.0 if lm < 1e-9 else min(v / (lm + 1e-9), 6.0))
    mn, mx = min(ratios), max(ratios)
    if mx == mn:
        return [1.0] * len(ratios)
    return [(r - mn) / (mx - mn) for r in ratios]


def hold_prob(w, ctx):
    """w: dict for one strike; ctx: per-snapshot context. Mirrors computeHoldProb."""
    spot   = ctx["futures"]
    netgex = w["net_gex"]
    absgex = w.get("abs_gex") or abs(netgex)
    netdex = w["net_dex"]
    netdag = abs(w.get("net_dag") or 0.0)
    dist   = w["dist_nq"]

    # 1. regime baseline with vol-scaled NEAR_FLIP band (0.5 / 0.3 / 0.1)
    B = 0.3
    flip = ctx.get("gamma_flip")
    if flip is not None and spot is not None:
        band = max(30.0, 0.5 * spot * (ctx["iv"] / 100.0) / math.sqrt(252)) if ctx.get("iv") else 50.0
        diff = spot - flip
        B = 0.5 if diff > band else 0.1 if diff < -band else 0.3

    # 2a protrusion → 0.5–1.5
    P = 0.5 + (w.get("protrusion") if w.get("protrusion") is not None else 0.5)

    # 2b one-sidedness |GEX|/ag → 0.85–1.15
    ratio = min(1.0, abs(netgex) / absgex) if absgex > 0 else 0.5
    O = max(0.85, min(1.15, 0.85 + 0.6 * (ratio - 0.5)))

    # 2c hedge polarity (counter-trend ×1.25 / pro-trend ×0.5)
    above = dist > 0
    counter = (netdex > 0) if above else (netdex < 0)
    A_dex = 1.25 if counter else 0.5

    # 2d dominant-side OI asymmetry → 0.9–1.1; neutral (1.0) when split unlogged
    co, po = w.get("call_oi"), w.get("put_oi")
    if co is not None and po is not None and (co + po) > 0:
        asym = max(co, po) / (co + po)
        PCR = 0.9 + 0.4 * (asym - 0.5)
    else:
        PCR = 1.0

    # 2e side-aware skew; neutral when strike_iv unlogged
    S_skew = 1.0
    siv, atm = w.get("strike_iv"), ctx.get("iv")
    if siv is not None and atm and 0 < siv < 5 * atm and abs(siv - atm) / atm > 0.20:
        S_skew = 1.15

    # 3a term structure
    F_term = 0.85 if (ctx.get("hv_term") is not None and ctx["hv_term"] > 1.25) else 1.0
    # 3b variance risk premium
    F_vrp = 1.15 if (ctx.get("rv_iv") is not None and ctx["rv_iv"] < 0.5) else 1.0
    # 3c GTBR inelasticity
    F_gtbr = 0.2 if (ctx.get("gtbr") is not None and abs(dist) > ctx["gtbr"]) else 1.0
    # 4 late-session pinning
    G_pin = 1.25 if (ctx.get("t_et", 0) > 14 and ctx.get("dag_thr") is not None
                     and netdag >= ctx["dag_thr"]) else 1.0

    R = B * P * O * A_dex * PCR * S_skew * F_term * F_vrp * F_gtbr * G_pin
    return round(min(1.0, R), 3)


# ── snapshot loading + context assembly ───────────────────────────────────────
def _f(x):
    try:
        return float(x) if x not in (None, "") else None
    except (ValueError, TypeError):
        return None


def load_day(date_str):
    """Return {timestamp_et: [wall dicts]} for the 0DTE expiry of one logged day."""
    out = subprocess.run(["git", "show", f"origin/gex-snapshots:logs/levels_{date_str}.csv"],
                         capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(out)))
    snaps = {}
    for r in rows:
        if r["expiry"] != r["snapshot_date"]:   # 0DTE only
            continue
        snaps.setdefault(r["intended_timestamp"], []).append(r)

    result = {}
    for ts, rws in snaps.items():
        spot = _f(rws[0]["nq_price"])
        if not spot:
            continue
        # filter to ±FILTER_PCT, mirror nearbyStrikes
        near = [r for r in rws if _f(r["dist_pct"]) is not None
                and abs(_f(r["dist_pct"])) <= FILTER_PCT]
        if len(near) < 5:
            continue
        gex = [_f(r["net_gex"]) or 0 for r in near]
        prot = compute_protrusion(gex)
        dags = sorted(abs(_f(r["net_dag"]) or 0) for r in near)
        dag_thr = dags[min(len(dags) - 1, int(0.9 * (len(dags) - 1)))] if dags else None

        t_et = datetime.strptime(ts.replace(" ET", ""), "%Y-%m-%dT%H:%M:%S")
        hv5, hv63 = _f(rws[0]["hv5"]), _f(rws[0]["hv63"])
        iv = _f(rws[0]["iv"])
        ctx = {
            "futures": spot, "gamma_flip": _f(rws[0]["gamma_flip"]),
            "iv": iv, "rv_iv": _f(rws[0]["rv_iv_ratio"]),
            "hv_term": (hv5 / hv63) if (hv5 and hv63) else None,
            "t_et": t_et.hour + t_et.minute / 60.0,
            "gtbr": compute_gtbr(spot, iv, t_et.hour + t_et.minute / 60.0),
            "dag_thr": dag_thr,
        }
        walls = []
        for r, p in zip(near, prot):
            walls.append({
                "strike": _f(r["strike_futures"]), "dist_nq": _f(r["dist_nq"]) or 0,
                "net_gex": _f(r["net_gex"]) or 0, "abs_gex": _f(r["abs_gex"]),
                "net_dex": _f(r["net_dex"]) or 0, "net_dag": _f(r["net_dag"]),
                "total_oi": _f(r["total_oi"]), "protrusion": p,
                "call_oi": _f(r.get("call_oi")), "put_oi": _f(r.get("put_oi")),
                "strike_iv": _f(r.get("strike_iv")),
                "type": r["type"], "dt_et": t_et, "ctx": ctx,
            })
        result[ts] = walls
    return result


# ── outcome labeling against NQ 1m bars ───────────────────────────────────────
def label_wall(w, bars):
    """Return 1 (held) / 0 (broke) / None (never touched) over the hold window."""
    strike = w["strike"]
    above  = w["dist_nq"] > 0
    start  = w["dt_et"].replace(tzinfo=ET).astimezone(UTC)
    day_end = w["dt_et"].replace(hour=16, minute=0, second=0, tzinfo=ET).astimezone(UTC)

    win = bars[(bars.index >= start) & (bars.index <= day_end)]
    if win.empty:
        return None

    tol = strike * TOUCH_TOL
    touch_idx = None
    for ts, row in win.iterrows():
        hi, lo = float(row["High"]), float(row["Low"])
        if (above and hi >= strike - tol) or (not above and lo <= strike + tol):
            touch_idx = ts
            break
    if touch_idx is None:
        return None

    hold_end = touch_idx + timedelta(minutes=HOLD_WINDOW_MIN)
    after = bars[(bars.index > touch_idx) & (bars.index <= hold_end)]
    buf = strike * BREAK_BUFFER
    consec = 0
    for ts, row in after.iterrows():
        c = float(row["Close"])
        beyond = (c > strike + buf) if above else (c < strike - buf)
        consec = consec + 1 if beyond else 0
        if consec >= BREAK_CONFIRM:
            return 0  # confirmed break
    return 1          # held


def rank_auc(scores, labels):
    """Mann-Whitney AUC = P(score_held > score_broke)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="optional path to write per-wall labeled rows")
    args = ap.parse_args()

    import yfinance as yf
    bars = yf.download("NQ=F", period="7d", interval="1m", progress=False, auto_adjust=False)
    if isinstance(bars.columns, type(bars.columns)) and bars.columns.nlevels > 1:
        bars.columns = bars.columns.get_level_values(0)  # flatten ('High','NQ=F') -> 'High'
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize(UTC)
    else:
        bars.index = bars.index.tz_convert(UTC)
    bar_days = {d.astimezone(ET).date() for d in bars.index}

    # discover logged days
    listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", "origin/gex-snapshots"],
                            capture_output=True, text=True).stdout
    dates = sorted(l.split("levels_")[1].replace(".csv", "")
                   for l in listing.splitlines() if "levels_" in l)

    rows_out, scores, labels = [], [], []
    used_days = []
    for d in dates:
        dd = datetime.strptime(d, "%Y-%m-%d").date()
        if dd not in bar_days:
            continue  # outside yfinance's 1m window
        used_days.append(d)
        for ts, walls in load_day(d).items():
            for w in walls:
                y = label_wall(w, bars)
                if y is None:
                    continue
                R = hold_prob(w, w["ctx"])
                scores.append(R); labels.append(y)
                rows_out.append({"date": d, "time": ts[11:16], "strike": w["strike"],
                                 "dist_nq": round(w["dist_nq"], 1), "type": w["type"],
                                 "R": R, "held": y})

    # ── report ────────────────────────────────────────────────────────────────
    print("=" * 64)
    print("FORWARD TEST — mechanical hold_prob vs realized NQ 1m outcomes")
    print("=" * 64)
    print(f"Days labeled (in yfinance window): {', '.join(used_days) or 'none'}")
    n = len(labels)
    if n == 0:
        print("\nNo touched walls to label. Need denser snapshots or recent data.")
        return
    held = sum(labels)
    print(f"Labeled wall touches: {n}   |   overall hold rate: {held}/{n} = {held/n:.1%}")

    print("\nHold rate by predicted R bucket:")
    edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in edges:
        idx = [i for i, s in enumerate(scores) if lo <= s < hi]
        if not idx:
            print(f"  R {lo:.1f}-{hi if hi <= 1 else 1.0:.1f}:   n=0")
            continue
        hr = sum(labels[i] for i in idx) / len(idx)
        print(f"  R {lo:.1f}-{hi if hi <= 1 else 1.0:.1f}:   n={len(idx):<3}  hold={hr:.1%}")

    auc = rank_auc(scores, labels)
    print(f"\nRank AUC (R vs held): {auc:.3f}" if auc is not None
          else "\nRank AUC: undefined (need both held and broke examples)")
    print("  >0.5 = higher R really does hold more often (the model has signal)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            wri.writeheader(); wri.writerows(rows_out)
        print(f"\nWrote {len(rows_out)} labeled rows -> {args.csv}")


if __name__ == "__main__":
    main()
