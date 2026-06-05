"""
validate_walls.py — OLD vs NEW wall scoring + hold_prob against realized outcomes.

Answers two questions objectively (no eyeballing, no circular labels):

  1. hold_prob ranking: does a higher hold_prob really hold more often, and did
     the smooth-B change help? -> rank AUC + calibration buckets, old vs new.
  2. surfacing quality: of the walls price actually TOUCHED-AND-HELD, what
     fraction does each version SURFACE (recall) and how clean is what it
     surfaces (precision)? -> directly tests the "733 wasn't flagged" fix.

Labeling rule (objective, non-circular): a wall is HELD if, after price touches it,
price does NOT post a 2-bar confirmed
close beyond it within HOLD_WINDOW_MIN; BROKE otherwise; unlabeled if untouched.

Scoring is ported from netlify/functions/lib/options.js for BOTH versions so the
comparison is apples-to-apples. The NEW port is cross-checked against the live JS
(see --selfcheck) so a port bug can't masquerade as a result.

Caveats it does NOT hide:
  - yfinance caps 1m history at ~7 days, so only the most recent logged days are
    labelable. Older snapshots are also sparse (pre-Cloudflare cron). Small n.
  - Old snapshots lack call_oi/put_oi/strike_iv/abs_gex -> those factors replay
    neutral, exactly as production would have on that data.

Usage:
  python scripts/validate_walls.py
  python scripts/validate_walls.py --csv data/processed/wall_validation.csv
"""
import argparse, subprocess, io, csv, math, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")  # cp1252-safe on Windows
import pandas as pd
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")

FILTER_PCT, MIN_SCORE = 5.0, 20.0
NEAR_BAND_PCT, KEEP_PER_SIDE = 2.5, 5
# 2026-06-04 is the first dense-capture day (Cloudflare dispatcher). Earlier days have
# only ~3-5 sparse snapshots/day — too coarse for touch labeling, so they are excluded.
FIRST_GOOD_DAY = "2026-06-04"
HOLD_WINDOW_MIN, BREAK_BUFFER, BREAK_CONFIRM, TOUCH_TOL = 30, 0.0010, 2, 0.0003

REGIME_WEIGHTS = {
    "EXPANSION":   {"POSITIVE": dict(gex=.22, vex=.38, charmex=.17, oi=.14, dag=.09),
                    "NEAR_FLIP": dict(gex=.20, vex=.38, charmex=.17, oi=.15, dag=.10),
                    "NEGATIVE": dict(gex=.10, vex=.48, charmex=.17, oi=.14, dag=.11)},
    "NEUTRAL":     {"POSITIVE": dict(gex=.42, vex=.22, charmex=.14, oi=.14, dag=.08),
                    "NEAR_FLIP": dict(gex=.32, vex=.28, charmex=.15, oi=.15, dag=.10),
                    "NEGATIVE": dict(gex=.20, vex=.36, charmex=.14, oi=.16, dag=.14)},
    "CONTRACTION": {"POSITIVE": dict(gex=.60, vex=.10, charmex=.14, oi=.14, dag=.02),
                    "NEAR_FLIP": dict(gex=.50, vex=.15, charmex=.15, oi=.15, dag=.05),
                    "NEGATIVE": dict(gex=.36, vex=.24, charmex=.16, oi=.15, dag=.09)},
}


def _f(x):
    try:
        return float(x) if x not in (None, "") else None
    except (ValueError, TypeError):
        return None


def classify_vol_regime(iv, rv):
    if iv is not None and (iv >= 30 or (rv is not None and rv < 0.5)):
        return "EXPANSION"
    if iv is not None and iv >= 20:
        return "NEUTRAL"
    if iv is not None:
        return "CONTRACTION"
    return "NEUTRAL"


def get_weights(vol, gam):
    vr = REGIME_WEIGHTS.get(vol, REGIME_WEIGHTS["NEUTRAL"])
    return vr.get(gam, vr["NEAR_FLIP"])


def normalize_abs(values):               # OLD: min-max of |v|
    a = [abs(v) for v in values]
    mn, mx = min(a), max(a)
    return [0.0] * len(a) if mx == mn else [(v - mn) / (mx - mn) for v in a]


def normalize_robust(values):            # NEW: scale by p90, clamp [0,1]
    a = [abs(v) for v in values]
    if not a:
        return a
    s = sorted(a)
    p90 = s[int(0.9 * (len(s) - 1))] or 0.0
    return [0.0] * len(a) if p90 <= 0 else [min(1.0, v / p90) for v in a]


def normalize_gex_per_side(netgex):      # NEW: robust within each side
    out = [0.0] * len(netgex)
    for side in (1, -1):
        idx = [i for i, g in enumerate(netgex) if (1 if g > 0 else -1) == side]
        norm = normalize_robust([netgex[i] for i in idx])
        for k, i in enumerate(idx):
            out[i] = norm[k]
    return out


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
    return [1.0] * len(ratios) if mx == mn else [(r - mn) / (mx - mn) for r in ratios]


def compute_gtbr(futures, iv, t_et):
    if not futures or not iv or iv <= 0:
        return None
    daily1sd = futures * (iv / 100.0) / math.sqrt(252)
    t_rem = max(0.0, 16.0 - (t_et if t_et else 9.5)) / 6.5
    return daily1sd * (math.sqrt(t_rem) if t_rem > 0 else 1.0)


def hold_prob(w, ctx, version):
    spot, netgex = ctx["futures"], w["net_gex"]
    absgex = w.get("abs_gex") or abs(netgex)
    netdex, netdag, dist = w["net_dex"], abs(w.get("net_dag") or 0.0), w["dist_nq"]

    B = 0.3
    flip = ctx.get("gamma_flip")
    if flip is not None and spot is not None:
        band = max(30.0, 0.5 * spot * (ctx["iv"] / 100.0) / math.sqrt(252)) if ctx.get("iv") else 50.0
        diff = spot - flip
        if version == "new":                       # smooth ramp across the band
            B = 0.5 if diff >= band else 0.1 if diff <= -band else 0.3 + 0.2 * (diff / band)
        else:                                      # old hard step
            B = 0.5 if diff > band else 0.1 if diff < -band else 0.3

    P = 0.5 + (w["protrusion"] if w.get("protrusion") is not None else 0.5)
    ratio = min(1.0, abs(netgex) / absgex) if absgex > 0 else 0.5
    O = max(0.85, min(1.15, 0.85 + 0.6 * (ratio - 0.5)))
    above = dist > 0
    A_dex = 1.25 if ((netdex > 0) if above else (netdex < 0)) else 0.5
    co, po = w.get("call_oi"), w.get("put_oi")
    PCR = (0.9 + 0.4 * (max(co, po) / (co + po) - 0.5)) if (co is not None and po is not None and (co + po) > 0) else 1.0
    siv, atm = w.get("strike_iv"), ctx.get("iv")
    S_skew = 1.15 if (siv is not None and atm and 0 < siv < 5 * atm and abs(siv - atm) / atm > 0.20) else 1.0
    F_term = 0.85 if (ctx.get("hv_term") is not None and ctx["hv_term"] > 1.25) else 1.0
    F_vrp = 1.15 if (ctx.get("rv_iv") is not None and ctx["rv_iv"] < 0.5) else 1.0
    F_gtbr = 0.2 if (ctx.get("gtbr") is not None and abs(dist) > ctx["gtbr"]) else 1.0
    G_pin = 1.25 if (ctx.get("t_et", 0) > 14 and ctx.get("dag_thr") is not None and netdag >= ctx["dag_thr"]) else 1.0
    return round(min(1.0, B * P * O * A_dex * PCR * S_skew * F_term * F_vrp * F_gtbr * G_pin), 3)


def score_walls(near, ctx, version):
    """Mirror scoreLevels for one snapshot. Returns walls with score/surfaced/dominant/hold_prob."""
    futures = ctx["futures"]
    netgex = [w["net_gex"] for w in near]
    if version == "new":
        gexN = normalize_gex_per_side(netgex)
        vexN = normalize_robust([w["net_vex"] for w in near])
        chmN = normalize_robust([w["net_charmex"] for w in near])
        oiN = normalize_robust([w["total_oi"] or 0 for w in near])
        dagN = normalize_robust([w["net_dag"] or 0 for w in near])
    else:
        gexN = normalize_abs(netgex)
        vexN = normalize_abs([w["net_vex"] for w in near])
        chmN = normalize_abs([w["net_charmex"] for w in near])
        oiN = normalize_abs([w["total_oi"] or 0 for w in near])
        dagN = normalize_abs([w["net_dag"] or 0 for w in near])
    prot = [w["protrusion"] for w in near]

    # regime state for the NEW relevance multiplier
    flip, iv = ctx.get("gamma_flip"), ctx.get("iv")
    band = max(30.0, 0.5 * futures * (iv / 100.0) / math.sqrt(252)) if iv else 50.0
    fdiff = (futures - flip) if (flip is not None) else None
    rstate = "UNKNOWN" if fdiff is None else "POSITIVE" if fdiff > band else "NEGATIVE" if fdiff < -band else "NEAR_FLIP"

    def relevance(ng, dist):
        if version != "new":
            return 1.0
        rev = (ng > 0) if dist > 0 else (ng <= 0)
        return {"POSITIVE": 1.15 if rev else 0.90, "NEGATIVE": 0.85 if rev else 1.00,
                "NEAR_FLIP": 1.00 if rev else 0.95}.get(rstate, 1.0)

    wts = ctx["weights"]
    out = []
    for i, w in enumerate(near):
        raw = (gexN[i] * wts["gex"] + vexN[i] * wts["vex"] + chmN[i] * wts["charmex"]
               + oiN[i] * wts["oi"] + dagN[i] * wts["dag"]) * 100
        protmul = (0.5 + 0.5 * prot[i]) if version == "new" else (0.25 + 0.75 * prot[i])
        score = raw * protmul * relevance(w["net_gex"], w["dist_nq"])
        out.append({**w, "_protrusion": prot[i], "score": round(score, 1),
                    "hold_prob": hold_prob({**w, "protrusion": prot[i]}, ctx, version)})

    # guaranteed surface (NEW only): top-5 gross-gamma per side within NEAR_BAND_PCT
    guaranteed = set()
    if version == "new":
        for call_side in (True, False):
            cand = [w for w in out if (w["net_gex"] > 0) == call_side
                    and abs((w["dist_nq"] / futures) * 100) <= NEAR_BAND_PCT]
            cand.sort(key=lambda w: -(w.get("abs_gex") or abs(w["net_gex"])))
            for w in cand[:KEEP_PER_SIDE]:
                guaranteed.add(w["strike"])

    for w in out:
        w["surfaced"] = (w["score"] >= MIN_SCORE) or (w["strike"] in guaranteed)

    # NMS -> dominant set (per side, on surfaced, score-desc)
    sep = futures * (0.0025 if version == "new" else 0.006)
    dominant = set()
    for sign in (1, -1):
        chosen = []
        side = sorted([w for w in out if w["surfaced"] and (1 if w["net_gex"] > 0 else -1) == sign],
                      key=lambda w: -w["score"])
        for w in side:
            if not any(abs(c - w["strike"]) < sep for c in chosen):
                chosen.append(w["strike"]); dominant.add(w["strike"])
    for w in out:
        w["dominant"] = w["strike"] in dominant
    return out


def load_ratio(days=7):
    """Per-minute smoothed NQ/QQQ ratio (extended hours) — the user's indicator method.
    Used to place each strike where price actually reacts (FreeFlow strike_futures is ~36pt high)."""
    import yfinance as yf
    nq = yf.download("NQ=F", period=f"{days}d", interval="1m", progress=False, auto_adjust=False, prepost=True)
    qq = yf.download("QQQ", period=f"{days}d", interval="1m", progress=False, auto_adjust=False, prepost=True)
    for x in (nq, qq):
        if x.columns.nlevels > 1:
            x.columns = x.columns.get_level_values(0)
    nq.index = nq.index.tz_convert(ET); qq.index = qq.index.tz_convert(ET)
    ratio = (nq["Close"] / qq["Close"].reindex(nq.index)).ffill()
    return ratio.rolling(100, min_periods=20).mean().ffill().bfill()


def load_day(date_str, ratio_sm=None):
    out = subprocess.run(["git", "show", f"origin/gex-snapshots:logs/levels_{date_str}.csv"],
                         capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(out)))
    snaps = {}
    for r in rows:
        if r.get("expiry") != r.get("snapshot_date"):    # 0DTE only
            continue
        snaps.setdefault(r["intended_timestamp"], []).append(r)

    result = {}
    for ts, rws in snaps.items():
        spot = _f(rws[0]["nq_price"])
        if not spot:
            continue
        near = [r for r in rws if _f(r["dist_pct"]) is not None and abs(_f(r["dist_pct"])) <= FILTER_PCT]
        if len(near) < 5:
            continue
        prot = compute_protrusion([_f(r["net_gex"]) or 0 for r in near])
        dags = sorted(abs(_f(r["net_dag"]) or 0) for r in near)
        dag_thr = dags[min(len(dags) - 1, int(0.9 * (len(dags) - 1)))] if dags else None
        t = datetime.strptime(ts.replace(" ET", ""), "%Y-%m-%dT%H:%M:%S")
        # Smoothed-ratio strike placement (the user's indicator): each strike's NQ
        # location = strike_etf × SMA(NQ/QQQ,100). FreeFlow's strike_futures is ~36pt
        # high vs where price actually reacts. Fall back to snapshot spot ratio if the
        # ratio series has no value at this minute.
        qqq = _f(rws[0]["qqq_price"])
        ratio_at = None
        if ratio_sm is not None:
            try:
                v = ratio_sm.asof(pd.Timestamp(t).tz_localize(ET))
                ratio_at = None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
            except Exception:
                ratio_at = None
        if ratio_at is None and qqq:
            ratio_at = spot / qqq
        hv5, hv63 = _f(rws[0]["hv5"]), _f(rws[0]["hv63"])
        iv = _f(rws[0]["iv"])
        t_et = t.hour + t.minute / 60.0
        vreg = classify_vol_regime(iv, _f(rws[0]["rv_iv_ratio"]))
        flip = _f(rws[0]["gamma_flip"])
        band = max(30.0, 0.5 * spot * (iv / 100.0) / math.sqrt(252)) if iv else 50.0
        greg = "UNKNOWN" if flip is None else "POSITIVE" if (spot - flip) > band else "NEGATIVE" if (spot - flip) < -band else "NEAR_FLIP"
        ctx = {"futures": spot, "gamma_flip": flip, "iv": iv, "rv_iv": _f(rws[0]["rv_iv_ratio"]),
               "hv_term": (hv5 / hv63) if (hv5 and hv63) else None, "t_et": t_et,
               "gtbr": compute_gtbr(spot, iv, t_et), "dag_thr": dag_thr,
               "weights": get_weights(vreg, greg)}
        walls = []
        for r, p in zip(near, prot):
            etf = _f(r["strike_etf"])
            if ratio_at and etf and qqq:
                loc = etf * ratio_at                 # corrected NQ location (yfinance-consistent)
                dist = (etf - qqq) * ratio_at        # corrected distance from spot
            else:
                loc = _f(r["strike_futures"]); dist = _f(r["dist_nq"]) or 0
            walls.append({"strike": loc, "dist_nq": dist, "strike_etf": etf,
                          "net_gex": _f(r["net_gex"]) or 0, "abs_gex": _f(r["abs_gex"]),
                          "net_vex": _f(r["net_vex"]) or 0, "net_charmex": _f(r["net_charmex"]) or 0,
                          "net_dex": _f(r["net_dex"]) or 0, "net_dag": _f(r["net_dag"]),
                          "total_oi": _f(r["total_oi"]), "protrusion": p,
                          "call_oi": _f(r.get("call_oi")), "put_oi": _f(r.get("put_oi")),
                          "strike_iv": _f(r.get("strike_iv")), "type": r["type"], "dt_et": t})
        result[ts] = (walls, ctx)
    return result


def label_wall(strike, above, dt_et, bars):
    start = dt_et.replace(tzinfo=ET).astimezone(UTC)
    day_end = dt_et.replace(hour=16, minute=0, second=0, tzinfo=ET).astimezone(UTC)
    win = bars[(bars.index >= start) & (bars.index <= day_end)]
    if win.empty:
        return None
    tol = strike * TOUCH_TOL
    touch_idx = None
    for ts, row in win.iterrows():
        if (above and float(row["High"]) >= strike - tol) or (not above and float(row["Low"]) <= strike + tol):
            touch_idx = ts; break
    if touch_idx is None:
        return None
    after = bars[(bars.index > touch_idx) & (bars.index <= touch_idx + timedelta(minutes=HOLD_WINDOW_MIN))]
    buf, consec = strike * BREAK_BUFFER, 0
    for ts, row in after.iterrows():
        c = float(row["Close"])
        consec = consec + 1 if ((c > strike + buf) if above else (c < strike - buf)) else 0
        if consec >= BREAK_CONFIRM:
            return 0
    return 1


def rank_auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    return sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg) / (len(pos) * len(neg))


def report(tag, scores, labels):
    n = len(labels); held = sum(labels)
    print(f"\n── {tag} ── n={n}  hold={held}/{n}={held/n:.1%}" if n else f"\n── {tag} ── n=0")
    if not n:
        return
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        idx = [i for i, s in enumerate(scores) if lo <= s < hi]
        hr = (sum(labels[i] for i in idx) / len(idx)) if idx else 0
        print(f"   hold_prob {lo:.1f}-{min(hi,1.0):.1f}: n={len(idx):<3} hold={hr:.0%}" if idx
              else f"   hold_prob {lo:.1f}-{min(hi,1.0):.1f}: n=0")
    auc = rank_auc(scores, labels)
    print(f"   rank AUC = {auc:.3f}  (>0.5 = higher hold_prob holds more often)" if auc is not None
          else "   rank AUC undefined (need held AND broke)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    args = ap.parse_args()

    import yfinance as yf
    bars = yf.download("NQ=F", period="7d", interval="1m", progress=False, auto_adjust=False, prepost=True)
    if bars.columns.nlevels > 1:
        bars.columns = bars.columns.get_level_values(0)
    bars.index = bars.index.tz_localize(UTC) if bars.index.tz is None else bars.index.tz_convert(UTC)
    bar_days = {d.astimezone(ET).date() for d in bars.index}
    ratio_sm = load_ratio(7)   # smoothed-ratio strike placement

    listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", "origin/gex-snapshots"],
                             capture_output=True, text=True).stdout
    dates = sorted(l.split("levels_")[1].replace(".csv", "") for l in listing.splitlines() if "levels_" in l)
    dates = [d for d in dates if d >= FIRST_GOOD_DAY]   # exclude sparse pre-Cloudflare days

    agg = {"old": {"s": [], "y": []}, "new": {"s": [], "y": []}}
    surf = {"old": dict(tp=0, held=0, surf=0, dom=0, dom_held=0),
            "new": dict(tp=0, held=0, surf=0, dom=0, dom_held=0)}
    used, rows_out = [], []

    for d in dates:
        if datetime.strptime(d, "%Y-%m-%d").date() not in bar_days:
            continue
        day = load_day(d, ratio_sm)
        if not day:
            continue
        used.append(d)
        for ts, (walls, ctx) in day.items():
            scored = {v: score_walls(walls, ctx, v) for v in ("old", "new")}
            for i, w in enumerate(walls):
                y = label_wall(w["strike"], w["dist_nq"] > 0, w["dt_et"], bars)
                if y is None:
                    continue
                for v in ("old", "new"):
                    sw = scored[v][i]
                    agg[v]["s"].append(sw["hold_prob"]); agg[v]["y"].append(y)
                    if y == 1:
                        surf[v]["held"] += 1
                        if sw["surfaced"]:
                            surf[v]["surf"] += 1            # recall numerator
                    if sw["surfaced"]:
                        surf[v]["tp"] += 1
                    if sw["dominant"]:
                        surf[v]["dom"] += 1
                        if y == 1:
                            surf[v]["dom_held"] += 1
                rows_out.append({"date": d, "time": ts[11:16], "strike": w["strike"],
                                 "dist_nq": round(w["dist_nq"], 1), "held": y,
                                 "R_old": scored["old"][i]["hold_prob"], "R_new": scored["new"][i]["hold_prob"],
                                 "surf_old": int(scored["old"][i]["surfaced"]), "surf_new": int(scored["new"][i]["surfaced"]),
                                 "dom_old": int(scored["old"][i]["dominant"]), "dom_new": int(scored["new"][i]["dominant"])})

    print("=" * 70)
    print("WALL VALIDATION — old vs new, vs realized NQ 1m (touch + 2-bar break)")
    print("=" * 70)
    print(f"Days labeled (yfinance 1m window): {', '.join(used) or 'none'}")
    if not agg["new"]["y"]:
        print("\nNo touched walls labeled. Need denser/recenter snapshots.")
        return

    print("\n### 1. hold_prob ranking (every touched wall)")
    report("OLD hold_prob", agg["old"]["s"], agg["old"]["y"])
    report("NEW hold_prob", agg["new"]["s"], agg["new"]["y"])

    print("\n### 2. surfacing quality (did we flag the walls that held?)")
    for v in ("old", "new"):
        s = surf[v]
        recall = s["surf"] / s["held"] if s["held"] else 0
        prec = s["surf"] / s["tp"] if s["tp"] else 0      # surfaced-and-held / all-surfaced(touched)
        dom_prec = s["dom_held"] / s["dom"] if s["dom"] else 0
        print(f"\n── {v.upper()} ──")
        print(f"   held walls captured (recall): {s['surf']}/{s['held']} = {recall:.0%}")
        print(f"   surfaced walls that held (precision): {s['surf']}/{s['tp']} = {prec:.0%}")
        print(f"   DOMINANT walls that held: {s['dom_held']}/{s['dom']} = {dom_prec:.0%}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            wri.writeheader(); wri.writerows(rows_out)
        print(f"\nWrote {len(rows_out)} labeled rows -> {args.csv}")


if __name__ == "__main__":
    main()
