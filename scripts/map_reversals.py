"""
map_reversals.py — accurate strike->futures reversal mapping.

Fixes the two errors from the ad-hoc analysis:
  1. Strike->NQ projection is NOT a fixed ratio. Mirrors the user's TradingView
     indicator: ratio = SMA(NQ_close / QQQ_close, 100) on EXTENDED-hours data,
     then nq_level = qqq_strike * smoothed_ratio. The ratio drifts ~41.0-41.2
     intraday; a fixed guess put strikes ~100 NQ pts off.
  2. A valid reversal = a swing of >= PCT percent (default 0.33% ≈ 100 NQ pts at
     current levels) off a level, measured relative to the running extreme so the
     bar is scale-invariant (a fixed point threshold drifts as price levels move
     and won't port across instruments). Detection runs on High/Low wicks over the
     FULL session (globex + pre/post), so premarket rejections (e.g. the ~07:30
     733 test) are included.

For every >=THRESHOLD pivot it finds the nearest projected strike AT THAT MINUTE.
Pivots that land near a strike are strike reversals; pivots with no strike within
--tol are flagged NO-STRIKE (blindspot / volume / cross-asset candidates).

Usage:
  python scripts/map_reversals.py                       # latest day
  python scripts/map_reversals.py --date 2026-06-04 --threshold 100 --tol 20
"""
import argparse, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import yfinance as yf
import pandas as pd, numpy as np


def load_aligned(days=7):
    """NQ globex 1m + QQQ pre/post 1m, aligned, with smoothed NQ/QQQ ratio."""
    nq = yf.download("NQ=F", period=f"{days}d", interval="1m", progress=False,
                     auto_adjust=False, prepost=True)
    qq = yf.download("QQQ", period=f"{days}d", interval="1m", progress=False,
                     auto_adjust=False, prepost=True)
    for x in (nq, qq):
        if x.columns.nlevels > 1:
            x.columns = x.columns.get_level_values(0)
    nq.index = nq.index.tz_convert("America/New_York")
    qq.index = qq.index.tz_convert("America/New_York")
    ratio = (nq["Close"] / qq["Close"].reindex(nq.index)).rename("ratio")
    # forward-fill QQQ gaps (NQ trades when QQQ doesn't), then smooth like the indicator
    ratio = ratio.ffill()
    sm = ratio.rolling(100, min_periods=20).mean().ffill().bfill().rename("sm_ratio")
    nq = nq.join(sm)
    return nq


def zigzag_hl(high, low, times, pct):
    """High/Low wick zigzag. A swing counts when price retraces >= pct (fraction,
    e.g. 0.0033) of the running extreme. Returns [(idx, price, 'HIGH'|'LOW')]."""
    h, l = high.values, low.values
    pivots = []
    direction = 0
    hi_p, hi_i = h[0], 0
    lo_p, lo_i = l[0], 0
    for i in range(1, len(h)):
        if h[i] > hi_p: hi_p, hi_i = h[i], i
        if l[i] < lo_p: lo_p, lo_i = l[i], i
        if direction >= 0 and (hi_p - l[i]) >= pct * hi_p:
            pivots.append((hi_i, hi_p, "HIGH")); direction = -1; lo_p, lo_i = l[i], i
        elif direction <= 0 and (h[i] - lo_p) >= pct * lo_p:
            pivots.append((lo_i, lo_p, "LOW")); direction = +1; hi_p, hi_i = h[i], i
    return pivots


def map_day(nq_all, date, pct, tol):
    d = nq_all[nq_all.index.date == date].dropna(subset=["High", "Low", "Close"])
    if len(d) < 50:
        return None
    rth = d.between_time("09:30", "16:00")
    o = rth["Open"].iloc[0] if len(rth) else d["Open"].iloc[0]
    approx_pts = pct * d["Close"].iloc[-1]
    print(f"\n{'='*72}\n{date}   NQ session {d.index.min():%H:%M}-{d.index.max():%H:%M} ET   "
          f"O {o:.0f}  H {d['High'].max():.0f}  L {d['Low'].min():.0f}  C {d['Close'].iloc[-1]:.0f}")
    print(f"ratio (smoothed NQ/QQQ): {d['sm_ratio'].min():.3f}–{d['sm_ratio'].max():.3f}")

    pivots = zigzag_hl(d["High"], d["Low"], d.index, pct)
    if not pivots:
        print(f"  no >= {pct*100:.2f}% reversals."); return []
    qmin = int(np.floor(d["Low"].min() / d["sm_ratio"].max()))
    qmax = int(np.ceil(d["High"].max() / d["sm_ratio"].min()))
    strikes = np.arange(qmin, qmax + 1, 1.0)

    print(f"\nVALID REVERSALS (>= {pct*100:.2f}% ≈ {approx_pts:.0f} NQ pts off the pivot):")
    print(f"{'time':>5} {'sess':>4} {'type':4} {'NQ wick':>8}  {'strike':>6} {'proj NQ':>8} {'dist':>5}  follow%")
    rows = []
    for k, (idx, price, kind) in enumerate(pivots):
        t = d.index[idx]
        r = d["sm_ratio"].iloc[idx]
        proj = strikes * r
        j = int(np.argmin(np.abs(proj - price)))
        dist = price - proj[j]
        nearest = strikes[j]
        # follow-through to next pivot (size of the leg leaving this pivot)
        follow = abs(pivots[k+1][1] - price) if k + 1 < len(pivots) else np.nan
        sess = "RTH" if t.time() >= pd.Timestamp("09:30").time() and t.time() <= pd.Timestamp("16:00").time() else "EXT"
        tag = f"{nearest:.0f}" if abs(dist) <= tol else "NO-STRIKE"
        flag = "" if abs(dist) <= tol else "  <-- not at a strike"
        fol = f"{follow/price*100:.2f}%" if not np.isnan(follow) else "(last)"
        print(f"{t:%H:%M} {sess:>4} {kind:4} {price:>8.0f}  {tag:>6} "
              f"{proj[j]:>8.0f} {dist:>+5.0f}  {fol}{flag}")
        rows.append(dict(date=str(date), time=f"{t:%H:%M}", session=sess, type=kind,
                         nq=round(price), strike=(round(nearest) if abs(dist) <= tol else None),
                         proj_nq=round(proj[j]), dist=round(dist),
                         follow_pt=(round(follow) if not np.isnan(follow) else None),
                         follow_pct=(round(follow / price * 100, 2) if not np.isnan(follow) else None)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: latest available)")
    ap.add_argument("--pct", type=float, default=0.33,
                    help="min swing as %% of price (default 0.33%% ≈ 100 NQ pts at current levels)")
    ap.add_argument("--tol", type=float, default=20.0, help="max NQ pts pivot->strike to count as a strike reversal")
    ap.add_argument("--days", type=int, default=7, help="yfinance lookback window")
    ap.add_argument("--csv")
    args = ap.parse_args()

    nq = load_aligned(args.days)
    all_days = sorted({d.date() for d in nq.index})
    days = [pd.Timestamp(args.date).date()] if args.date else [all_days[-1]]

    out = []
    for dt in days:
        rows = map_day(nq, dt, args.pct / 100.0, args.tol)
        if rows:
            out += rows
    if args.csv and out:
        pd.DataFrame(out).to_csv(args.csv, index=False)
        print(f"\nwrote {len(out)} reversals -> {args.csv}")


if __name__ == "__main__":
    main()
