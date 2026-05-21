"""
FreeFlow Level Calculator — NQ/QQQ Reversal Level Identification
================================================================
Fetches pre-computed Greeks exposure data from FreeFlow API and
scores each strike as a high-probability NQ reversal level using
GEX, VEX, CharmEX, DEX, and vol regime weighting.

Project: h41_bias&regime_engine — Layer 2 (Positioning)
Place in: C:\\Users\\asare\\Downloads\\h41_bias&regime_engine\\
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import sys
import os
import json

# ============================================================
# AUTHENTICATION
# ============================================================

COOKIES = {
    'ff_session': 'j_g_U67Ox9np5aXxK5YKHkVnQt41_RnyUEtmo7Bw7q0',
}

HEADERS = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Referer': 'https://www.free-flow.site/?auth=success',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}

BASE_URL = "https://www.free-flow.site/api"
SYMBOL   = "QQQ"

# ============================================================
# CONFIGURATION
# ============================================================

FILTER_PCT = 5.0   # Only show levels within this % of current NQ price
MIN_SCORE  = 20.0  # Minimum composite score to include in output

# ============================================================
# DATA FETCHING
# ============================================================

def make_session():
    s = requests.Session()
    s.cookies.update(COOKIES)
    s.headers.update(HEADERS)
    return s


def fetch_futures_levels(session, symbol=SYMBOL, exp=None):
    if exp is None:
        exp = date.today().strftime("%Y-%m-%d")
    url = f"{BASE_URL}/futures-levels?symbol={symbol}&exp={exp}"
    r = session.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_cone(session, symbol=SYMBOL):
    """
    Vol cone: percentile distribution of HV across lookback periods.
    Returns current HV5/10/21/63 and their percentile ranks.
    """
    url = f"{BASE_URL}/vol/cone?symbol={symbol}"
    r = session.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()

    cone = data.get('cone', {})
    return {
        'hv5':    cone.get('5',  {}).get('current', 17.0),
        'hv10':   cone.get('10', {}).get('current', 18.0),
        'hv21':   cone.get('21', {}).get('current', 17.0),
        'hv63':   cone.get('63', {}).get('current', 19.0),
        'rank21': cone.get('21', {}).get('pct_rank', 50.0),
    }


def fetch_realized(session, symbol=SYMBOL):
    """
    Realized vol endpoint — contains current_iv and rv_iv_ratio.
    """
    url = f"{BASE_URL}/vol/realized?symbol={symbol}"
    r = session.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def get_vol_context(session):
    """
    Combine cone + realized into a single vol context dict.
    Falls back gracefully if either endpoint fails.
    """
    ctx = {
        'hv5':        17.0,
        'hv10':       18.0,
        'hv21':       17.0,
        'hv63':       19.0,
        'rank21':     50.0,
        'current_iv': 37.0,
        'rv_iv_ratio': 0.46,
    }

    try:
        cone = fetch_cone(session)
        ctx.update(cone)
    except Exception as e:
        print(f"  Warning: cone fetch failed ({e})")

    try:
        realized = fetch_realized(session)
        # Try common field names
        ctx['current_iv']  = (realized.get('current_iv')
                               or realized.get('iv')
                               or realized.get('implied_vol')
                               or ctx['current_iv'])
        ctx['rv_iv_ratio'] = (realized.get('rv_iv_ratio')
                               or realized.get('ratio')
                               or ctx['rv_iv_ratio'])
        # HV from realized if better than cone
        ctx['hv21'] = realized.get('hv21') or ctx['hv21']
    except Exception as e:
        print(f"  Warning: realized fetch failed ({e})")

    return ctx


def get_trading_days(n=5):
    days = []
    d = date.today()
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days

# ============================================================
# AGGREGATION
# ============================================================

def aggregate_exposures(fl_data):
    """
    Aggregate all per-option exposures by NQ strike.

    FreeFlow fields (already OI-weighted per option):
      gex      net gamma exposure  (+ = call wall, - = put wall)
      ag       absolute gamma
      vex      vanna exposure      (vol-sensitivity of delta hedging)
      charmex  charm exposure      (time-decay drift in hedging)
      dex      delta exposure
      vegaex   vega exposure
      dag      delta-adjusted gamma
    """
    rows = fl_data.get('rows', [])
    if not rows:
        raise ValueError("No rows in futures-levels response.")

    df = pd.DataFrame(rows)

    if 'strike_futures' not in df.columns:
        ratio = fl_data.get('ratio', 41.14)
        df['strike_futures'] = (df['strike_etf'] * ratio).round(1)

    agg = df.groupby('strike_futures').agg(
        strike_etf  = ('strike_etf',  'first'),
        net_gex     = ('gex',         'sum'),
        abs_gex     = ('ag',          'sum'),
        net_vex     = ('vex',         'sum'),
        net_charmex = ('charmex',     'sum'),
        net_dex     = ('dex',         'sum'),
        net_vegaex  = ('vegaex',      'sum'),
        net_dag     = ('dag',         'sum'),
        total_oi    = ('oi',          'sum'),
    ).reset_index()

    return agg


def aggregate_multi_expiry(fl_datasets):
    dfs = []
    for fl_data in fl_datasets:
        rows = fl_data.get('rows', [])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if 'strike_futures' not in df.columns:
            ratio = fl_data.get('ratio', 41.14)
            df['strike_futures'] = (df['strike_etf'] * ratio).round(1)
        dfs.append(df)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)

    agg = combined.groupby('strike_futures').agg(
        strike_etf  = ('strike_etf',  'first'),
        net_gex     = ('gex',         'sum'),
        abs_gex     = ('ag',          'sum'),
        net_vex     = ('vex',         'sum'),
        net_charmex = ('charmex',     'sum'),
        net_dex     = ('dex',         'sum'),
        net_vegaex  = ('vegaex',      'sum'),
        net_dag     = ('dag',         'sum'),
        total_oi    = ('oi',          'sum'),
    ).reset_index()

    return agg

# ============================================================
# VOL REGIME
# ============================================================

def classify_regime(ctx):
    """
    Classify vol regime and return scoring weights.

    CONTRACTION (IV < 20%):
      Gamma walls are sticky. GEX dominates.

    NEUTRAL (20-30%):
      Mixed. GEX and VEX both matter.

    EXPANSION (IV > 30% or RV/IV < 0.5):
      Vol-driven flows overwhelm gamma hedging. VEX dominates.
    """
    iv    = ctx.get('current_iv',  37.0)
    rv_iv = ctx.get('rv_iv_ratio', 0.46)

    if iv >= 30 or rv_iv < 0.5:
        regime  = "EXPANSION"
        weights = {'gex': 0.20, 'vex': 0.38, 'charmex': 0.17, 'oi': 0.15, 'dag': 0.10}
    elif iv >= 20:
        regime  = "NEUTRAL"
        weights = {'gex': 0.32, 'vex': 0.28, 'charmex': 0.15, 'oi': 0.15, 'dag': 0.10}
    else:
        regime  = "CONTRACTION"
        weights = {'gex': 0.50, 'vex': 0.15, 'charmex': 0.15, 'oi': 0.15, 'dag': 0.05}

    return regime, weights

# ============================================================
# SCORING
# ============================================================

def normalize(series):
    s = series.abs()
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (s - mn) / (mx - mn)


def score_levels(agg, weights, futures_price, filter_pct=FILTER_PCT):
    agg = agg.copy()
    agg['dist_nq']  = agg['strike_futures'] - futures_price
    agg['dist_pct'] = (agg['dist_nq'] / futures_price) * 100

    nearby = agg[agg['dist_pct'].abs() <= filter_pct].copy()
    if nearby.empty:
        return nearby

    nearby['gex_n']     = normalize(nearby['net_gex'])
    nearby['vex_n']     = normalize(nearby['net_vex'])
    nearby['charmex_n'] = normalize(nearby['net_charmex'])
    nearby['oi_n']      = normalize(nearby['total_oi'])
    nearby['dag_n']     = normalize(nearby['net_dag'])

    nearby['score'] = (
        nearby['gex_n']     * weights['gex']     +
        nearby['vex_n']     * weights['vex']      +
        nearby['charmex_n'] * weights['charmex']  +
        nearby['oi_n']      * weights['oi']       +
        nearby['dag_n']     * weights['dag']
    ) * 100

    def classify(row):
        g        = row['net_gex']
        v        = row['net_vex']
        vol_sens = abs(v) / (abs(g) + 1e-9)
        base     = "CALL WALL" if g > 0 else "PUT WALL"
        tag      = " + VOL SENSITIVE" if vol_sens > 2.0 else ""
        return base + tag

    nearby['type'] = nearby.apply(classify, axis=1)
    nearby = nearby[nearby['score'] >= MIN_SCORE]
    nearby = nearby.sort_values('score', ascending=False).reset_index(drop=True)
    return nearby

# ============================================================
# OUTPUT
# ============================================================

def print_output(nearby, ctx, regime, futures_price, spot_etf, exp, multi=False):
    iv    = ctx.get('current_iv',  '?')
    rv_iv = ctx.get('rv_iv_ratio', '?')
    hv21  = ctx.get('hv21',        '?')
    bar   = "=" * 85

    if isinstance(iv, float):
        iv_premium = f"  |  IV Premium: {(1/rv_iv - 1)*100:.0f}% above RV" if isinstance(rv_iv, float) and rv_iv > 0 else ""
        iv_str     = f"{iv:.1f}%"
        rv_str     = f"{rv_iv:.3f}" if isinstance(rv_iv, float) else str(rv_iv)
        hv_str     = f"{hv21:.1f}%" if isinstance(hv21, float) else str(hv21)
    else:
        iv_str, rv_str, hv_str, iv_premium = str(iv), str(rv_iv), str(hv21), ""

    mode = "MULTI-EXPIRY" if multi else f"EXP: {exp}"

    print(f"\n{bar}")
    print(f"  FREEFLOW LEVEL CALCULATOR  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {mode}  |  QQQ: ${spot_etf:.2f}  |  NQ: {futures_price:.1f}")
    print(f"  Regime: {regime}  |  IV: {iv_str}  |  RV/IV: {rv_str}  |  HV21: {hv_str}{iv_premium}")
    print(bar)

    if nearby is None or nearby.empty:
        print("  No significant levels found in range.\n")
        return

    print(f"\n  {'NQ':>10}  {'QQQ':>6}  {'DIST':>7}  {'SCORE':>5}  {'TYPE':<30}  {'NET GEX':>13}  {'NET VEX':>13}  {'OI':>5}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*30}  {'-'*13}  {'-'*13}  {'-'*5}")

    for _, row in nearby.head(25).iterrows():
        dist_str = f"{row['dist_nq']:+.0f}"
        atm      = " ◄" if abs(row['dist_nq']) < 50 else ""
        print(
            f"  {row['strike_futures']:>10.1f}"
            f"  {row['strike_etf']:>6.1f}"
            f"  {dist_str:>7}"
            f"  {row['score']:>5.1f}"
            f"  {row['type']:<30}"
            f"  {row['net_gex']:>13,.0f}"
            f"  {row['net_vex']:>13,.0f}"
            f"  {row['total_oi']:>5.0f}"
            f"{atm}"
        )

    print(f"\n  ── SUPPORT (PUT WALLS) ──────────────────────────────────────────")
    for _, row in nearby[nearby['net_gex'] < 0].head(5).iterrows():
        print(f"  NQ {row['strike_futures']:.1f}  |  Score: {row['score']:.0f}"
              f"  |  GEX: {row['net_gex']:,.0f}"
              f"  |  VEX: {row['net_vex']:,.0f}"
              f"  |  OI: {row['total_oi']:.0f}")

    print(f"\n  ── RESISTANCE (CALL WALLS) ──────────────────────────────────────")
    for _, row in nearby[nearby['net_gex'] > 0].head(5).iterrows():
        print(f"  NQ {row['strike_futures']:.1f}  |  Score: {row['score']:.0f}"
              f"  |  GEX: {row['net_gex']:,.0f}"
              f"  |  VEX: {row['net_vex']:,.0f}"
              f"  |  OI: {row['total_oi']:.0f}")

    print(f"\n{bar}\n")


def export_json(nearby, ctx, regime, futures_price, spot_etf, output_path=None):
    """Write levels_data.json consumed by the dashboard Levels tab."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'levels_data.json')

    now_et = datetime.now().strftime('%Y-%m-%d %H:%M ET')

    levels = []
    if nearby is not None and not nearby.empty:
        for _, row in nearby.iterrows():
            levels.append({
                'strike_futures': round(float(row['strike_futures']), 1),
                'strike_etf':     round(float(row['strike_etf']), 2),
                'dist_nq':        round(float(row['dist_nq']), 1),
                'score':          round(float(row['score']), 1),
                'type':           str(row['type']),
                'net_gex':        int(row['net_gex']),
                'net_vex':        int(row['net_vex']),
                'net_charmex':    int(row.get('net_charmex', 0)),
                'total_oi':       int(row['total_oi']),
            })

    payload = {
        'updated':     now_et,
        'nq_price':    round(float(futures_price), 1),
        'qqq_price':   round(float(spot_etf), 2),
        'regime':      regime,
        'iv':          round(float(ctx.get('current_iv',  0)), 1),
        'rv_iv_ratio': round(float(ctx.get('rv_iv_ratio', 0)), 3),
        'hv21':        round(float(ctx.get('hv21',         0)), 1),
        'levels':      levels,
    }

    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {output_path}  ({len(levels)} levels)")
    return output_path


def export_csv(nearby, exp):
    if nearby is None or nearby.empty:
        return
    fname = f"levels_{exp}_{datetime.now().strftime('%H%M')}.csv"
    cols  = ['strike_futures', 'strike_etf', 'dist_nq', 'dist_pct',
             'score', 'type', 'net_gex', 'net_vex', 'net_charmex',
             'net_dex', 'net_dag', 'total_oi']
    nearby[cols].to_csv(fname, index=False)
    print(f"  Saved: {fname}")

# ============================================================
# MAIN ENTRY POINTS
# ============================================================

def run_single(exp=None):
    if exp is None:
        exp = date.today().strftime("%Y-%m-%d")

    print(f"\nFetching FreeFlow data — {SYMBOL} | exp={exp}...")
    session = make_session()

    try:
        fl_data = fetch_futures_levels(session, exp=exp)
    except requests.HTTPError as e:
        print(f"\nERROR: {e}")
        if '401' in str(e) or '403' in str(e):
            print("Authentication failed. Update your COOKIES dict.")
        return None, None

    ctx              = get_vol_context(session)
    futures_price    = fl_data.get('futures_price', 0)
    spot_etf         = fl_data.get('etf_spot', 0)
    agg              = aggregate_exposures(fl_data)
    regime, weights  = classify_regime(ctx)
    nearby           = score_levels(agg, weights, futures_price)

    print_output(nearby, ctx, regime, futures_price, spot_etf, exp)
    try:
        export_json(nearby, ctx, regime, futures_price, spot_etf)
    except Exception as e:
        print(f"  Warning: failed to write levels_data.json ({e})")
    return nearby, fl_data


def run_multi(n_expiries=3):
    expiries = get_trading_days(n_expiries)
    print(f"\nFetching multi-expiry: {expiries}")

    session    = make_session()
    datasets   = []
    first_meta = None

    for exp in expiries:
        try:
            fl_data = fetch_futures_levels(session, exp=exp)
            datasets.append(fl_data)
            if first_meta is None:
                first_meta = fl_data
            print(f"  ✓ {exp}")
        except Exception as e:
            print(f"  ✗ {exp}: {e}")

    if not datasets or first_meta is None:
        print("No data retrieved.")
        return None

    ctx             = get_vol_context(session)
    futures_price   = first_meta.get('futures_price', 0)
    spot_etf        = first_meta.get('etf_spot', 0)
    agg             = aggregate_multi_expiry(datasets)
    regime, weights = classify_regime(ctx)
    nearby          = score_levels(agg, weights, futures_price)

    print_output(nearby, ctx, regime, futures_price, spot_etf,
                 exp=expiries[0], multi=True)
    try:
        export_json(nearby, ctx, regime, futures_price, spot_etf)
    except Exception as e:
        print(f"  Warning: failed to write levels_data.json ({e})")
    return nearby

# ============================================================
# LIVE MODE
# ============================================================

def run_live(exp=None, interval=60, multi=False, n=3):
    clear = 'cls' if os.name == 'nt' else 'clear'
    print(f"\nLIVE MODE — refreshing every {interval}s. Ctrl+C to stop.\n")

    while True:
        os.system(clear)
        try:
            if multi:
                run_multi(n_expiries=n)
            else:
                run_single(exp=exp)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"\nError: {e} — retrying in {interval}s...")

        for remaining in range(interval, 0, -1):
            sys.stdout.write(f"\r  Next refresh in {remaining:3d}s...  ")
            sys.stdout.flush()
            try:
                import time
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopped.")
                return

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FreeFlow NQ Level Calculator")
    parser.add_argument("--exp",      type=str,  default=None)
    parser.add_argument("--multi",    action="store_true")
    parser.add_argument("--n",        type=int,  default=3)
    parser.add_argument("--csv",      action="store_true")
    parser.add_argument("--live",     action="store_true")
    parser.add_argument("--interval", type=int,  default=60)
    args = parser.parse_args()

    if args.live:
        run_live(exp=args.exp, interval=args.interval, multi=args.multi, n=args.n)
    elif args.multi:
        run_multi(n_expiries=args.n)
    else:
        result, fl_data = run_single(exp=args.exp)
        if args.csv and result is not None:
            export_csv(result, args.exp or date.today().strftime("%Y-%m-%d"))
