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
from datetime import datetime, date, timedelta
import sys
import os
import json

# ============================================================
# AUTHENTICATION
# ============================================================

COOKIES = {
    'ff_session': os.environ.get('FF_SESSION', 'j_g_U67Ox9np5aXxK5YKHkVnQt41_RnyUEtmo7Bw7q0'),
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


def get_vol_context(session):
    """
    Fetch vol context from /api/vol/realized.
    Returns current_iv, rv_iv_ratio, hv5/10/21/63.
    /api/cone is 404. /api/vol/cone has no IV. /api/vol/realized has everything.
    """
    ctx = {
        'current_iv':  None,
        'rv_iv_ratio': None,
        'hv5':         None,
        'hv10':        None,
        'hv21':        None,
        'hv63':        None,
    }
    try:
        url = f"{BASE_URL}/vol/realized?symbol={SYMBOL}"
        r = session.get(url, timeout=15)
        r.raise_for_status()
        d = r.json()
        ctx['current_iv']  = d.get('current_iv')
        ctx['rv_iv_ratio'] = d.get('rv_iv_ratio')
        ctx['hv5']         = d.get('hv5')
        ctx['hv10']        = d.get('hv10')
        ctx['hv21']        = d.get('hv21')
        ctx['hv63']        = d.get('hv63')
    except Exception as e:
        print(f"  Warning: vol fetch failed ({e})")
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

# Per-strike call/put OI split (from raw `right`) and OI-weighted strike IV
# (from `iv_pct`). Mirrors the live function so logged snapshots replay the
# mechanical hold_prob's PCR + skew factors faithfully. Tolerates older payloads
# missing the columns by leaving the splits at 0 / IV at NaN.
def _add_split_cols(df):
    oi = pd.to_numeric(df.get('oi', 0), errors='coerce').fillna(0)
    right = df['right'] if 'right' in df.columns else pd.Series([''] * len(df), index=df.index)
    df['_call_oi'] = oi.where(right == 'C', 0.0)
    df['_put_oi']  = oi.where(right == 'P', 0.0)
    ivp = pd.to_numeric(df.get('iv_pct'), errors='coerce') if 'iv_pct' in df.columns else None
    df['_iv_oi'] = (ivp * oi) if ivp is not None else float('nan')
    return df


def _finalize_split(agg):
    agg = agg.rename(columns={'_call_oi': 'call_oi', '_put_oi': 'put_oi'})
    agg['strike_iv'] = (agg['_iv_oi'] / agg['total_oi']).where(agg['total_oi'] > 0)
    return agg.drop(columns=['_iv_oi'])


def aggregate_exposures(fl_data):
    rows = fl_data.get('rows', [])
    if not rows:
        raise ValueError("No rows in futures-levels response.")

    df = pd.DataFrame(rows)

    if 'strike_futures' not in df.columns:
        ratio = fl_data.get('ratio', 41.14)
        df['strike_futures'] = (df['strike_etf'] * ratio).round(1)

    df = _add_split_cols(df)
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
        _call_oi    = ('_call_oi',    'sum'),
        _put_oi     = ('_put_oi',     'sum'),
        _iv_oi      = ('_iv_oi',      'sum'),
    ).reset_index()

    return _finalize_split(agg)


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

    combined = _add_split_cols(pd.concat(dfs, ignore_index=True))

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
        _call_oi    = ('_call_oi',    'sum'),
        _put_oi     = ('_put_oi',     'sum'),
        _iv_oi      = ('_iv_oi',      'sum'),
    ).reset_index()

    return _finalize_split(agg)

# ============================================================
# VOL REGIME
# ============================================================

def classify_regime(ctx):
    """
    CONTRACTION (IV < 20%): gamma walls sticky, GEX dominates
    NEUTRAL     (20-30%):   mixed, GEX and VEX both matter
    EXPANSION   (IV > 30% or RV/IV < 0.5): vol flows dominate, VEX dominates
    """
    iv    = ctx.get('current_iv')  or 37.0
    rv_iv = ctx.get('rv_iv_ratio') or 0.46

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
    iv    = ctx.get('current_iv')
    rv_iv = ctx.get('rv_iv_ratio')
    hv21  = ctx.get('hv21')
    bar   = "=" * 85

    iv_str  = f"{iv:.1f}%"  if isinstance(iv,    float) else str(iv)
    rv_str  = f"{rv_iv:.3f}" if isinstance(rv_iv, float) else str(rv_iv)
    hv_str  = f"{hv21:.1f}%" if isinstance(hv21,  float) else str(hv21)

    if isinstance(iv, float) and isinstance(rv_iv, float) and rv_iv > 0:
        iv_premium = f"  |  IV Premium: {(1/rv_iv - 1)*100:.0f}% above RV"
    else:
        iv_premium = ""

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
    """Write levels_data.json consumed by the Vanta dashboard Levels tab."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'levels_data.json')

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

    iv    = ctx.get('current_iv')
    rv_iv = ctx.get('rv_iv_ratio')
    hv21  = ctx.get('hv21')

    payload = {
        'updated':     datetime.now().strftime('%Y-%m-%d %H:%M ET'),
        'nq_price':    round(float(futures_price), 1),
        'qqq_price':   round(float(spot_etf), 2),
        'regime':      regime,
        'iv':          round(float(iv),    1) if iv    is not None else None,
        'rv_iv_ratio': round(float(rv_iv), 3) if rv_iv is not None else None,
        'hv21':        round(float(hv21),  1) if hv21  is not None else None,
        'levels':      levels,
    }

    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved levels_data.json  ({len(levels)} levels)")
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

    ctx             = get_vol_context(session)
    futures_price   = fl_data.get('futures_price', 0)
    spot_etf        = fl_data.get('etf_spot', 0)
    agg             = aggregate_exposures(fl_data)
    regime, weights = classify_regime(ctx)
    nearby          = score_levels(agg, weights, futures_price)

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
