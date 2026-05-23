"""
FreeFlow Daily Level Logger
===========================
Captures options positioning snapshots 4x per trading day at
meaningful market structure times (ET):

  08:30 — Pre-market / econ data window
  09:35 — Just after open (let first 5 min settle)
  13:00 — Midday repositioning
  15:30 — Power hour / final 30 min

Saves to: logs/levels_YYYY-MM-DD.csv
One row per strike per snapshot. Builds a dataset for backtesting
which levels actually generated NQ reactions.

SETUP (Windows Task Scheduler):
  Action: python C:\\path\\to\\freeflow_logger.py
  Trigger: Daily, repeat every 30 min from 8:00 to 16:00
  The script self-checks whether it's a snapshot time and exits
  immediately if not. Zero overhead on non-snapshot runs.

OR run manually:
  python freeflow_logger.py            # logs if current time is a snapshot time
  python freeflow_logger.py --force    # logs immediately regardless of time
  python freeflow_logger.py --schedule # runs continuously and waits for snapshot times
"""

from datetime import datetime, time as dtime
import math
import os
import sys
import json
import time as time_module
import pandas as pd
import pytz

# ============================================================
# IMPORT CORE FUNCTIONS FROM freeflow_levels.py
# ============================================================
# Must be in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeflow_levels import (
    make_session,
    fetch_futures_levels,
    get_vol_context,
    classify_regime,
    normalize,
    get_trading_days,
)

# Snapshot-specific knobs: wider proximity window than dashboard (5%) so we
# capture strikes that may move into range over the trading day.
SNAPSHOT_FILTER_PCT = 10.0   # %-of-price window for scoring + saving
SCORE_FILTER_PCT    = 8.0    # tighter window used for score normalization

# ============================================================
# SNAPSHOT TIMES (Eastern Time)
# ============================================================

ET = pytz.timezone('America/New_York')

SNAPSHOT_TIMES = [
    dtime(8, 30),   # Pre-market / economic data
    dtime(9, 35),   # Just after open (first 5 min settle)
    dtime(13, 0),   # Midday repositioning
    dtime(15, 30),  # Power hour
]

WINDOW_MINUTES = 4   # How many minutes either side of target to accept
N_EXPIRIES     = 3   # Expiries to aggregate
LOG_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

# Ensure logs/ exists at module load — required by both snapshot save and JSONL audit.
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# LOGGING
# ============================================================

def current_et():
    return datetime.now(ET)


def is_trading_day():
    """Simple check — Mon-Fri in ET. Does not account for holidays.

    Uses ET explicitly (not system local time) so this is correct when run on
    UTC-based CI runners at Friday/Monday boundaries.
    """
    return current_et().weekday() < 5


def is_snapshot_time():
    """Returns (True, label) if within window of a snapshot time.

    All times compared in ET (matches SNAPSHOT_TIMES). No system local time.
    """
    et_now = current_et()
    today_et = et_now.date()
    now = et_now.time()
    for snap in SNAPSHOT_TIMES:
        snap_dt   = datetime.combine(today_et, snap)
        now_dt    = datetime.combine(today_et, now)
        diff_mins = abs((now_dt - snap_dt).total_seconds()) / 60
        if diff_mins <= WINDOW_MINUTES:
            return True, snap.strftime('%H%M')
    return False, None


def already_logged_today(label):
    """Prevent double-logging if script runs multiple times in the window.

    Uses ET date so the marker filename matches the ET trading day, not the
    UTC date the CI runner happens to be on.
    """
    today  = current_et().strftime('%Y-%m-%d')
    marker = os.path.join(LOG_DIR, f'.logged_{today}_{label}')
    if os.path.exists(marker):
        return True
    open(marker, 'w').close()
    return False


def compute_gamma_flip(agg, futures_price):
    """Linear-interpolate gamma flip (net_gex zero crossing nearest to current
    price). Mirrors netlify/functions/lib/options.js::computeGammaFlip so
    snapshot-time gamma regime matches what the dashboard sees live.
    """
    if agg is None or len(agg) == 0 or futures_price is None:
        return None
    s = agg.sort_values('strike_futures').reset_index(drop=True)
    best_flip, best_dist = None, float('inf')
    for i in range(len(s) - 1):
        a_strike, a_gex = float(s.iloc[i]['strike_futures']),     float(s.iloc[i]['net_gex'])
        b_strike, b_gex = float(s.iloc[i + 1]['strike_futures']), float(s.iloc[i + 1]['net_gex'])
        if (a_gex > 0 and b_gex <= 0) or (a_gex < 0 and b_gex >= 0):
            denom = abs(a_gex) + abs(b_gex)
            if denom == 0:
                continue
            flip = a_strike + (b_strike - a_strike) * abs(a_gex) / denom
            dist = abs(flip - futures_price)
            if dist < best_dist:
                best_dist = dist
                best_flip = flip
    if best_flip is None:
        idx = s['net_gex'].abs().idxmin()
        best_flip = float(s.loc[idx, 'strike_futures'])
    return round(best_flip * 10) / 10


def classify_gamma_regime(futures_price, gamma_flip, iv):
    """Vol-scaled gamma regime band — mirrors the JS handler.
    POSITIVE if price - flip > _ivBand; NEGATIVE if < -_ivBand; else NEAR_FLIP.
    _ivBand = max(30, 0.5 * price * iv/100 / sqrt(252))   (1σ half-day move / 2)
    """
    if gamma_flip is None or futures_price is None:
        return None
    iv_val  = iv if iv is not None else 30.0
    iv_band = max(30.0, 0.5 * futures_price * iv_val / 100.0 / math.sqrt(252))
    diff    = futures_price - gamma_flip
    if diff >  iv_band: return 'POSITIVE'
    if diff < -iv_band: return 'NEGATIVE'
    return 'NEAR_FLIP'


def _build_per_expiry_and_aggregate(datasets, expiries_kept):
    """Returns (per_expiry_df, aggregate_df) with full greeks columns.
    per_expiry_df has one row per (strike, expiry); aggregate_df one per strike.
    """
    per_exp_dfs = []
    for exp, fl_data in zip(expiries_kept, datasets):
        rows = fl_data.get('rows', [])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if 'strike_futures' not in df.columns:
            ratio = fl_data.get('ratio', 41.14)
            df['strike_futures'] = (df['strike_etf'] * ratio).round(1)
        df['expiry'] = exp
        per_exp_dfs.append(df)

    if not per_exp_dfs:
        return None, None

    combined = pd.concat(per_exp_dfs, ignore_index=True)

    greek_agg = dict(
        strike_etf  = ('strike_etf',  'first'),
        net_gex     = ('gex',         'sum'),
        abs_gex     = ('ag',          'sum'),
        net_vex     = ('vex',         'sum'),
        net_charmex = ('charmex',     'sum'),
        net_dex     = ('dex',         'sum'),
        net_vegaex  = ('vegaex',      'sum'),
        net_dag     = ('dag',         'sum'),
        total_oi    = ('oi',          'sum'),
    )
    per_exp = (combined
               .groupby(['strike_futures', 'expiry'], as_index=False)
               .agg(**greek_agg))
    agg     = (combined
               .groupby('strike_futures', as_index=False)
               .agg(**greek_agg))
    agg['expiry'] = 'AGGREGATE'
    return per_exp, agg


def build_snapshot():
    """Fetch multi-expiry data, score the aggregate (no MIN_SCORE filter), and
    return a DataFrame with per-(strike, expiry) rows AND per-strike aggregate
    rows. Snapshot-level metadata (gamma flip, gamma regime, all HVs, vol
    regime, spot prices) is duplicated on every row so each row is
    self-contained for later joining to outcomes.

    Returned tuple matches the prior signature so schedule_freeflow_logger
    keeps working without changes:
        (df, futures_price, spot_etf, ctx, regime)
    """
    expiries      = get_trading_days(N_EXPIRIES)
    session       = make_session()
    datasets      = []
    expiries_kept = []

    for exp in expiries:
        try:
            fl_data = fetch_futures_levels(session, exp=exp)
            datasets.append(fl_data)
            expiries_kept.append(exp)
        except Exception as e:
            print(f"  Warning: could not fetch {exp}: {e}")

    if not datasets:
        raise RuntimeError("No data fetched.")

    ctx             = get_vol_context(session)
    futures_price   = datasets[0].get('futures_price', 0)
    spot_etf        = datasets[0].get('etf_spot', 0)
    regime, weights = classify_regime(ctx)

    per_exp, agg = _build_per_expiry_and_aggregate(datasets, expiries_kept)
    if agg is None:
        raise RuntimeError("Aggregation produced no rows.")

    # Distance columns on both frames.
    for df in (per_exp, agg):
        df['dist_nq']  = df['strike_futures'] - futures_price
        df['dist_pct'] = (df['dist_nq'] / futures_price) * 100.0

    # Score the aggregate within SCORE_FILTER_PCT (no MIN_SCORE filter — keep
    # every strike with its score attached so post-hoc analysis can re-threshold).
    nearby_mask = agg['dist_pct'].abs() <= SCORE_FILTER_PCT
    nearby      = agg[nearby_mask].copy()
    agg['score'] = None
    agg['type']  = None
    if not nearby.empty:
        nearby['gex_n']     = normalize(nearby['net_gex'])
        nearby['vex_n']     = normalize(nearby['net_vex'])
        nearby['charmex_n'] = normalize(nearby['net_charmex'])
        nearby['oi_n']      = normalize(nearby['total_oi'])
        nearby['dag_n']     = normalize(nearby['net_dag'])
        nearby['score'] = (
            nearby['gex_n']     * weights['gex']     +
            nearby['vex_n']     * weights['vex']     +
            nearby['charmex_n'] * weights['charmex'] +
            nearby['oi_n']      * weights['oi']      +
            nearby['dag_n']     * weights['dag']
        ) * 100.0

        def _typ(row):
            g = row['net_gex']; v = row['net_vex']
            vol_sens = abs(v) / (abs(g) + 1e-9)
            base = 'CALL WALL' if g > 0 else 'PUT WALL'
            return base + (' + VOL SENSITIVE' if vol_sens > 2.0 else '')
        nearby['type'] = nearby.apply(_typ, axis=1)

        score_map = nearby.set_index('strike_futures')[['score', 'type']]
        agg = agg.drop(columns=['score', 'type']).merge(
            score_map, left_on='strike_futures', right_index=True, how='left'
        )

    # Per-expiry rows have no score attached (would skew normalization across slices).
    per_exp['score'] = None
    per_exp['type']  = None

    # Keep everything inside the wider SNAPSHOT_FILTER_PCT window.
    agg_keep     = agg    [agg    ['dist_pct'].abs() <= SNAPSHOT_FILTER_PCT].copy()
    per_exp_keep = per_exp[per_exp['dist_pct'].abs() <= SNAPSHOT_FILTER_PCT].copy()

    out = pd.concat([per_exp_keep, agg_keep], ignore_index=True)

    # Gamma flip + regime from the FULL aggregate (zero crossing nearest spot).
    iv_val       = ctx.get('current_iv')
    gamma_flip   = compute_gamma_flip(agg, futures_price)
    gamma_regime = classify_gamma_regime(futures_price, gamma_flip, iv_val)

    # Snapshot-level metadata duplicated on every row.
    now_et = current_et()
    out['timestamp']     = now_et.strftime('%Y-%m-%d %H:%M:%S')
    out['session_label'] = now_et.strftime('%H%M') + 'ET'
    out['nq_price']      = futures_price
    out['qqq_price']     = spot_etf
    out['gamma_flip']    = gamma_flip
    out['gamma_regime']  = gamma_regime
    out['vol_regime']    = regime
    out['iv']            = ctx.get('current_iv')
    out['rv_iv_ratio']   = ctx.get('rv_iv_ratio')
    out['hv5']           = ctx.get('hv5')
    out['hv10']          = ctx.get('hv10')
    out['hv21']          = ctx.get('hv21')
    out['hv63']          = ctx.get('hv63')

    return out, futures_price, spot_etf, ctx, regime


def save_snapshot(df, label):
    """Append snapshot to today's CSV log (ET date, not UTC)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    today   = current_et().strftime('%Y-%m-%d')
    fpath   = os.path.join(LOG_DIR, f'levels_{today}.csv')

    cols = [
        # Snapshot context (duplicated on every row)
        'timestamp', 'intended_timestamp', 'snapshot_date', 'snapshot_weekday',
        'session_label', 'expiry',
        'nq_price', 'qqq_price', 'gamma_flip', 'gamma_regime', 'vol_regime',
        'iv', 'rv_iv_ratio', 'hv5', 'hv10', 'hv21', 'hv63',
        # Strike geometry
        'strike_futures', 'strike_etf', 'dist_nq', 'dist_pct',
        # Full raw greeks (per-expiry rows have per-expiry sums; AGGREGATE rows have totals)
        'net_gex', 'abs_gex', 'net_vex', 'net_charmex', 'net_dex',
        'net_dag', 'net_vegaex', 'total_oi',
        # Aggregate-row scoring (None on per-expiry rows)
        'score', 'type',
    ]
    cols = [c for c in cols if c in df.columns]

    write_header = not os.path.exists(fpath)
    df[cols].to_csv(fpath, mode='a', header=write_header, index=False)
    print(f"  Saved {len(df)} rows → {fpath}")

    # Append intraday-bias audit line: a JSONL marker that this snapshot exists.
    # Once 30+ days of data accumulate, these markers can be joined to next-day
    # outcomes to calibrate STRONG_WALL, EXCEPTIONAL_WALL, PROXIMITY_EFOLD,
    # and REGIME_WEIGHTS.
    audit_path = os.path.join(LOG_DIR, 'intraday_inputs_log.jsonl')
    now_et = current_et()
    audit_rec = {
        'timestamp': now_et.isoformat(),
        'date':      now_et.strftime('%Y-%m-%d'),
        'source':    'freeflow_logger',
        'note':      'raw_snapshot — intraday bias inputs to be correlated with outcomes',
    }
    with open(audit_path, 'a') as f:
        f.write(json.dumps(audit_rec) + '\n')

    return fpath


def print_summary(df, futures_price, spot_etf, ctx, regime, label):
    iv    = ctx.get('current_iv',  '?')
    rv_iv = ctx.get('rv_iv_ratio', '?')
    print(f"\n{'='*60}")
    print(f"  SNAPSHOT {label}ET  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  QQQ: ${spot_etf:.2f}  NQ: {futures_price:.1f}")
    print(f"  Regime: {regime}  IV: {iv}  RV/IV: {rv_iv}")
    print(f"{'='*60}")

    # Aggregate rows only — per-expiry rows would duplicate strikes in the summary.
    agg_only = df[df['expiry'] == 'AGGREGATE'] if 'expiry' in df.columns else df
    if 'score' in agg_only.columns:
        agg_only = agg_only.sort_values('score', ascending=False, na_position='last')
    top_support    = agg_only[agg_only['net_gex'] < 0].head(3)
    top_resistance = agg_only[agg_only['net_gex'] > 0].head(3)

    def _fmt_score(s):
        return f"{s:.0f}" if (s is not None and pd.notna(s)) else "—"
    print("  RESISTANCE:")
    for _, r in top_resistance.iterrows():
        print(f"    NQ {r['strike_futures']:.0f}  Score:{_fmt_score(r.get('score'))}  GEX:{r['net_gex']:,.0f}")
    print("  SUPPORT:")
    for _, r in top_support.iterrows():
        print(f"    NQ {r['strike_futures']:.0f}  Score:{_fmt_score(r.get('score'))}  GEX:{r['net_gex']:,.0f}")
    print()


# ============================================================
# MAIN MODES
# ============================================================

def run_once(force=False):
    """Single execution — logs if it's a snapshot time (or forced)."""
    if not is_trading_day() and not force:
        print("Not a trading day. Use --force to override.")
        return

    if force:
        label = current_et().strftime('%H%M')
        print(f"Forced snapshot at {label}ET...")
    else:
        is_snap, label = is_snapshot_time()
        if not is_snap:
            print(f"Not a snapshot time. Current ET: {current_et().strftime('%H:%M')}")
            print(f"Snapshot times: {[t.strftime('%H:%M') for t in SNAPSHOT_TIMES]}")
            return
        if already_logged_today(label):
            print(f"Already logged {label} today. Skipping.")
            return
        print(f"Snapshot time: {label}ET — logging...")

    try:
        df, futures_price, spot_etf, ctx, regime = build_snapshot()
        print_summary(df, futures_price, spot_etf, ctx, regime, label)
        save_snapshot(df, label)
    except Exception as e:
        print(f"ERROR: {e}")


def run_schedule():
    """
    Runs continuously — checks every minute whether it's snapshot time.
    Use this instead of Task Scheduler if you prefer.
    """
    print("Scheduler running. Ctrl+C to stop.")
    print(f"Snapshot times (ET): {[t.strftime('%H:%M') for t in SNAPSHOT_TIMES]}\n")

    while True:
        try:
            if is_trading_day():
                is_snap, label = is_snapshot_time()
                if is_snap and not already_logged_today(label):
                    print(f"\n[{current_et().strftime('%H:%M')}] Snapshot triggered: {label}ET")
                    try:
                        df, fp, sp, ctx, regime = build_snapshot()
                        print_summary(df, fp, sp, ctx, regime, label)
                        save_snapshot(df, label)
                    except Exception as e:
                        print(f"  ERROR: {e}")
            time_module.sleep(60)
        except KeyboardInterrupt:
            print("\nScheduler stopped.")
            break


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FreeFlow Level Logger")
    parser.add_argument("--force",    action="store_true",
                        help="Log immediately regardless of time")
    parser.add_argument("--schedule", action="store_true",
                        help="Run continuously and trigger at snapshot times")
    args = parser.parse_args()

    if args.schedule:
        run_schedule()
    else:
        run_once(force=args.force)
