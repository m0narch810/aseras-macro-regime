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

import requests
import pandas as pd
from datetime import datetime, date, timedelta, time as dtime
import os
import sys
import json
import time as time_module
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
    aggregate_multi_expiry,
    classify_regime,
    score_levels,
    get_trading_days,
    SYMBOL,
)

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

def is_trading_day():
    """Simple check — Mon-Fri. Does not account for holidays."""
    return date.today().weekday() < 5


def current_et():
    return datetime.now(ET)


def is_snapshot_time():
    """Returns (True, label) if within window of a snapshot time."""
    now = current_et().time()
    for snap in SNAPSHOT_TIMES:
        snap_dt   = datetime.combine(date.today(), snap)
        now_dt    = datetime.combine(date.today(), now)
        diff_mins = abs((now_dt - snap_dt).total_seconds()) / 60
        if diff_mins <= WINDOW_MINUTES:
            return True, snap.strftime('%H%M')
    return False, None


def already_logged_today(label):
    """Prevent double-logging if script runs multiple times in the window."""
    today  = date.today().strftime('%Y-%m-%d')
    marker = os.path.join(LOG_DIR, f'.logged_{today}_{label}')
    if os.path.exists(marker):
        return True
    open(marker, 'w').close()
    return False


def build_snapshot():
    """Fetch multi-expiry data and return a scored DataFrame."""
    expiries = get_trading_days(N_EXPIRIES)
    session  = make_session()
    datasets = []

    for exp in expiries:
        try:
            fl_data = fetch_futures_levels(session, exp=exp)
            datasets.append(fl_data)
        except Exception as e:
            print(f"  Warning: could not fetch {exp}: {e}")

    if not datasets:
        raise RuntimeError("No data fetched.")

    ctx              = get_vol_context(session)
    futures_price    = datasets[0].get('futures_price', 0)
    spot_etf         = datasets[0].get('etf_spot', 0)
    agg              = aggregate_multi_expiry(datasets)
    regime, weights  = classify_regime(ctx)

    # Score with wider filter for logging (capture ±8% not just ±5%)
    nearby = score_levels(agg, weights, futures_price, filter_pct=8.0)

    # Attach metadata columns
    now_et = current_et()
    nearby['timestamp']    = now_et.strftime('%Y-%m-%d %H:%M:%S')
    nearby['session_label']= now_et.strftime('%H%M') + 'ET'
    nearby['nq_price']     = futures_price
    nearby['qqq_price']    = spot_etf
    nearby['iv']           = ctx.get('current_iv',  '?')
    nearby['rv_iv_ratio']  = ctx.get('rv_iv_ratio', '?')
    nearby['hv21']         = ctx.get('hv21',         '?')
    nearby['regime']       = regime

    return nearby, futures_price, spot_etf, ctx, regime


def save_snapshot(df, label):
    """Append snapshot to today's CSV log."""
    os.makedirs(LOG_DIR, exist_ok=True)
    today   = date.today().strftime('%Y-%m-%d')
    fpath   = os.path.join(LOG_DIR, f'levels_{today}.csv')

    cols = [
        'timestamp', 'session_label', 'nq_price', 'qqq_price',
        'iv', 'rv_iv_ratio', 'hv21', 'regime',
        'strike_futures', 'strike_etf', 'dist_nq', 'dist_pct',
        'score', 'type',
        'net_gex', 'net_vex', 'net_charmex', 'net_dex', 'net_dag',
        'total_oi',
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

    top_support    = df[df['net_gex'] < 0].head(3)
    top_resistance = df[df['net_gex'] > 0].head(3)

    print("  RESISTANCE:")
    for _, r in top_resistance.iterrows():
        print(f"    NQ {r['strike_futures']:.0f}  Score:{r['score']:.0f}  GEX:{r['net_gex']:,.0f}")
    print("  SUPPORT:")
    for _, r in top_support.iterrows():
        print(f"    NQ {r['strike_futures']:.0f}  Score:{r['score']:.0f}  GEX:{r['net_gex']:,.0f}")
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
