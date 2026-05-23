#!/usr/bin/env python3
"""
schedule_freeflow_logger.py
===========================
Continuous 5-minute scheduler around freeflow_logger.py for building a
labeled intraday GEX dataset.

Active window: Mon–Fri 03:00–17:00 ET (London open through RTH close).
Outside that window the loop sleeps in 5-minute increments. All timezone
arithmetic uses America/New_York explicitly on every iteration — no UTC,
no system local time.

Each tick during active hours:
  1. Calls freeflow_logger.build_snapshot()
  2. Overwrites the snapshot's `timestamp` column with the precise ET
     timestamp in the required format (YYYY-MM-DDTHH:MM:SS ET)
  3. Saves via freeflow_logger.save_snapshot()
  4. Appends one row to logs/snapshot_index.csv with empty outcome
     columns — to be filled later by the calibration script

Outputs:
  logs/levels_YYYY-MM-DD.csv      — raw strike-level snapshots
  logs/snapshot_index.csv         — index of snapshots ↔ outcomes
  logs/intraday_inputs_log.jsonl  — audit trail (written by freeflow_logger)

Usage:
  python schedule_freeflow_logger.py          # daemon mode (Ctrl+C to stop)
  python schedule_freeflow_logger.py --once   # one tick then exit (for GitHub Actions / cron)

In --once mode, the script checks ET and either takes a single snapshot
(if inside the active window) or no-ops with a short message, then exits 0.
This is the CI entrypoint; the workflow file at
.github/workflows/gex-snapshot.yml invokes it every 5 minutes.
"""

import csv
import os
import sys
import time
from datetime import datetime, time as dtime, timedelta

# ── TIMEZONE: ET ONLY ─────────────────────────────────────────────────────────
# Prefer pytz (already a dependency of freeflow_logger.py); fall back to zoneinfo.
try:
    import pytz
    ET = pytz.timezone('America/New_York')
    def current_et():
        return datetime.now(ET)
except ImportError:
    try:
        from zoneinfo import ZoneInfo
        ET = ZoneInfo('America/New_York')
        def current_et():
            return datetime.now(ET)
    except ImportError:
        print("ERROR: neither pytz nor zoneinfo is available.")
        print("Install with:  pip install pytz")
        sys.exit(1)

# ── IMPORT FREEFLOW LOGGER ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import freeflow_logger as ffl  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────
ACTIVE_START      = dtime(3, 0)    # 03:00 ET — London open
ACTIVE_END        = dtime(17, 0)   # 17:00 ET — past RTH close
INTERVAL_MINUTES  = 5
WEEKEND_DAYS      = {5, 6}          # Saturday, Sunday
LOG_DIR           = os.path.join(BASE_DIR, 'logs')
INDEX_PATH        = os.path.join(LOG_DIR, 'snapshot_index.csv')
INDEX_COLUMNS     = ['snapshot_timestamp_et', 'snapshot_file',
                     'outcome_nq_close', 'outcome_recorded']

os.makedirs(LOG_DIR, exist_ok=True)


# ── TIME HELPERS ──────────────────────────────────────────────────────────────
def in_active_window(now_et):
    """True iff weekday Mon-Fri and 03:00 ET <= time < 17:00 ET."""
    if now_et.weekday() in WEEKEND_DAYS:
        return False
    return ACTIVE_START <= now_et.time() < ACTIVE_END


def next_interval_boundary(now_et):
    """Return the next ET datetime at an INTERVAL_MINUTES boundary strictly after now."""
    minute = (now_et.minute // INTERVAL_MINUTES + 1) * INTERVAL_MINUTES
    if minute >= 60:
        bumped = now_et + timedelta(hours=1)
        return bumped.replace(minute=0, second=0, microsecond=0)
    return now_et.replace(minute=minute, second=0, microsecond=0)


def next_active_start(now_et):
    """ET datetime of the next active-window start (03:00 ET on the next weekday)."""
    candidate = now_et.replace(hour=ACTIVE_START.hour,
                               minute=ACTIVE_START.minute,
                               second=0, microsecond=0)
    if now_et.time() >= ACTIVE_START:
        candidate = candidate + timedelta(days=1)
    while candidate.weekday() in WEEKEND_DAYS:
        candidate = candidate + timedelta(days=1)
    return candidate


def fmt_ts(now_et):
    """YYYY-MM-DDTHH:MM:SS ET — the required timestamp format."""
    return now_et.strftime('%Y-%m-%dT%H:%M:%S') + ' ET'


# ── SNAPSHOT INDEX ────────────────────────────────────────────────────────────
def ensure_index_header():
    """Write the index header if the file does not yet exist."""
    if os.path.exists(INDEX_PATH):
        return
    with open(INDEX_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(INDEX_COLUMNS)


def append_index_row(ts_et_str, snapshot_file):
    """One row per snapshot; outcome columns stay empty for the calibration job."""
    ensure_index_header()
    with open(INDEX_PATH, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([ts_et_str, snapshot_file, '', ''])


# ── SNAPSHOT ──────────────────────────────────────────────────────────────────
def take_snapshot(now_et):
    """Build, stamp, save, and index a snapshot. Returns the saved file path."""
    ts_str = fmt_ts(now_et)
    label  = now_et.strftime('%H%M')
    print(f"  [{ts_str}] Building snapshot (label={label})...")

    df, futures_price, spot_etf, ctx, regime = ffl.build_snapshot()
    # Override freeflow_logger's plain timestamp with the ET-tagged format.
    df['timestamp'] = ts_str

    ffl.print_summary(df, futures_price, spot_etf, ctx, regime, label)
    snapshot_file = ffl.save_snapshot(df, label)
    append_index_row(ts_str, snapshot_file)
    print(f"  [{ts_str}] Indexed → {INDEX_PATH}")
    return snapshot_file


# ── ONE TICK (shared by daemon and CI modes) ──────────────────────────────────
def perform_tick():
    """
    Single tick: ET check, snapshot if in active window, return outcome.
    Returns (took_snapshot: bool, ts_str: str). Never raises — errors are logged.
    """
    now_et = current_et()
    ts_str = fmt_ts(now_et)

    if now_et.weekday() in WEEKEND_DAYS:
        print(f"[{ts_str}] weekend — no snapshot")
        return False, ts_str

    if not in_active_window(now_et):
        print(f"[{ts_str}] outside active hours "
              f"({ACTIVE_START.strftime('%H:%M')}-{ACTIVE_END.strftime('%H:%M')} ET) — no snapshot")
        return False, ts_str

    print(f"[{ts_str}] tick — weekday={now_et.strftime('%a')} ACTIVE")
    try:
        take_snapshot(now_et)
        return True, ts_str
    except Exception as e:
        print(f"  [{ts_str}] ERROR taking snapshot: {e}")
        return False, ts_str


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    session_timestamps = []
    first_snap, last_snap = None, None

    print("=" * 70)
    print(f"  freeflow_logger {INTERVAL_MINUTES}-minute scheduler")
    print(f"  Active window : Mon-Fri {ACTIVE_START.strftime('%H:%M')}-"
          f"{ACTIVE_END.strftime('%H:%M')} ET")
    print(f"  Interval      : every {INTERVAL_MINUTES} min on aligned boundaries")
    print(f"  Snapshot dir  : {LOG_DIR}")
    print(f"  Index file    : {INDEX_PATH}")
    print("  Stop with Ctrl+C.")
    print("=" * 70)

    try:
        while True:
            now_et = current_et()
            in_window = (now_et.weekday() not in WEEKEND_DAYS
                         and in_active_window(now_et))

            if not in_window:
                # Re-uses perform_tick's logging; it will print and skip.
                perform_tick()
                time.sleep(300)
                continue

            took, ts_str = perform_tick()
            if took:
                session_timestamps.append(ts_str)
                if first_snap is None:
                    first_snap = ts_str
                last_snap = ts_str

            # Sleep until the next aligned INTERVAL_MINUTES boundary, re-reading ET each loop.
            target = next_interval_boundary(current_et())
            while True:
                now2 = current_et()
                if now2 >= target:
                    break
                # Cap sleep to 30s so the ET clock is rechecked frequently.
                remaining = (target - now2).total_seconds()
                time.sleep(max(1, min(30, remaining)))

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("  SHUTDOWN — KeyboardInterrupt")
        print(f"  Snapshots taken this session : {len(session_timestamps)}")
        print(f"  First snapshot               : {first_snap or '(none)'}")
        print(f"  Last snapshot                : {last_snap or '(none)'}")
        print(f"  Snapshot index file          : {INDEX_PATH}")
        print("=" * 70)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="freeflow_logger 5-min scheduler")
    p.add_argument('--once', action='store_true',
                   help='Take a single snapshot if in active ET window, then exit. '
                        'Outside-window invocations exit 0 with a no-op message. '
                        'Used by .github/workflows/gex-snapshot.yml.')
    args = p.parse_args()

    if args.once:
        took, ts = perform_tick()
        # Always exit 0 — outside-hours is not an error condition.
        sys.exit(0)

    main()
