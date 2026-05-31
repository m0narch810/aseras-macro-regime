"""
Batch-decode all OPRA CBBO-1m files into intraday GEX snapshots.

Generates one GEX profile per 30-minute RTH interval per day (13 snapshots/day)
instead of just the opening snapshot. This provides dynamically updating walls
throughout the session, enabling the multi-trade-per-day pattern.

Outputs to: data/processed/gex_snapshots_0dte/
  gex_snapshot_YYYYMMDD_HHMM.csv  (one file per snapshot)

Also still writes the opening profile to gex_profiles_0dte/ for backward
compatibility with the hold-probability model.

Usage:
  python scripts/batch_decode_opra.py
  python scripts/batch_decode_opra.py --workers 4
  python scripts/batch_decode_opra.py --resume
"""

import argparse
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

DATA_DIR    = Path(__file__).parent.parent / "dataidk"
OUT_SNAPS   = Path(__file__).parent.parent / "data" / "processed" / "gex_snapshots_0dte"
OUT_PROFILE = Path(__file__).parent.parent / "data" / "processed" / "gex_profiles_0dte"
NQ_1M_PATH  = Path(__file__).parent.parent / "data" / "processed" / "NQ_1m_clean.csv"

OUT_SNAPS.mkdir(parents=True, exist_ok=True)
OUT_PROFILE.mkdir(parents=True, exist_ok=True)


def _build_nq_price_map() -> dict:
    """
    Load NQ 1-minute data.
    Returns {date: {(hour, minute): nq_open_price}} for every RTH bar.
    """
    print("Loading NQ 1-minute data for intraday spot computation...")
    nq = pd.read_csv(NQ_1M_PATH, parse_dates=["date"])
    nq = nq.rename(columns={"date": "dt"})
    nq["date_only"] = nq["dt"].dt.date
    nq["hour"]      = nq["dt"].dt.hour
    nq["minute"]    = nq["dt"].dt.minute
    # Filter to RTH only (9:30-16:00 ET)
    rth = nq[(nq["hour"] >= 9) & ~((nq["hour"] == 9) & (nq["minute"] < 30))
             & (nq["hour"] < 16)].copy()
    result = {}
    for row in rth.itertuples():
        d = row.date_only
        if d not in result:
            result[d] = {}
        result[d][(row.hour, row.minute)] = row.open
    print(f"  NQ intraday prices for {len(result)} dates")
    return result


def _worker(args_tuple):
    """
    Process one day: generate all intraday snapshots + opening profile.
    Returns (date, n_snapshots | 'empty' | error_msg).
    """
    d, nq_prices_today = args_tuple
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.decode_opra_day import process_day_snapshots, process_day

        nq_open = (nq_prices_today or {}).get((9, 31)) or (nq_prices_today or {}).get((9, 30))

        # ── Intraday snapshots ──────────────────────────────────────────────
        snapshots = process_day_snapshots(d, nq_prices_today or {}, debug=False)
        if not snapshots:
            return d, "empty"

        for dt_str, profile in snapshots:
            out = OUT_SNAPS / f"gex_snapshot_{dt_str}.csv"
            profile.to_csv(out, index=False)

        # ── Opening profile (backward compat for hold-prob model) ───────────
        opening = next((p for s, p in snapshots if s.endswith("_0931")), None)
        if opening is not None:
            out_p = OUT_PROFILE / f"gex_profile_{d.strftime('%Y%m%d')}.csv"
            opening.drop(columns=["snapshot_dt"], errors="ignore").to_csv(out_p, index=False)

        return d, len(snapshots)
    except Exception:
        return d, traceback.format_exc().strip().splitlines()[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="skip dates that already have all snapshots")
    args = parser.parse_args()

    pattern   = re.compile(r"opra-pillar-(\d{8})\.cbbo-1m\.dbn\.zst")
    all_dates = sorted(
        date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:]))
        for f in DATA_DIR.glob("*.zst")
        if (m := pattern.match(f.name))
    )

    if not all_dates:
        print("No .zst files found in dataidk/")
        sys.exit(1)

    nq_map = _build_nq_price_map()

    if args.resume:
        done_dates = {
            date(int(f.name[13:17]), int(f.name[17:19]), int(f.name[19:21]))
            for f in OUT_SNAPS.glob("gex_snapshot_*_0931.csv")
        }
        todo = [d for d in all_dates if d not in done_dates]
        print(f"{len(done_dates)} already done, {len(todo)} remaining")
    else:
        todo = all_dates

    todo_args = [(d, nq_map.get(d, {})) for d in todo]
    print(f"Processing {len(todo)} days ({len(todo) * 13} snapshots) "
          f"with {args.workers} workers ...\n")

    ok = total_snaps = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in todo_args}
        for i, fut in enumerate(as_completed(futures), 1):
            d, result = fut.result()
            if isinstance(result, int):
                ok += 1
                total_snaps += result
                status = f"ok ({result} snapshots)"
            elif result == "empty":
                skipped += 1
                status = "skip (no 0DTE data)"
            else:
                errors += 1
                status = f"ERROR: {result}"
            pct = i / len(todo) * 100
            print(f"  [{i:>3}/{len(todo)}  {pct:5.1f}%]  {d}  {status}")

    print(f"\nDone: {ok} days ok ({total_snaps} snapshots), "
          f"{skipped} skipped, {errors} errors")
    print(f"Snapshots in: {OUT_SNAPS}")


if __name__ == "__main__":
    main()
