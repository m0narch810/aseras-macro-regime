#!/usr/bin/env python3
"""
scripts/10_validate_intraday.py -- walk-forward validation of the intraday bias.

For each logged trading day that has both a levels snapshot and a next-day
NQ price outcome, runs the full 09_intraday_bias classifier in hindcast mode
and records the prediction vs actual outcome.

Accuracy breakdowns:
  - By entropy_state (STABLE vs CRITICAL)
  - By gamma_regime (POSITIVE / NEGATIVE / NEAR_FLIP / UNKNOWN)
  - By air_pocket_watch (True / False)
  - By macro alignment (macro agrees / disagrees / neutral)

Binomial test on STABLE sessions (where a directional edge is claimed).

Usage:
    python scripts/10_validate_intraday.py
    python scripts/10_validate_intraday.py --start 2026-01-01 --end 2026-05-22
    python scripts/10_validate_intraday.py --report-path validation_report.json
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from scipy import stats

# Import the classifier's core functions directly (not __main__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
ET       = pytz.timezone("America/New_York")

BULL_BIASES = {"BULLISH", "NEUTRAL_BULLISH", "BULLISH REVERSAL"}
BEAR_BIASES = {"BEARISH CONTINUATION", "BEARISH REVERSAL WATCH",
               "BEARISH REVERSAL", "REVERSAL WATCH"}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_daily_full() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "NQ_daily_clean.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def discover_log_dates() -> list[str]:
    """Find all dates for which a levels log file exists."""
    if not os.path.isdir(LOGS_DIR):
        return []
    dates = []
    for fname in sorted(os.listdir(LOGS_DIR)):
        if fname.startswith("levels_") and fname.endswith(".csv"):
            date_str = fname[len("levels_"):-len(".csv")]
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date_str)
            except ValueError:
                pass
    return dates


def load_levels_snapshot(date_str: str) -> dict | None:
    """
    Reconstruct a levels payload from the last intraday snapshot for date_str.
    Returns a dict matching the live levels.js response shape, or None.
    """
    log_path = os.path.join(LOGS_DIR, f"levels_{date_str}.csv")
    if not os.path.exists(log_path):
        return None
    try:
        df = pd.read_csv(log_path)
    except Exception:
        return None

    if df.empty:
        return None

    # Use last snapshot of the day
    snapshots = sorted(df["snapshot_time"].unique())
    last = df[df["snapshot_time"] == snapshots[-1]]

    # Reconstruct levels list and nq_price/gamma_flip from CSV columns
    levels = []
    for _, row in last.iterrows():
        lv = {
            "strike_futures": float(row.get("strike_futures", 0)),
            "dist_nq":        float(row.get("dist_nq", 0)),
            "score":          float(row.get("score", 0)),
            "type":           str(row.get("type", "")),
            "net_gex":        float(row.get("net_gex", 0)),
            "net_vex":        float(row.get("net_vex", 0)),
            "net_charmex":    float(row.get("net_charmex", 0)),
            "net_dex":        float(row.get("net_dex", 0)),
            "net_vegaex":     float(row.get("net_vegaex", 0)),
            "total_oi":       float(row.get("total_oi", 0)),
            "strike_etf":     float(row.get("strike_etf", 0)),
        }
        levels.append(lv)

    nq_price   = float(last["nq_price"].iloc[0])   if "nq_price"   in last.columns else None
    gamma_flip = float(last["gamma_flip"].iloc[0]) if "gamma_flip" in last.columns else None
    regime     = str(last["regime"].iloc[0])        if "regime"     in last.columns else "UNKNOWN"

    return {
        "nq_price":   nq_price,
        "gamma_flip": gamma_flip,
        "regime":     regime,
        "levels":     levels,
    }


def get_next_close(daily_df: pd.DataFrame, date_str: str) -> float | None:
    """Return the NQ close on the NEXT trading day after date_str."""
    ts = pd.Timestamp(date_str)
    future = daily_df[daily_df["date"] > ts]
    if future.empty:
        return None
    return float(future.iloc[0]["close"])


def get_current_close(daily_df: pd.DataFrame, date_str: str) -> float | None:
    """Return the NQ close ON date_str (the day being predicted)."""
    ts = pd.Timestamp(date_str)
    row = daily_df[daily_df["date"] == ts]
    if row.empty:
        return None
    return float(row.iloc[0]["close"])


# ============================================================================
# HINDCAST ENGINE
# ============================================================================

def run_hindcast(date_str: str, daily_df: pd.DataFrame) -> dict | None:
    """
    Run a single hindcast prediction for date_str.
    Uses only data strictly before date_str (leakage-safe).
    Imports 09_intraday_bias functions directly.
    """
    # Import dynamically to avoid polluting module namespace
    spec = importlib.util.spec_from_file_location(
        "intraday_bias",
        os.path.join(os.path.dirname(__file__), "09_intraday_bias.py")
    )
    ib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ib)

    pred_date = ET.localize(datetime.strptime(date_str, "%Y-%m-%d"))

    # Data strictly before pred_date
    hist = daily_df[daily_df["date"] < pd.Timestamp(date_str)].copy()
    if len(hist) < ib.ENTROPY_MIN_BARS:
        return None

    # Entropy
    entropy = ib.compute_return_entropy(hist, pred_date)

    # PCA — use a separate in-memory fit per hindcast to avoid leakage from future refits
    feat_df = ib.build_pca_feature_matrix(hist)
    try:
        pca_state = ib.load_or_fit_pca(feat_df, pred_date)
        pca_scores = ib.compute_pca_scores(feat_df, pred_date, pca_state)
    except Exception as e:
        log.debug("PCA failed for %s: %s", date_str, e)
        pca_scores = {"PC1": None, "PC2": None, "PC3": None,
                      "pca_fit_date": None, "pca_feature_date": None}

    # Load levels snapshot for this date
    levels_raw = load_levels_snapshot(date_str)
    nq_price   = levels_raw.get("nq_price")   if levels_raw else None
    gamma_flip = levels_raw.get("gamma_flip") if levels_raw else None
    levels     = levels_raw.get("levels", []) if levels_raw else []
    lvl_regime = levels_raw.get("regime", "UNKNOWN") if levels_raw else "UNKNOWN"

    H_GEX_norm   = ib.compute_H_GEX_norm(levels)
    gamma_regime = ib.compute_gamma_regime(nq_price, gamma_flip)
    top_wall     = ib.compute_top_wall(levels, nq_price)
    mm_intense   = ib.compute_mm_intensification(pred_date)

    # Load macro bias (use the version committed at the time — approximate via static file)
    macro_data = {}
    bias_path  = os.path.join(BASE_DIR, "bias_output.json")
    if os.path.exists(bias_path):
        try:
            with open(bias_path) as f:
                macro_data = json.load(f)
        except Exception:
            pass
    macro_bias = macro_data.get("confluence", "UNKNOWN")

    result = ib.classify_intraday_bias(
        entropy       = entropy,
        pca           = pca_scores,
        gamma_regime  = gamma_regime,
        gamma_flip    = gamma_flip,
        nq_price      = nq_price,
        top_wall      = top_wall,
        H_GEX_norm    = H_GEX_norm,
        mm_intense    = mm_intense,
        macro_bias    = macro_bias,
        levels_regime = lvl_regime,
    )

    return {
        "date":             date_str,
        "entropy_state":    entropy.get("entropy_state"),
        "H_returns":        entropy.get("H_returns"),
        "gamma_regime":     gamma_regime,
        "H_GEX_norm":       H_GEX_norm,
        "top_wall_score":   top_wall.get("score") if top_wall else None,
        "macro_bias":       macro_bias,
        "intraday_bias":    result["intraday_bias"],
        "confidence":       result["confidence"],
        "air_pocket_watch": result["air_pocket_watch"],
        "air_pocket_type":  result["air_pocket_type"],
        "nq_price":         nq_price,
        "gamma_flip":       gamma_flip,
    }


# ============================================================================
# ACCURACY ENGINE
# ============================================================================

def determine_outcome(prediction: dict, daily_df: pd.DataFrame) -> str | None:
    """
    Return "CORRECT", "INCORRECT", or None (no outcome available).
    Outcome = next-day return direction vs predicted bias direction.
    """
    date_str   = prediction["date"]
    bias       = prediction["intraday_bias"]
    curr_close = get_current_close(daily_df, date_str)
    next_close = get_next_close(daily_df, date_str)

    if curr_close is None or next_close is None:
        return None
    if bias in {"NO_BIAS", "NEUTRAL", "UNKNOWN"}:
        return None  # no directional call, don't score

    actual_bull = next_close > curr_close

    if bias in BULL_BIASES:
        return "CORRECT" if actual_bull else "INCORRECT"
    if bias in BEAR_BIASES:
        return "CORRECT" if not actual_bull else "INCORRECT"
    return None


def accuracy_breakdown(records: list[dict], group_col: str) -> dict:
    """Return {group_value: {correct, total, pct}} for each unique value of group_col."""
    groups: dict = {}
    for r in records:
        if r.get("outcome") is None:
            continue
        key = str(r.get(group_col, "UNKNOWN"))
        if key not in groups:
            groups[key] = {"correct": 0, "total": 0}
        groups[key]["total"] += 1
        if r["outcome"] == "CORRECT":
            groups[key]["correct"] += 1
    for k, v in groups.items():
        v["pct"] = round(v["correct"] / v["total"] * 100, 1) if v["total"] else None
    return groups


def binomial_test(records: list[dict], filter_entropy: str = "STABLE") -> dict:
    """
    Two-sided binomial test on sessions matching filter_entropy.
    H0: accuracy = 50% (no edge).
    """
    subset = [r for r in records
              if r.get("entropy_state") == filter_entropy and r.get("outcome") is not None]
    n       = len(subset)
    correct = sum(1 for r in subset if r["outcome"] == "CORRECT")
    if n == 0:
        return {"n": 0, "correct": 0, "pct": None, "p_value": None, "significant": None}
    pct    = correct / n
    result = stats.binomtest(correct, n, 0.5, alternative="two-sided")
    return {
        "n":           n,
        "correct":     correct,
        "pct":         round(pct * 100, 1),
        "p_value":     round(result.pvalue, 4),
        "significant": result.pvalue < 0.05,
    }


# ============================================================================
# MAIN
# ============================================================================

def main(start: str | None, end: str | None, report_path: str) -> None:
    daily_df = load_daily_full()
    dates    = discover_log_dates()

    if not dates:
        print("No log files found in logs/ — insufficient data for validation.")
        print("Run freeflow_logger.py for at least 2 trading days to generate logs.")
        return

    # Filter by date range
    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]

    log.info("Validating %d logged trading days", len(dates))

    records = []
    for date_str in dates:
        try:
            pred = run_hindcast(date_str, daily_df)
        except Exception as e:
            log.warning("Hindcast failed for %s: %s", date_str, e)
            continue
        if pred is None:
            log.debug("Skipping %s (insufficient bars)", date_str)
            continue
        outcome = determine_outcome(pred, daily_df)
        pred["outcome"] = outcome
        records.append(pred)
        log.info("%s  bias=%-30s  outcome=%s", date_str, pred["intraday_bias"], outcome or "-")

    if not records:
        print("No scorable predictions found.")
        return

    scored = [r for r in records if r.get("outcome") is not None]
    n_total   = len(records)
    n_scored  = len(scored)
    n_correct = sum(1 for r in scored if r["outcome"] == "CORRECT")

    print("\n" + "=" * 60)
    print(f"  VANTA INTRADAY BIAS — WALK-FORWARD VALIDATION")
    print("=" * 60)
    print(f"  Total days hindcast : {n_total}")
    print(f"  Scorable calls      : {n_scored}  (directional only)")
    print(f"  Correct             : {n_correct}")
    if n_scored:
        print(f"  Overall accuracy    : {n_correct/n_scored*100:.1f}%")

    # Breakdowns
    for col, label in [
        ("entropy_state",    "By Entropy State"),
        ("gamma_regime",     "By Gamma Regime"),
        ("air_pocket_watch", "By Air Pocket Watch"),
    ]:
        bd = accuracy_breakdown(scored, col)
        print(f"\n  {label}:")
        for k, v in bd.items():
            print(f"    {k:25s}  {v['correct']:3d}/{v['total']:3d}  ({v['pct']}%)")

    # Macro alignment
    macro_alignment = []
    for r in scored:
        bias = r.get("intraday_bias", "")
        macro = r.get("macro_bias", "UNKNOWN")
        bull_bias  = bias in BULL_BIASES
        bear_bias  = bias in BEAR_BIASES
        macro_bull = macro in {"STRONG BULL", "LEAN BULL"}
        macro_bear = macro in {"STRONG BEAR", "LEAN BEAR"}
        if (bull_bias and macro_bull) or (bear_bias and macro_bear):
            r["_macro_align"] = "AGREES"
        elif (bull_bias and macro_bear) or (bear_bias and macro_bull):
            r["_macro_align"] = "DISAGREES"
        else:
            r["_macro_align"] = "NEUTRAL"
        macro_alignment.append(r)

    ma_bd = accuracy_breakdown(macro_alignment, "_macro_align")
    print(f"\n  By Macro Alignment:")
    for k, v in ma_bd.items():
        print(f"    {k:25s}  {v['correct']:3d}/{v['total']:3d}  ({v['pct']}%)")

    # Binomial test on STABLE sessions
    binom = binomial_test(scored, "STABLE")
    print(f"\n  Binomial test (STABLE entropy sessions):")
    print(f"    n={binom['n']}  correct={binom['correct']}  acc={binom['pct']}%")
    print(f"    p-value={binom['p_value']}  significant={binom['significant']}")
    print("=" * 60)

    # Write report
    report = {
        "run_date":        datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"),
        "n_total":         n_total,
        "n_scored":        n_scored,
        "n_correct":       n_correct,
        "overall_pct":     round(n_correct / n_scored * 100, 1) if n_scored else None,
        "by_entropy":      accuracy_breakdown(scored, "entropy_state"),
        "by_gamma_regime": accuracy_breakdown(scored, "gamma_regime"),
        "by_air_pocket":   accuracy_breakdown(scored, "air_pocket_watch"),
        "by_macro_align":  ma_bd,
        "binomial_stable": binom,
        "records":         records,
    }

    report_out = os.path.join(BASE_DIR, report_path)
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report saved: %s", report_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward validation of intraday bias")
    parser.add_argument("--start",       metavar="YYYY-MM-DD", help="Validation start date")
    parser.add_argument("--end",         metavar="YYYY-MM-DD", help="Validation end date")
    parser.add_argument("--report-path", default="validation_report.json",
                        help="Output filename (relative to project root)")
    args = parser.parse_args()
    main(start=args.start, end=args.end, report_path=args.report_path)
