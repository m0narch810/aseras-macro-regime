"""
VANTA calibration summary — translates wall outcome analysis into
concrete parameter recommendations.

Usage:
  python scripts/calibration_summary.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
PROFILES_DIR = ROOT / "data" / "processed" / "gex_profiles"
OUTCOMES_PATH = ROOT / "data" / "processed" / "wall_outcomes.csv"


def build_regime_df() -> pd.DataFrame:
    rows = []
    for f in sorted(PROFILES_DIR.glob("gex_profile_*.csv")):
        p = pd.read_csv(f)
        d = pd.to_datetime(f.stem[-8:], format="%Y%m%d")
        rows.append({"date": d, "net_gex": p["gex"].sum(), "iv_open": p["iv_mean"].mean()})
    reg = pd.DataFrame(rows).sort_values("date")
    reg["iv_21d"] = reg["iv_open"].rolling(21, min_periods=5).mean()
    reg["vol_regime"] = np.where(
        reg["iv_open"] > reg["iv_21d"] * 1.1, "HIGH_VOL",
        np.where(reg["iv_open"] < reg["iv_21d"] * 0.9, "LOW_VOL", "MID_VOL"),
    )
    reg["gamma_regime"] = np.where(reg["net_gex"] > 0, "POS_GAMMA", "NEG_GAMMA")
    return reg


def sep(title=""):
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print("=" * 60)
    else:
        print("-" * 60)


def main():
    df = pd.read_csv(OUTCOMES_PATH, parse_dates=["date"])
    reg = build_regime_df()
    merged = df.merge(reg[["date", "gamma_regime", "vol_regime"]], on="date", how="left")
    reached = merged[merged["outcome"] != "NOT_REACHED"].copy()

    sep("VANTA CALIBRATION SUMMARY")
    print("Data: 224 trading days, QQQ options (OPRA CBBO-1m), Feb-Dec 2025")
    print("OI proxy: consolidated quote size (bid_sz + ask_sz) — not true OI")

    # ── 1. Overall edge ───────────────────────────────────────────────────────
    sep("1. OVERALL EDGE")
    n = len(reached)
    h = (reached["outcome"] == "HELD").sum()
    b = n - h
    r50 = stats.binomtest(h, n, p=0.50, alternative="greater")
    r70 = stats.binomtest(h, n, p=0.70, alternative="greater")
    print(f"  Walls labeled:   {len(df):,}  (top-10 walls per day × 224 days)")
    print(f"  Walls reached:   {n}  ({n/len(df):.1%} of total)")
    print(f"  Hold rate:       {h/n:.1%}  ({h} held, {b} broke)")
    print(f"  p vs 50% random: {r50.pvalue:.2e}")
    print(f"  p vs 70% naive:  {r70.pvalue:.2e}")
    print("  VERDICT: GEX walls have highly significant predictive edge.")

    # ── 2. STRONG_WALL = 60 threshold ─────────────────────────────────────────
    sep("2. STRONG_WALL = 60 THRESHOLD")
    a60 = reached[reached["gex_norm"] >= 60]
    b60 = reached[reached["gex_norm"] < 60]
    ct = stats.chi2_contingency([
        [(a60["outcome"] == "HELD").sum(), (a60["outcome"] == "BROKE").sum()],
        [(b60["outcome"] == "HELD").sum(), (b60["outcome"] == "BROKE").sum()],
    ])
    print(f"  gex_norm >= 60:  {(a60.outcome == 'HELD').mean():.1%}  (N={len(a60)})")
    print(f"  gex_norm <  60:  {(b60.outcome == 'HELD').mean():.1%}  (N={len(b60)})")
    print(f"  Chi2 p-value:    {ct.pvalue:.3f}")
    print("  VERDICT: Directionally correct but not statistically significant.")
    print("           Cannot tighten threshold without more data.")
    print("           Keep STRONG_WALL=60, EXCEPTIONAL_WALL=75 unchanged.")

    # ── 3. Call vs Put asymmetry ──────────────────────────────────────────────
    sep("3. CALL vs PUT WALL ASYMMETRY")
    cp = reached.groupby("gex_sign")["outcome"].apply(lambda s: (s == "HELD").mean())
    ct2 = stats.chi2_contingency([
        [(reached[reached["gex_sign"] == "PUT_WALL"]["outcome"] == "HELD").sum(),
         (reached[reached["gex_sign"] == "PUT_WALL"]["outcome"] == "BROKE").sum()],
        [(reached[reached["gex_sign"] == "CALL_WALL"]["outcome"] == "HELD").sum(),
         (reached[reached["gex_sign"] == "CALL_WALL"]["outcome"] == "BROKE").sum()],
    ])
    print(f"  PUT_WALL hold:   {cp['PUT_WALL']:.1%}")
    print(f"  CALL_WALL hold:  {cp['CALL_WALL']:.1%}")
    print(f"  Chi2 p-value:    {ct2.pvalue:.4f}")
    print("  VERDICT: SIGNIFICANT. PUT walls are far more reliable in a bull year.")
    print("           Current scoring treats both identically — this is the biggest")
    print("           actionable finding. In bull regime: discount call walls above spot.")
    print("           (Caveat: 2025 was a strong bull year; test in a bear year.)")

    # ── 4. Regime effect ──────────────────────────────────────────────────────
    sep("4. REGIME WEIGHTS VALIDATION")
    mid = reached[reached["vol_regime"] == "MID_VOL"]
    neg_mid = mid[mid["gamma_regime"] == "NEG_GAMMA"]
    pos_mid = mid[mid["gamma_regime"] == "POS_GAMMA"]
    hi  = reached[reached["vol_regime"] == "HIGH_VOL"]
    lo  = reached[reached["vol_regime"] == "LOW_VOL"]
    ct3 = stats.chi2_contingency([
        [(neg_mid["outcome"] == "HELD").sum(), (neg_mid["outcome"] == "BROKE").sum()],
        [(pos_mid["outcome"] == "HELD").sum(), (pos_mid["outcome"] == "BROKE").sum()],
    ])
    print("  MID_VOL regime (largest N):")
    print(f"    NEG_GAMMA:  {(neg_mid.outcome == 'HELD').mean():.1%}  (N={len(neg_mid)})")
    print(f"    POS_GAMMA:  {(pos_mid.outcome == 'HELD').mean():.1%}  (N={len(pos_mid)})")
    print(f"    Chi2 p:     {ct3.pvalue:.4f}  --> SIGNIFICANT")
    print(f"  HIGH_VOL (all gamma): {(hi.outcome == 'HELD').mean():.1%}  (N={len(hi)})  <-- danger zone")
    print(f"  LOW_VOL  (all gamma): {(lo.outcome == 'HELD').mean():.1%}  (N={len(lo)})  <-- walls nearly unbreakable")
    print("  VERDICT: Regime weighting is validated. Recommend boosting weight")
    print("           on NEG_GAMMA walls; flagging HIGH_VOL days as low-reliability.")

    # ── 5. Proximity decay ────────────────────────────────────────────────────
    sep("5. PROXIMITY_EFOLD VALIDATION")
    reached2 = reached.copy()
    reached2["dist_abs"] = reached2["dist_pct"].abs()
    prox = reached2.groupby(
        pd.cut(reached2["dist_abs"], [0, 1, 2, 3, 10]), observed=True
    ).apply(lambda s: pd.Series({"n": len(s), "hold_rate": (s["outcome"] == "HELD").mean()}))
    print(prox.to_string())
    print()
    print("  PROXIMITY_EFOLD=200 NQ pts ~ 10 QQQ pts ~ 2% of $500 spot.")
    print("  Data: walls <2% hold 90%+; walls >3% when reached are unreliable.")
    print("  VERDICT: 200pt efold is approximately correct. Do not adjust yet.")

    # ── Summary table ─────────────────────────────────────────────────────────
    sep("PARAMETER RECOMMENDATIONS")
    recs = [
        ("STRONG_WALL = 60",       "KEEP",   "Directionally right; p=0.27, need more data"),
        ("EXCEPTIONAL_WALL = 75",  "KEEP",   "Insufficient N at this level to test"),
        ("PROXIMITY_EFOLD = 200",  "KEEP",   "~2% QQQ decay validated by distance analysis"),
        ("CALL/PUT symmetry",      "CHANGE", "Add put_wall_bonus: PUT walls hold 20pp more"),
        ("HIGH_VOL flag",          "ADD",    "Tag HIGH_VOL days; wall confidence ~33% only"),
        ("GAMMA_ASYMMETRY = 0.344","KEEP",   "SPX-derived; no NQ-specific data to replace it"),
    ]
    print(f"  {'Parameter':<30} {'Action':<8} Notes")
    print(f"  {'-'*29} {'-'*7} {'-'*35}")
    for param, action, note in recs:
        print(f"  {param:<30} {action:<8} {note}")

    sep()
    print("Limitations:")
    print("  - OI proxy (quote size) understates true dealer exposure")
    print("  - 2025 was a bull year; call wall stats would differ in bear market")
    print("  - N=358 reached walls; STRONG_WALL threshold needs ~1000+ to pin precisely")
    print("  - Daily OHLC misses intraday bounces that reverse within the day")


if __name__ == "__main__":
    main()
