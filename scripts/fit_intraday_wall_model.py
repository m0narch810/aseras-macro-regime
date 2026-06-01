"""
Data-driven wall scoring model using intraday touch events.

Fits on 1,925 intraday touch events (vs 358 daily labels) with a
45/55 hold/break split — balanced enough to actually learn from.

New features vs daily model:
  time_of_day    — morning walls behave differently from afternoon
  approach_vel   — momentum into the wall (fast approach = more likely to break)
  prior_touches  — wall that has already held 2x is different from fresh wall
  approach_dir   — FROM_BELOW (call wall test) vs FROM_ABOVE (put wall test)

Outputs:
  models/wall_score_intraday.pkl        — trained model (logistic + XGBoost)
  models/wall_score_intraday.json       — coefficients + scaler for JS port
  data/processed/intraday_score_eval.csv

Usage:
  python scripts/fit_intraday_wall_model.py
  python scripts/fit_intraday_wall_model.py --combined   # merge QQQ + SPY datasets
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT           = Path(__file__).parent.parent
DATA_PATH      = ROOT / "data" / "processed" / "intraday_touches.csv"
SPY_DATA_PATH  = ROOT / "data" / "processed" / "spy_touches.csv"
MODELS         = ROOT / "models"
MODELS.mkdir(exist_ok=True)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def sep(title=""):
    if title:
        print(f"\n{'=' * 65}")
        print(f"  {title}")
        print("=" * 65)
    else:
        print("-" * 65)


# ── Current scoring (same weights as options.js) ──────────────────────────────
REGIME_WEIGHTS = {
    "EXPANSION":   {"POSITIVE": dict(gex=0.22, vex=0.38, charmex=0.17, oi=0.14),
                    "NEGATIVE":  dict(gex=0.10, vex=0.48, charmex=0.17, oi=0.14)},
    "NEUTRAL":     {"POSITIVE": dict(gex=0.42, vex=0.22, charmex=0.14, oi=0.14),
                    "NEGATIVE":  dict(gex=0.20, vex=0.36, charmex=0.14, oi=0.16)},
    "CONTRACTION": {"POSITIVE": dict(gex=0.60, vex=0.10, charmex=0.14, oi=0.14),
                    "NEGATIVE":  dict(gex=0.36, vex=0.24, charmex=0.16, oi=0.15)},
}

def score_current(row):
    gamma = "POSITIVE" if row["is_put"] == 0 else "NEGATIVE"
    w = REGIME_WEIGHTS.get(row["vol_regime"], REGIME_WEIGHTS["NEUTRAL"]).get(gamma, REGIME_WEIGHTS["NEUTRAL"]["POSITIVE"])
    return (row["gex_norm"] * w["gex"]
          + row["vex_norm"] * w["vex"]
          + row["charmex_norm"] * w["charmex"]
          + row["oi_norm"] * w["oi"]) * 100


def auc_brier(y, p):
    return roc_auc_score(y, p), brier_score_loss(y, p)


def build_features(df):
    X = pd.DataFrame({
        # Greek magnitude (normalized 0-100 within day)
        "gex_norm":           df["gex_norm"],
        "vex_norm":           df["vex_norm"],
        "charmex_norm":       df["charmex_norm"],
        "oi_norm":            df["oi_norm"],
        # Greek ratios: relative dominance at the wall
        "vex_over_gex":       df["vex_over_gex"],
        "charmex_over_gex":   df["charmex_over_gex"],
        # Distance: signed — positive = wall above spot (CALL), negative = below (PUT)
        "dist_pct":           df["dist_pct"],
        # Vol regime
        "is_high_vol":        df["is_high_vol"],
        "is_contraction":     (df["vol_regime"] == "CONTRACTION").astype(float),
        # Gamma structure
        "is_put":             df["is_put"],
        "in_neg_gamma":       df["in_neg_gamma"],
        "wall_above_flip":    df["wall_above_flip"],
        # Multi-Greek confluence
        "confluence":         df["confluence"] if "confluence" in df.columns else 0,
        # Intraday context
        "time_of_day":        df["time_of_day"],
        "approach_vel":       df["approach_vel"],
        "from_below":         (df["approach_dir"] == "FROM_BELOW").astype(float),
        # Volume profile structural alignment (previous day)
        "pd_on_hvn":          df["pd_on_hvn"]         if "pd_on_hvn"         in df.columns else 0,
        "pd_on_lvn":          df["pd_on_lvn"]         if "pd_on_lvn"         in df.columns else 0,
        "pd_in_value_area":   df["pd_in_value_area"]  if "pd_in_value_area"  in df.columns else 0,
        "pd_dist_to_poc":     df["pd_dist_to_poc"]    if "pd_dist_to_poc"    in df.columns else 0,
        "pd_dist_to_hvn":     df["pd_dist_to_hvn"]    if "pd_dist_to_hvn"    in df.columns else 0,
        "pd_dist_to_lvn":     df["pd_dist_to_lvn"]    if "pd_dist_to_lvn"    in df.columns else 0,
        # Volume profile structural alignment (previous week)
        "pw_on_hvn":          df["pw_on_hvn"]         if "pw_on_hvn"         in df.columns else 0,
        "pw_on_lvn":          df["pw_on_lvn"]         if "pw_on_lvn"         in df.columns else 0,
        "pw_in_value_area":   df["pw_in_value_area"]  if "pw_in_value_area"  in df.columns else 0,
        "pw_dist_to_poc":     df["pw_dist_to_poc"]    if "pw_dist_to_poc"    in df.columns else 0,
        # Composite: on HVN from either lookback (replaces/supplements confluence in reversal filter)
        "vp_aligned":         df["vp_aligned"]        if "vp_aligned"        in df.columns else 0,
    })
    return X.fillna(0)


def load_combined_dataset(verbose: bool = True) -> pd.DataFrame:
    """
    Load QQQ intraday touches and SPY EOD touches, tag sources, and merge.

    QQQ touches have a full VP feature set; SPY touches have VP columns zeroed.
    Both must have the same model feature columns — missing columns are filled
    with 0 to ensure the feature matrix is identical.

    Returns the combined DataFrame with a 'source' column ('qqq' or 'spy').
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"QQQ touches not found: {DATA_PATH}\n"
                                "  Run scripts/label_intraday_touches.py first.")
    if not SPY_DATA_PATH.exists():
        raise FileNotFoundError(f"SPY touches not found: {SPY_DATA_PATH}\n"
                                "  Run scripts/label_spy_touches.py first.")

    qqq = pd.read_csv(DATA_PATH)
    spy = pd.read_csv(SPY_DATA_PATH)

    # Tag source if not already present
    if "source" not in qqq.columns:
        qqq["source"] = "qqq"
    if "source" not in spy.columns:
        spy["source"] = "spy"

    # Ensure both DataFrames have the same columns (fill missing with 0)
    all_cols = sorted(set(qqq.columns) | set(spy.columns))
    for col in all_cols:
        if col not in qqq.columns:
            qqq[col] = 0
        if col not in spy.columns:
            spy[col] = 0

    combined = pd.concat([qqq, spy], ignore_index=True)

    if verbose:
        print()
        print("  Source breakdown:")
        for src, grp in combined.groupby("source"):
            n     = len(grp)
            hold  = grp["held"].mean()
            print(f"    {src:<6}  N={n:>7,}  hold_rate={hold:.1%}")

        print()
        print("  Regime breakdown (combined):")
        for vr, grp in combined.groupby("vol_regime"):
            n    = len(grp)
            hold = grp["held"].mean()
            print(f"    {vr:<15}  N={n:>7,}  hold_rate={hold:.1%}")

        print()
        print("  Year breakdown (combined):")
        combined["_year"] = pd.to_datetime(combined["date"], errors="coerce").dt.year
        for yr, grp in combined.groupby("_year"):
            n    = len(grp)
            hold = grp["held"].mean()
            print(f"    {yr}  N={n:>7,}  hold_rate={hold:.1%}")
        combined.drop(columns=["_year"], inplace=True)

    return combined


def main():
    parser = argparse.ArgumentParser(description="Train intraday wall hold model")
    parser.add_argument(
        "--combined", action="store_true",
        help="Merge intraday_touches.csv + spy_touches.csv before training",
    )
    args = parser.parse_args()

    if args.combined:
        sep("COMBINED QQQ + SPY INTRADAY WALL TOUCH MODEL")
        print("  Loading combined dataset ...")
        df = load_combined_dataset(verbose=True)
        print(f"\n  Total samples:  {len(df)}")
    else:
        df = pd.read_csv(DATA_PATH)

    y  = df["held"].values

    sep("INTRADAY WALL TOUCH MODEL" + (" [COMBINED]" if args.combined else ""))
    print(f"  Samples:     {len(df)}")
    print(f"  Held:        {y.sum()}  ({y.mean():.1%})")
    print(f"  Broke:       {len(y) - y.sum()}  ({1 - y.mean():.1%})")
    print(f"  XGBoost:     {'available' if HAS_XGB else 'not installed — logistic only'}")

    # ── 1. Baseline: current composite score ─────────────────────────────────
    sep("1. CURRENT SCORING SYSTEM vs INTRADAY OUTCOMES")
    df["score_current"] = df.apply(score_current, axis=1)
    cur_prob = (df["score_current"] - df["score_current"].min()) / (df["score_current"].max() - df["score_current"].min() + 1e-9)
    auc_cur, bs_cur = auc_brier(y, cur_prob.values)
    r, p = stats.pointbiserialr(df["score_current"], y)

    print(f"  AUC-ROC:        {auc_cur:.4f}  (0.5=random, 1.0=perfect)")
    print(f"  Brier score:    {bs_cur:.4f}")
    print(f"  Correlation:    r={r:.4f}  p={p:.4f}")

    df["score_quintile"] = pd.qcut(df["score_current"], q=5, labels=False, duplicates="drop")
    print("\n  Hold rate by current-score quintile:")
    for q, grp in df.groupby("score_quintile"):
        print(f"    Q{q}: {grp['held'].mean():.1%}  (N={len(grp)})")
    print("  Expected if scoring works: monotonic increase Q0 -> Q4")

    # ── 2. Feature analysis ───────────────────────────────────────────────────
    sep("2. RAW FEATURE CORRELATIONS WITH HOLD")
    feature_names = ["gex_norm", "vex_norm", "charmex_norm", "oi_norm",
                     "vex_over_gex", "charmex_over_gex", "dist_pct",
                     "is_high_vol", "is_put", "in_neg_gamma", "wall_above_flip",
                     "time_of_day", "approach_vel"]
    print(f"  {'Feature':<20} {'r':>7}  {'p':>9}  Signal")
    print(f"  {'-'*19} {'-'*7}  {'-'*9}  {'-'*20}")
    for feat in feature_names:
        vals = df[feat].abs() if feat == "dist_pct" else df[feat]
        r2, p2 = stats.pointbiserialr(vals, y)
        sig = "**strong**" if abs(r2) > 0.15 else ("*moderate*" if abs(r2) > 0.07 else "weak")
        print(f"  {feat:<20} {r2:>7.4f}  {p2:>9.4f}  {sig}")

    # ── 3. Logistic regression ────────────────────────────────────────────────
    sep("3. LOGISTIC REGRESSION (interpretable weights)")
    X        = build_features(df)
    feat_cols = X.columns.tolist()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_auc_lr = cross_val_score(lr, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"  5-fold CV AUC:  {cv_auc_lr.mean():.4f} +/- {cv_auc_lr.std():.4f}")
    print(f"  vs current:     {auc_cur:.4f}")

    lr.fit(X_scaled, y)
    prob_lr = lr.predict_proba(X_scaled)[:, 1]
    auc_lr, bs_lr = auc_brier(y, prob_lr)
    print(f"  Full-fit AUC:   {auc_lr:.4f}")
    print(f"  Full-fit Brier: {bs_lr:.4f}")

    print("\n  Learned coefficients (larger abs = more predictive):")
    coef = dict(zip(feat_cols, lr.coef_[0]))
    coef_sorted = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"  {'Feature':<20} {'Coeff':>8}  Meaning")
    print(f"  {'-'*19} {'-'*8}  {'-'*30}")
    for feat, c in coef_sorted:
        meaning = "holds more" if c > 0 else "breaks more"
        print(f"  {feat:<20} {c:>8.4f}  {meaning}")

    # ── 4. XGBoost ────────────────────────────────────────────────────────────
    if HAS_XGB:
        sep("4. XGBOOST (non-linear interactions)")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(len(y) - y.sum()) / y.sum(),
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        cv_auc_xgb = cross_val_score(xgb_model, X, y, cv=cv, scoring="roc_auc")
        print(f"  5-fold CV AUC:  {cv_auc_xgb.mean():.4f} +/- {cv_auc_xgb.std():.4f}")

        xgb_model.fit(X, y)
        prob_xgb = xgb_model.predict_proba(X)[:, 1]
        auc_xgb, bs_xgb = auc_brier(y, prob_xgb)
        print(f"  Full-fit AUC:   {auc_xgb:.4f}")

        importances = dict(zip(feat_cols, xgb_model.feature_importances_))
        imp_sorted  = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        print("\n  Feature importances (XGBoost gain):")
        for feat, imp in imp_sorted:
            bar = "#" * int(imp * 50)
            print(f"  {feat:<20} {imp:.4f}  {bar}")
    else:
        cv_auc_xgb = None
        xgb_model  = None
        prob_xgb   = None

    # ── 5. Calibration ────────────────────────────────────────────────────────
    sep("5. CALIBRATION: PREDICTED vs ACTUAL HOLD RATE")
    df["prob_lr"] = prob_lr
    df["prob_q"]  = pd.qcut(df["prob_lr"], q=5, labels=False, duplicates="drop")
    cal = df.groupby("prob_q").agg(
        n=("held", "count"),
        actual=("held", "mean"),
        pred=("prob_lr", "mean"),
    )
    print(f"  {'Q':<4} {'N':>5}  {'Predicted':>10}  {'Actual':>8}")
    for q, row in cal.iterrows():
        print(f"  Q{q:<3} {row['n']:>5}  {row['pred']:>10.1%}  {row['actual']:>8.1%}")
    print("  Well-calibrated: Predicted close to Actual in each row")

    # ── 6. Key breakdowns ─────────────────────────────────────────────────────
    sep("6. HOLD RATE BY KEY DIMENSIONS")
    print("  Vol regime:")
    for vr, grp in df.groupby("vol_regime"):
        print(f"    {vr:<15} hold={grp['held'].mean():.1%}  N={len(grp)}")

    print("  Wall type:")
    df["wall_type"] = np.where(df["is_put"] == 1, "PUT_WALL", "CALL_WALL")
    for wt, grp in df.groupby("wall_type"):
        print(f"    {wt:<15} hold={grp['held'].mean():.1%}  N={len(grp)}")

    print("  Approach direction:")
    for ad, grp in df.groupby("approach_dir"):
        print(f"    {ad:<15} hold={grp['held'].mean():.1%}  N={len(grp)}")

    print("  Prior touches today (0=fresh wall, 1+=already tested):")
    df["prior_grp"] = df["prior_touches"].clip(0, 3)
    for pt, grp in df.groupby("prior_grp"):
        label = f"{pt}+" if pt == 3 else str(pt)
        print(f"    prior={label:<4}  hold={grp['held'].mean():.1%}  N={len(grp)}")

    print("  Time of day bucket:")
    df["tod_bucket"] = pd.cut(df["time_of_day"],
        bins=[9.5, 10.5, 12.0, 14.0, 16.0],
        labels=["open(9:30-10:30)", "mid-morn(10:30-12)", "midday(12-14)", "afternoon(14-16)"]
    )
    for tod, grp in df.groupby("tod_bucket", observed=True):
        print(f"    {str(tod):<22} hold={grp['held'].mean():.1%}  N={len(grp)}")

    # ── 7. Save ────────────────────────────────────────────────────────────────
    sep("7. SAVING")
    best_model = xgb_model if HAS_XGB else lr
    best_prob  = prob_xgb  if HAS_XGB else prob_lr

    pkl_path = MODELS / "wall_score_intraday.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"lr": lr, "scaler": scaler, "features": feat_cols,
                     "xgb": xgb_model}, f)
    print(f"  Saved: {pkl_path}")

    source_breakdown = {}
    if "source" in df.columns:
        for src, grp in df.groupby("source"):
            source_breakdown[src] = {
                "n":         int(len(grp)),
                "n_held":    int(grp["held"].sum()),
                "hold_rate": round(float(grp["held"].mean()), 4),
            }

    export = {
        "features": feat_cols,
        "lr_coefficients": {k: round(float(v), 6) for k, v in coef.items()},
        "lr_intercept": round(float(lr.intercept_[0]), 6),
        "scaler_mean":  {k: round(float(v), 6) for k, v in zip(feat_cols, scaler.mean_)},
        "scaler_scale": {k: round(float(v), 6) for k, v in zip(feat_cols, scaler.scale_)},
        "cv_auc_lr_mean": round(float(cv_auc_lr.mean()), 4),
        "cv_auc_lr_std":  round(float(cv_auc_lr.std()),  4),
        "cv_auc_xgb_mean": round(float(cv_auc_xgb.mean()), 4) if cv_auc_xgb is not None else None,
        "n_samples": len(df),
        "n_held": int(y.sum()),
        "n_broke": int(len(y) - y.sum()),
        "xgb_importances": {k: round(float(v), 6) for k, v in imp_sorted} if HAS_XGB else None,
        "training_dataset": "combined_qqq_spy" if args.combined else "qqq_only",
        "source_breakdown": source_breakdown if source_breakdown else None,
    }
    json_path = MODELS / "wall_score_intraday.json"
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"  Saved: {json_path}")

    eval_cols = ["date", "touch_time", "wall_id", "approach_dir", "time_of_day",
                 "approach_vel", "prior_touches", "gex_norm", "vex_norm",
                 "charmex_norm", "oi_norm", "dist_pct", "is_put", "vol_regime",
                 "score_current", "held", "outcome"]
    if "source" in df.columns:
        eval_cols.append("source")
    eval_df = df[[c for c in eval_cols if c in df.columns]].copy()
    eval_df["prob_lr"] = prob_lr
    if prob_xgb is not None:
        eval_df["prob_xgb"] = prob_xgb
    eval_df.to_csv(ROOT / "data" / "processed" / "intraday_score_eval.csv", index=False)
    print(f"  Saved: data/processed/intraday_score_eval.csv")

    sep()
    print("SUMMARY")
    print(f"  Current score AUC (intraday):  {auc_cur:.4f}")
    print(f"  Logistic CV AUC:               {cv_auc_lr.mean():.4f} +/- {cv_auc_lr.std():.4f}")
    if cv_auc_xgb is not None:
        print(f"  XGBoost CV AUC:                {cv_auc_xgb.mean():.4f} +/- {cv_auc_xgb.std():.4f}")
    print()
    best_auc = cv_auc_xgb.mean() if cv_auc_xgb is not None else cv_auc_lr.mean()
    improvement = best_auc - auc_cur
    if improvement > 0.08:
        print(f"  VERDICT: Data-driven model beats current scoring by {improvement:.2f} AUC.")
        print(f"  Current scoring system should be replaced.")
    elif improvement > 0.03:
        print(f"  VERDICT: Meaningful improvement ({improvement:.2f} AUC).")
        print(f"  Intraday features add signal. Coefficients should update weights.")
    else:
        print(f"  VERDICT: Marginal improvement. Intraday features may need refinement.")


if __name__ == "__main__":
    main()
