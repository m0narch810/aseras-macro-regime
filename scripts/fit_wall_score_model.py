"""
Data-driven wall scoring model.

Validates the current arbitrary composite score against real hold/break outcomes,
then fits a logistic regression whose coefficients become the actual learned weights.

Usage:
  python scripts/fit_wall_score_model.py

Outputs:
  models/wall_score_model.pkl       -- trained logistic regression
  models/wall_score_features.json   -- feature list + coefficients for JS port
  data/processed/wall_score_eval.csv -- per-wall scores + probabilities for inspection
"""

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

ROOT = Path(__file__).parent.parent
PROFILES_DIR  = ROOT / "data" / "processed" / "gex_profiles"
OUTCOMES_PATH = ROOT / "data" / "processed" / "wall_outcomes.csv"
MODELS_DIR    = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Current scoring formula (mirrored from options.js) ───────────────────────
# Regime weights from REGIME_WEIGHTS table in lib/options.js.
# We need a vol_regime + gamma_regime to pick a row; we derive them from iv_mean
# and net_gex just like the JS does.

REGIME_WEIGHTS = {
    "EXPANSION": {
        "POSITIVE":  dict(gex=0.22, vex=0.38, charmex=0.17, oi=0.14, dag=0.09),
        "NEAR_FLIP": dict(gex=0.20, vex=0.38, charmex=0.17, oi=0.15, dag=0.10),
        "NEGATIVE":  dict(gex=0.10, vex=0.48, charmex=0.17, oi=0.14, dag=0.11),
    },
    "NEUTRAL": {
        "POSITIVE":  dict(gex=0.42, vex=0.22, charmex=0.14, oi=0.14, dag=0.08),
        "NEAR_FLIP": dict(gex=0.32, vex=0.28, charmex=0.15, oi=0.15, dag=0.10),
        "NEGATIVE":  dict(gex=0.20, vex=0.36, charmex=0.14, oi=0.16, dag=0.14),
    },
    "CONTRACTION": {
        "POSITIVE":  dict(gex=0.60, vex=0.10, charmex=0.14, oi=0.14, dag=0.02),
        "NEAR_FLIP": dict(gex=0.50, vex=0.15, charmex=0.15, oi=0.15, dag=0.05),
        "NEGATIVE":  dict(gex=0.36, vex=0.24, charmex=0.16, oi=0.15, dag=0.09),
    },
}


def classify_vol_regime(iv):
    if iv >= 0.30:  return "EXPANSION"
    if iv >= 0.20:  return "NEUTRAL"
    return "CONTRACTION"


def classify_gamma_regime(net_gex, futures_price, flip_price=None):
    # Without per-day flip data here we approximate: positive net_gex → POSITIVE regime
    if net_gex > 0:  return "POSITIVE"
    return "NEGATIVE"


def get_weights(vol_regime, gamma_regime):
    return REGIME_WEIGHTS[vol_regime][gamma_regime]


def score_current(row):
    w = get_weights(row["vol_regime"], row["gamma_regime"])
    # dag not in our data → 0
    s = (row["gex_norm"] * w["gex"]
       + row["vex_norm"] * w["vex"]
       + row["charmex_norm"] * w["charmex"]
       + row["oi_norm"] * w["oi"]) * 100
    return s


# ── Data loading ──────────────────────────────────────────────────────────────

def load_profiles():
    dfs = []
    for f in sorted(PROFILES_DIR.glob("gex_profile_*.csv")):
        dfs.append(pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True)


def build_dataset():
    outcomes = pd.read_csv(OUTCOMES_PATH, parse_dates=["date"])
    profiles = load_profiles()
    profiles["date"] = pd.to_datetime(profiles["date"])

    # Merge to get charmex + normalized columns from profiles
    merged = outcomes.merge(
        profiles[["date", "strike", "charmex", "vex_norm", "charmex_norm", "oi_norm"]],
        on=["date", "strike"],
        how="left",
    )

    # Regime labels
    merged["vol_regime"]   = merged["iv_mean"].apply(classify_vol_regime)
    merged["gamma_regime"] = merged.apply(
        lambda r: classify_gamma_regime(r["gex"], r["spot"]), axis=1
    )

    # Current composite score
    merged["score_current"] = merged.apply(score_current, axis=1)

    return merged


# ── Evaluation helpers ────────────────────────────────────────────────────────

def sep(title=""):
    if title:
        print(f"\n{'=' * 65}")
        print(f"  {title}")
        print("=" * 65)
    else:
        print("-" * 65)


def auc_and_brier(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    bs  = brier_score_loss(y_true, y_prob)
    return auc, bs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = build_dataset()
    reached = df[df["outcome"] != "NOT_REACHED"].copy()
    reached["held"] = (reached["outcome"] == "HELD").astype(int)

    n_held  = reached["held"].sum()
    n_broke = len(reached) - n_held

    sep("WALL SCORE MODEL — DATA-DRIVEN VALIDATION")
    print(f"  Reached walls: {len(reached)}  ({n_held} held, {n_broke} broke)")
    print(f"  Hold base rate: {n_held/len(reached):.1%}")
    print(f"  Features available: gex_norm, vex_norm, charmex_norm, oi_norm,")
    print(f"                      dist_pct, is_put_wall, vol_regime, gamma_regime")

    # ── 1. Validate current score ─────────────────────────────────────────────
    sep("1. CURRENT COMPOSITE SCORE — DOES IT PREDICT HOLDS?")
    cur = reached["score_current"].values
    held = reached["held"].values

    # Normalise current score to 0-1 probability for fair comparison
    cur_prob = (cur - cur.min()) / (cur.max() - cur.min() + 1e-9)
    auc_cur, bs_cur = auc_and_brier(held, cur_prob)

    print(f"  AUC-ROC:     {auc_cur:.4f}  (0.5=random, 1.0=perfect)")
    print(f"  Brier score: {bs_cur:.4f}  (lower is better; 0=perfect)")

    # Point-biserial: does higher score → more holds?
    r, p = stats.pointbiserialr(cur, held)
    print(f"  Correlation with held (r): {r:.4f}  p={p:.3f}")

    if auc_cur < 0.55:
        print("  VERDICT: Current score is essentially random. The regime-weight")
        print("           composite adds no predictive value over base rates.")
    elif auc_cur < 0.65:
        print("  VERDICT: Current score has weak signal. Logistic regression")
        print("           from actual data will significantly outperform it.")
    else:
        print("  VERDICT: Current score has some signal but may not be calibrated.")

    # Score decile analysis
    reached["score_decile"] = pd.qcut(reached["score_current"], q=5, labels=False, duplicates="drop")
    decile_holds = reached.groupby("score_decile")["held"].mean()
    print(f"\n  Hold rate by score quintile (Q0=lowest, Q4=highest):")
    for q, hr in decile_holds.items():
        n = (reached["score_decile"] == q).sum()
        print(f"    Q{q}: {hr:.1%}  (N={n})")
    print("  If scoring worked, hold rate should increase monotonically Q0 -> Q4.")

    # ── 2. Fit logistic regression ────────────────────────────────────────────
    sep("2. FITTING DATA-DRIVEN MODEL")

    feature_cols = ["gex_norm", "vex_norm", "charmex_norm", "oi_norm", "dist_pct"]
    cat_features = {
        "is_put":     (reached["gex_sign"] == "PUT_WALL").astype(float),
        "is_high_vol":(reached["vol_regime"] == "EXPANSION").astype(float),
        "is_neg_gam": (reached["gamma_regime"] == "NEGATIVE").astype(float),
    }

    X = reached[feature_cols].copy()
    for name, col in cat_features.items():
        X[name] = col.values

    X = X.fillna(0)
    y = held

    all_feature_names = feature_cols + list(cat_features.keys())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balanced class weight handles the 323:35 imbalance
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    # Cross-validated AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(lr, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"  5-fold CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"  vs current score AUC: {auc_cur:.4f}")

    # Fit on full dataset for coefficient inspection
    lr.fit(X_scaled, y)
    y_prob = lr.predict_proba(X_scaled)[:, 1]
    auc_lr, bs_lr = auc_and_brier(y, y_prob)
    print(f"  Full-fit AUC:    {auc_lr:.4f}")
    print(f"  Full-fit Brier:  {bs_lr:.4f}")

    sep("3. LEARNED FEATURE COEFFICIENTS (what actually matters)")
    coef = dict(zip(all_feature_names, lr.coef_[0]))
    coef_sorted = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"  {'Feature':<20} {'Coeff':>8}  Direction")
    print(f"  {'-'*19} {'-'*8}  {'-'*20}")
    for feat, c in coef_sorted:
        direction = "HOLDS MORE" if c > 0 else "BREAKS MORE"
        print(f"  {feat:<20} {c:>8.4f}  {direction}")
    print()
    print("  Intercept:", round(lr.intercept_[0], 4))
    print()
    print("  These are the data-driven weights. Larger absolute value = more")
    print("  predictive. The current scoring system ignores these relationships.")

    # ── 3. Hold probability by decile of fitted model ────────────────────────
    sep("4. CALIBRATION: FITTED HOLD PROBABILITY vs ACTUAL HOLD RATE")
    reached2 = reached.copy()
    reached2["hold_prob"] = y_prob
    reached2["prob_decile"] = pd.qcut(reached2["hold_prob"], q=5, labels=False, duplicates="drop")
    cal = reached2.groupby("prob_decile").agg(
        n=("held", "count"),
        actual_hold=("held", "mean"),
        mean_prob=("hold_prob", "mean"),
    )
    print(f"  {'Quintile':<10} {'N':>5}  {'Predicted':>10}  {'Actual Hold':>12}")
    print(f"  {'-'*9} {'-'*5}  {'-'*10}  {'-'*12}")
    for q, row in cal.iterrows():
        print(f"  Q{q:<9} {row['n']:>5}  {row['mean_prob']:>10.1%}  {row['actual_hold']:>12.1%}")
    print()
    print("  Well-calibrated model: Predicted ~= Actual across all quintiles.")

    # ── 4. PUT vs CALL decomposed by model ───────────────────────────────────
    sep("5. PUT vs CALL — MODEL VS REALITY")
    for wall_type in ["PUT_WALL", "CALL_WALL"]:
        sub = reached2[reached2["gex_sign"] == wall_type]
        actual = sub["held"].mean()
        predicted = sub["hold_prob"].mean()
        print(f"  {wall_type:<12}: actual {actual:.1%}  |  model predicts {predicted:.1%}  (N={len(sub)})")

    # ── 5. Save model ─────────────────────────────────────────────────────────
    sep("6. SAVING MODEL")
    model_path = MODELS_DIR / "wall_score_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": lr, "scaler": scaler, "features": all_feature_names}, f)
    print(f"  Saved: {model_path}")

    feat_export = {
        "features": all_feature_names,
        "coefficients": {k: round(v, 6) for k, v in coef.items()},
        "intercept": round(float(lr.intercept_[0]), 6),
        "scaler_mean": dict(zip(all_feature_names, [round(m, 6) for m in scaler.mean_])),
        "scaler_scale": dict(zip(all_feature_names, [round(s, 6) for s in scaler.scale_])),
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std": round(float(cv_auc.std()), 4),
        "n_held": int(n_held),
        "n_broke": int(n_broke),
    }
    feat_path = MODELS_DIR / "wall_score_features.json"
    with open(feat_path, "w") as f:
        json.dump(feat_export, f, indent=2)
    print(f"  Saved: {feat_path}")

    # Save per-wall eval for inspection
    eval_cols = ["date", "strike", "gex_sign", "gex_norm", "vex_norm", "charmex_norm",
                 "oi_norm", "dist_pct", "vol_regime", "gamma_regime",
                 "score_current", "hold_prob", "outcome", "held"]
    reached2[eval_cols].to_csv(ROOT / "data" / "processed" / "wall_score_eval.csv", index=False)
    print(f"  Saved: data/processed/wall_score_eval.csv")

    sep()
    print("SUMMARY")
    print(f"  Current score AUC:     {auc_cur:.4f}")
    print(f"  Logistic model CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    improvement = cv_auc.mean() - auc_cur
    if improvement > 0.05:
        print(f"  Data-driven model is {improvement:.2f} AUC better. Replace current scoring.")
    elif improvement > 0.01:
        print(f"  Modest improvement ({improvement:.2f} AUC). Current structure is directionally")
        print(f"  right but coefficients need updating.")
    else:
        print(f"  Similar performance. Limited data (N={len(reached)}) may be the bottleneck.")
        print(f"  More labeled outcomes needed before scoring overhaul is justified.")


if __name__ == "__main__":
    main()
