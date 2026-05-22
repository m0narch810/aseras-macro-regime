#!/usr/bin/env python3
"""
scripts/09_intraday_bias.py -- intraday NQ bias classifier.

Combines return entropy gate, PCA price structure, options flow
state (H_GEX_norm, gamma regime, top wall), MM intensification,
and the weekly macro bias as a fixed prior.

Inputs (all strictly past data -- no lookahead):
  data/processed/NQ_daily_clean.csv
  bias_output.json
  /.netlify/functions/levels  OR  levels_data.json
  logs/levels_YYYY-MM-DD.csv  (optional, for MM intensification)
  models/pca_intraday.pkl     (auto-created/refreshed)

Output: intraday_bias.json

Usage:
    python scripts/09_intraday_bias.py
    python scripts/09_intraday_bias.py --dry-run
    python scripts/09_intraday_bias.py --date 2026-05-22
"""
import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests
from dotenv import load_dotenv
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# -- PATHS -------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR  = os.path.join(BASE_DIR, "logs")
PCA_PATH  = os.path.join(MODEL_DIR, "pca_intraday.pkl")
OUT_PATH  = os.path.join(BASE_DIR, "intraday_bias.json")
BIAS_PATH = os.path.join(BASE_DIR, "bias_output.json")
LVL_PATH  = os.path.join(BASE_DIR, "levels_data.json")

ET = pytz.timezone("America/New_York")
NETLIFY_URL = os.getenv("NETLIFY_URL", "").rstrip("/")

# -- CONSTANTS (VALIDATE before changing) ------------------------------------
ENTROPY_WINDOW        = 20      # bars for return histogram
ENTROPY_BINS          = 10      # equal-width bins
ENTROPY_LOOKBACK      = 252     # bars for dynamic threshold
ENTROPY_PCTILE        = 75.0    # percentile cutoff for CRITICAL state
ENTROPY_MIN_BARS      = 60      # minimum bars required

PCA_COMPONENTS        = 3
REFIT_INTERVAL        = 63      # trading days between PCA refits

STRONG_WALL           = 60.0    # score threshold for strong wall
EXCEPTIONAL_WALL      = 75.0    # score threshold for exceptional wall
NEAR_FLIP_BUFFER      = 50.0    # NQ points either side of flip for NEAR_FLIP
AIR_POCKET_PROXIMITY  = 150.0   # NQ points: wall must be within this of flip
H_GEX_CONFIDENCE_CUT  = 0.6    # H_GEX_norm above this penalizes confidence
PROXIMITY_HALFLIFE    = 200.0   # NQ points for proximity-weighted wall scoring

MACRO_BULL = {"STRONG BULL", "LEAN BULL"}
MACRO_BEAR = {"STRONG BEAR", "LEAN BEAR"}


# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

def load_daily(pred_date: datetime) -> pd.DataFrame:
    """Load NQ daily clean CSV; return only rows strictly before pred_date."""
    path = os.path.join(PROC_DIR, "NQ_daily_clean.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # LEAKAGE GUARD: strict less-than — pred_date bar excluded
    df = df[df["date"] < pd.Timestamp(pred_date.date())]
    if len(df) < ENTROPY_MIN_BARS:
        raise ValueError(f"Insufficient daily bars: {len(df)} < {ENTROPY_MIN_BARS}")
    return df


def load_macro_bias() -> dict:
    """Load weekly macro bias from bias_output.json."""
    if not os.path.exists(BIAS_PATH):
        log.warning("bias_output.json not found; macro_bias=UNKNOWN")
        return {"confluence": "UNKNOWN", "macro_regime": {"name": "UNKNOWN", "score": 0}}
    with open(BIAS_PATH) as f:
        return json.load(f)


def load_levels(dry_run: bool) -> dict | None:
    """Fetch live levels from Netlify function or fall back to levels_data.json."""
    if not dry_run and NETLIFY_URL:
        url = f"{NETLIFY_URL}/.netlify/functions/levels"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data.get("error"):
                return data
            log.warning("Netlify levels returned error: %s", data.get("message"))
        except Exception as e:
            log.warning("Netlify fetch failed: %s", e)

    if os.path.exists(LVL_PATH):
        log.info("Using levels_data.json (local snapshot)")
        with open(LVL_PATH) as f:
            return json.load(f)

    log.warning("No levels data available")
    return None


# ============================================================================
# SECTION 2: RETURN ENTROPY
# ============================================================================

def compute_return_entropy(daily_df: pd.DataFrame, pred_date: datetime) -> dict:
    """
    Shannon entropy gate.  H_returns is computed on the last ENTROPY_WINDOW
    log-returns.  Threshold = 75th-pctile of H_returns over the prior
    ENTROPY_LOOKBACK bars.  CRITICAL = hard NO_BIAS gate.

    All inputs are already filtered to < pred_date by load_daily().
    """
    closes = daily_df["close"].values.astype(float)
    if len(closes) < ENTROPY_MIN_BARS + ENTROPY_WINDOW:
        return {"H_returns": None, "H_threshold": None, "entropy_state": "UNKNOWN"}

    log_rets = np.log(closes[1:] / closes[:-1])

    # H_returns: entropy of the last ENTROPY_WINDOW returns
    window = log_rets[-ENTROPY_WINDOW:]
    counts, _ = np.histogram(window, bins=ENTROPY_BINS)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    H_now = float(-np.sum(probs * np.log2(probs)))

    # Dynamic threshold: 75th-pctile of rolling-window entropy over lookback
    # LEAKAGE GUARD: only historical bars; pred_date bar excluded upstream
    lookback_rets = log_rets[-(ENTROPY_LOOKBACK + ENTROPY_WINDOW):-ENTROPY_WINDOW]
    if len(lookback_rets) < ENTROPY_WINDOW:
        H_thresh = H_now  # can't compute; treat as STABLE
    else:
        rolling_H = []
        for i in range(ENTROPY_WINDOW, len(lookback_rets) + 1):
            w = lookback_rets[i - ENTROPY_WINDOW:i]
            c, _ = np.histogram(w, bins=ENTROPY_BINS)
            p = c / c.sum()
            p = p[p > 0]
            rolling_H.append(-np.sum(p * np.log2(p)))
        H_thresh = float(np.percentile(rolling_H, ENTROPY_PCTILE))

    state = "CRITICAL" if H_now > H_thresh else "STABLE"
    return {
        "H_returns": round(H_now, 4),
        "H_threshold": round(H_thresh, 4),
        "entropy_state": state,
    }


# ============================================================================
# SECTION 3: PCA PRICE STRUCTURE
# ============================================================================

def build_pca_feature_matrix(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    8 features, all backward-looking.  Returns a DataFrame indexed by date.
    No future data can enter: all computations are rolling on prior closes.
    """
    df = daily_df.copy()
    df["oc_ret"]     = (df["close"] - df["open"]) / df["open"]
    df["hl_range"]   = (df["high"]  - df["low"])  / df["close"]
    df["mom_5d"]     = df["close"].pct_change(5)
    df["mom_10d"]    = df["close"].pct_change(10)
    df["mom_20d"]    = df["close"].pct_change(20)
    log_ret          = np.log(df["close"] / df["close"].shift(1))
    df["rvol_5d"]    = log_ret.rolling(5).std()
    df["rvol_10d"]   = log_ret.rolling(10).std()
    df["rvol_20d"]   = log_ret.rolling(20).std()

    feat_cols = ["oc_ret", "hl_range", "mom_5d", "mom_10d", "mom_20d",
                 "rvol_5d", "rvol_10d", "rvol_20d"]
    feat = df[["date"] + feat_cols].dropna().set_index("date")
    return feat


def load_or_fit_pca(feat_df: pd.DataFrame, pred_date: datetime) -> dict:
    """
    Load existing PCA state from pickle or fit a new one.
    Refit if models/pca_intraday.pkl is missing or older than REFIT_INTERVAL days.
    LEAKAGE GUARD: fit uses only rows strictly before pred_date.
    """
    pred_ts = pd.Timestamp(pred_date.date())

    if os.path.exists(PCA_PATH):
        with open(PCA_PATH, "rb") as f:
            state = pickle.load(f)
        fit_date = pd.Timestamp(state["fit_date"])
        days_since = (pred_ts - fit_date).days
        # LEAKAGE GUARD: assert fit used only historical data
        assert fit_date < pred_ts, (
            f"LEAKAGE: PCA fit_date {fit_date.date()} >= pred_date {pred_ts.date()}"
        )
        if days_since < REFIT_INTERVAL:
            return state
        log.info("PCA refit triggered (%d days since last fit)", days_since)

    # Fit on all available rows strictly before pred_date
    train = feat_df[feat_df.index < pred_ts]
    if len(train) < 30:
        raise ValueError(f"Too few bars to fit PCA: {len(train)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(train.values)
    pca = SklearnPCA(n_components=PCA_COMPONENTS, random_state=42)
    pca.fit(X)

    # fit_date = last training bar (strictly before pred_date)
    fit_date = train.index[-1]
    assert fit_date < pred_ts, (
        f"LEAKAGE: PCA fit_date {fit_date.date()} >= pred_date {pred_ts.date()}"
    )

    state = {
        "pca": pca,
        "scaler": scaler,
        "columns": list(train.columns),
        "fit_date": str(fit_date.date()),
        "n_train": len(train),
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(PCA_PATH, "wb") as f:
        pickle.dump(state, f)
    log.info("PCA fitted on %d bars (fit_date=%s)", len(train), fit_date.date())
    return state


def compute_pca_scores(feat_df: pd.DataFrame, pred_date: datetime, pca_state: dict) -> dict:
    """
    Project the most recent feature row through the frozen PCA.
    The feature row must be strictly before pred_date.
    """
    pred_ts = pd.Timestamp(pred_date.date())
    fit_date = pd.Timestamp(pca_state["fit_date"])

    # LEAKAGE GUARD
    assert fit_date < pred_ts, (
        f"LEAKAGE: pca_state fit_date {fit_date.date()} >= pred_date {pred_ts.date()}"
    )

    hist = feat_df[feat_df.index < pred_ts]
    if hist.empty:
        return {"PC1": None, "PC2": None, "PC3": None,
                "pca_fit_date": str(fit_date.date()), "pca_feature_date": None}

    last_row = hist.iloc[[-1]][pca_state["columns"]]
    feature_date = hist.index[-1]

    assert feature_date < pred_ts, (
        f"LEAKAGE: PCA feature_date {feature_date.date()} >= pred_date {pred_ts.date()}"
    )

    X_scaled = pca_state["scaler"].transform(last_row.values)
    scores   = pca_state["pca"].transform(X_scaled)[0]

    return {
        "PC1": round(float(scores[0]), 4),
        "PC2": round(float(scores[1]), 4),
        "PC3": round(float(scores[2]), 4) if len(scores) > 2 else None,
        "pca_fit_date":     str(fit_date.date()),
        "pca_feature_date": str(feature_date.date()),
    }


# ============================================================================
# SECTION 4: OPTIONS FLOW FEATURES
# ============================================================================

def compute_H_GEX_norm(levels: list) -> float:
    """
    Shannon entropy of the |GEX| distribution across nearby levels,
    normalized by log2(N).  ~0 = single dominant wall; ~1 = dispersed GEX.
    High H_GEX_norm (>0.6) penalizes confidence by ×0.7.
    """
    if not levels:
        return 0.5  # neutral fallback
    gex_abs = np.array([abs(lv.get("net_gex", 0)) for lv in levels], dtype=float)
    total   = gex_abs.sum()
    if total == 0 or len(gex_abs) < 2:
        return 0.0
    probs = gex_abs / total
    probs = probs[probs > 0]
    H     = -np.sum(probs * np.log2(probs))
    H_max = np.log2(len(gex_abs))
    return round(float(H / H_max) if H_max > 0 else 0.0, 4)


def compute_gamma_regime(nq_price: float, gamma_flip: float | None) -> str:
    """
    POSITIVE: price > flip + NEAR_FLIP_BUFFER  (dealers dampen moves)
    NEGATIVE: price < flip - NEAR_FLIP_BUFFER  (dealers amplify moves)
    NEAR_FLIP: within buffer                    (transition zone)
    UNKNOWN:  no flip data
    """
    if gamma_flip is None or nq_price is None:
        return "UNKNOWN"
    diff = nq_price - gamma_flip
    if diff > NEAR_FLIP_BUFFER:
        return "POSITIVE"
    if diff < -NEAR_FLIP_BUFFER:
        return "NEGATIVE"
    return "NEAR_FLIP"


def compute_top_wall(levels: list, nq_price: float) -> dict | None:
    """
    Identify the most significant nearby wall using a proximity-weighted score.
    proximity_weight = exp(-|dist_nq| / PROXIMITY_HALFLIFE)
    weighted_score = score * proximity_weight
    Returns the level with highest weighted_score.
    """
    if not levels or nq_price is None:
        return None
    best = None
    best_wscore = -1.0
    for lv in levels:
        dist    = abs(lv.get("dist_nq", 9999))
        score   = lv.get("score", 0.0)
        weight  = np.exp(-dist / PROXIMITY_HALFLIFE)
        wscore  = score * weight
        if wscore > best_wscore:
            best_wscore = wscore
            best = {**lv, "proximity_score": round(wscore, 2)}
    return best


# ============================================================================
# SECTION 5: MM INTENSIFICATION
# ============================================================================

def compute_mm_intensification(today: datetime) -> list:
    """
    Compare two most recent snapshots in logs/levels_YYYY-MM-DD.csv.
    intensification_score = |delta_gex| / (|gex_snap1| + 1e-9)
    Returns top 5 strikes sorted by intensification_score.
    """
    date_str = today.strftime("%Y-%m-%d")
    log_path = os.path.join(LOGS_DIR, f"levels_{date_str}.csv")
    if not os.path.exists(log_path):
        return []

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        log.warning("Could not read log file: %s", e)
        return []

    required = {"snapshot_time", "strike_futures", "net_gex"}
    if not required.issubset(df.columns):
        log.warning("Log file missing columns: %s", required - set(df.columns))
        return []

    snapshots = sorted(df["snapshot_time"].unique())
    if len(snapshots) < 2:
        return []

    snap1_data = df[df["snapshot_time"] == snapshots[-2]].set_index("strike_futures")
    snap2_data = df[df["snapshot_time"] == snapshots[-1]].set_index("strike_futures")

    common = snap1_data.index.intersection(snap2_data.index)
    if common.empty:
        return []

    results = []
    for strike in common:
        gex1 = float(snap1_data.loc[strike, "net_gex"])
        gex2 = float(snap2_data.loc[strike, "net_gex"])
        delta = gex2 - gex1
        score = abs(delta) / (abs(gex1) + 1e-9)
        if score < 0.05:  # ignore noise
            continue
        direction = "BUILDING_POSITIVE" if delta > 0 else "BUILDING_NEGATIVE"
        results.append({
            "strike_futures":       round(float(strike), 1),
            "gex_snap1":            round(gex1),
            "gex_snap2":            round(gex2),
            "gex_delta":            round(delta),
            "intensification_score": round(score, 3),
            "direction":            direction,
        })

    results.sort(key=lambda x: x["intensification_score"], reverse=True)
    return results[:5]


# ============================================================================
# SECTION 6: LEAKAGE AUDIT
# ============================================================================

def leakage_audit(feature_dates: dict, pred_date: datetime) -> None:
    """
    Assert that every named feature date is strictly before pred_date.
    Raises ValueError if any violation is found.
    """
    pred_ts = pd.Timestamp(pred_date.date())
    violations = []
    for name, date_val in feature_dates.items():
        if date_val is None:
            continue
        ts = pd.Timestamp(date_val)
        if ts >= pred_ts:
            violations.append(f"{name}: {ts.date()} >= pred {pred_ts.date()}")
    if violations:
        raise ValueError("LEAKAGE DETECTED:\n" + "\n".join(violations))
    log.info("Leakage audit PASSED (pred_date=%s)", pred_ts.date())


# ============================================================================
# SECTION 7: INTRADAY BIAS CLASSIFIER
# ============================================================================

def classify_intraday_bias(
    entropy:       dict,
    pca:           dict,
    gamma_regime:  str,
    gamma_flip:    float | None,
    nq_price:      float | None,
    top_wall:      dict | None,
    H_GEX_norm:    float,
    mm_intense:    list,
    macro_bias:    str,
    levels_regime: str,
) -> dict:
    """
    Core classifier.  Returns intraday_bias, confidence, air_pocket_watch,
    air_pocket_type, and reason.

    Regime priority:
    1. CRITICAL entropy → hard NO_BIAS gate
    2. POSITIVE gamma regime
    3. NEGATIVE gamma regime
    4. NEAR_FLIP gamma regime
    5. UNKNOWN → NEUTRAL fallback
    """

    # ── HARD GATE ────────────────────────────────────────────────────────────
    if entropy.get("entropy_state") == "CRITICAL":
        return {
            "intraday_bias":   "NO_BIAS",
            "confidence":      "AVOID",
            "air_pocket_watch": False,
            "air_pocket_type": None,
            "reason": (
                f"CRITICAL entropy (H={entropy.get('H_returns'):.3f} > "
                f"threshold {entropy.get('H_threshold'):.3f}): market is "
                "chaotic, no directional edge."
            ),
        }

    top_score   = top_wall.get("score", 0.0)   if top_wall else 0.0
    top_type    = top_wall.get("type", "")     if top_wall else ""
    top_dist    = top_wall.get("dist_nq", 999) if top_wall else 999
    top_strike  = top_wall.get("strike_futures") if top_wall else None

    macro_bull = macro_bias in MACRO_BULL
    macro_bear = macro_bias in MACRO_BEAR

    # MM flow: any BUILDING_NEGATIVE in top intensifiers?
    mm_neg = any(x["direction"] == "BUILDING_NEGATIVE" for x in mm_intense)
    mm_pos = any(x["direction"] == "BUILDING_POSITIVE" for x in mm_intense)

    # PC1 sign: positive = upward price momentum structure
    pc1_bull = (pca.get("PC1") or 0.0) > 0

    air_pocket_watch = False
    air_pocket_type  = None

    # ── POSITIVE GAMMA ────────────────────────────────────────────────────────
    if gamma_regime == "POSITIVE":
        # Check for FLIP_CROSS: near flip, strong wall near flip, BUILDING_NEGATIVE
        flip_close = (nq_price is not None and gamma_flip is not None and
                      abs(nq_price - gamma_flip) < NEAR_FLIP_BUFFER + 30)
        wall_near_flip = (top_wall is not None and gamma_flip is not None and
                          abs((top_strike or 9999) - gamma_flip) < AIR_POCKET_PROXIMITY)

        if flip_close and wall_near_flip and mm_neg:
            air_pocket_watch = True
            air_pocket_type  = "FLIP_CROSS"
            bias, conf = "BEARISH REVERSAL WATCH", "MODERATE"
            reason = (
                "FLIP_CROSS: price approaching gamma flip from positive side, "
                "strong wall near flip, BUILDING_NEGATIVE MM flow detected."
            )
        elif top_score >= STRONG_WALL:
            if macro_bull or pc1_bull:
                bias, conf = "BULLISH", "HIGH"
                reason = (
                    f"Positive gamma regime ({gamma_regime}), strong {top_type} "
                    f"(score={top_score:.0f}), macro confirms bull."
                )
            else:
                bias, conf = "NEUTRAL_BULLISH", "MODERATE"
                reason = (
                    f"Positive gamma regime, strong {top_type} "
                    f"(score={top_score:.0f}), macro not confirming."
                )
        else:
            bias = "NEUTRAL"
            conf = "LOW"
            reason = (
                f"Positive gamma regime but no strong wall "
                f"(top score={top_score:.0f}). Range-bound likely."
            )

    # ── NEGATIVE GAMMA ────────────────────────────────────────────────────────
    elif gamma_regime == "NEGATIVE":
        if top_score >= EXCEPTIONAL_WALL:
            air_pocket_watch = True
            air_pocket_type  = "EXCEPTIONAL_PUT_WALL"
            bias, conf = "BEARISH CONTINUATION", "MODERATE"
            reason = (
                f"Negative gamma regime, exceptional {top_type} "
                f"(score={top_score:.0f}): dealers amplify moves, "
                "put wall may offer temporary support or acceleration."
            )
        else:
            bias = "BEARISH CONTINUATION"
            conf = "MODERATE"
            reason = (
                f"Negative gamma regime (price below flip by >{NEAR_FLIP_BUFFER}pts): "
                "dealers amplify directional moves. "
            )
            if mm_neg:
                air_pocket_watch = True
                air_pocket_type  = "MM_NEGATIVE_BUILD"
                reason += "BUILDING_NEGATIVE MM flow detected — air pocket risk elevated."

        if macro_bull and not air_pocket_watch:
            conf = "LOW"
            reason += " Note: macro bias is bullish, creating conflicting signal."

    # ── NEAR FLIP ─────────────────────────────────────────────────────────────
    elif gamma_regime == "NEAR_FLIP":
        air_pocket_watch = True
        if top_score >= STRONG_WALL:
            bias = "REVERSAL WATCH"
            conf = "MODERATE"
            if "CALL" in top_type.upper():
                bias = "BEARISH REVERSAL"
            elif "PUT" in top_type.upper():
                bias = "BULLISH REVERSAL"
            reason = (
                f"NEAR_FLIP: price within {NEAR_FLIP_BUFFER}pts of gamma flip. "
                f"Strong {top_type} (score={top_score:.0f}) nearby. "
                "Flip crossing could accelerate move."
            )
        else:
            bias = "NEUTRAL"
            conf = "LOW"
            reason = (
                f"NEAR_FLIP: price within {NEAR_FLIP_BUFFER}pts of gamma flip. "
                "No strong wall anchoring direction. High uncertainty."
            )
        air_pocket_type = "FLIP_CROSS"

    # ── UNKNOWN ───────────────────────────────────────────────────────────────
    else:
        bias, conf = "NEUTRAL", "LOW"
        reason = "Gamma regime unknown (no flip data). Cannot classify."

    # ── CONFIDENCE MODIFIERS ──────────────────────────────────────────────────
    CONF_ORDER = ["AVOID", "LOW", "MODERATE", "HIGH"]

    def downgrade(c, reason_suffix):
        idx = CONF_ORDER.index(c)
        return CONF_ORDER[max(0, idx - 1)], reason_suffix

    if H_GEX_norm > H_GEX_CONFIDENCE_CUT:
        conf, sfx = downgrade(conf, f" H_GEX_norm={H_GEX_norm:.2f} >0.6: GEX dispersed, confidence penalized.")
        reason += sfx

    if macro_bias not in MACRO_BULL and macro_bias not in MACRO_BEAR:
        conf, sfx = downgrade(conf, f" Macro neutral ({macro_bias}): confidence penalized.")
        reason += sfx

    return {
        "intraday_bias":    bias,
        "confidence":       conf,
        "air_pocket_watch": air_pocket_watch,
        "air_pocket_type":  air_pocket_type,
        "reason":           reason,
    }


# ============================================================================
# SECTION 8: MAIN
# ============================================================================

def main(pred_date_override: str | None = None, dry_run: bool = False) -> None:
    now_et = datetime.now(ET)
    if pred_date_override:
        pred_date = ET.localize(datetime.strptime(pred_date_override, "%Y-%m-%d"))
    else:
        pred_date = now_et

    log.info("Intraday bias run — pred_date=%s  dry_run=%s", pred_date.date(), dry_run)

    # Load inputs
    daily_df   = load_daily(pred_date)
    macro_data = load_macro_bias()
    levels_raw = load_levels(dry_run)

    macro_bias    = macro_data.get("confluence", "UNKNOWN")
    macro_regime  = macro_data.get("macro_regime", {})

    nq_price   = levels_raw.get("nq_price")   if levels_raw else None
    gamma_flip = levels_raw.get("gamma_flip") if levels_raw else None
    levels     = levels_raw.get("levels", []) if levels_raw else []
    lvl_regime = levels_raw.get("regime", "UNKNOWN") if levels_raw else "UNKNOWN"

    # Entropy
    entropy = compute_return_entropy(daily_df, pred_date)
    log.info("Entropy: H=%.4f  threshold=%.4f  state=%s",
             entropy.get("H_returns", 0),
             entropy.get("H_threshold", 0),
             entropy.get("entropy_state"))

    # PCA
    feat_df   = build_pca_feature_matrix(daily_df)
    pca_state = load_or_fit_pca(feat_df, pred_date)
    pca_scores = compute_pca_scores(feat_df, pred_date, pca_state)
    log.info("PCA: PC1=%.3f  PC2=%.3f  PC3=%s",
             pca_scores.get("PC1") or 0,
             pca_scores.get("PC2") or 0,
             pca_scores.get("PC3"))

    # Options flow
    H_GEX_norm    = compute_H_GEX_norm(levels)
    gamma_regime  = compute_gamma_regime(nq_price, gamma_flip)
    top_wall      = compute_top_wall(levels, nq_price)
    mm_intense    = compute_mm_intensification(pred_date)

    log.info("Gamma regime=%s  H_GEX_norm=%.3f  top_wall_score=%s",
             gamma_regime, H_GEX_norm,
             top_wall.get("score") if top_wall else "N/A")

    # Leakage audit
    feature_dates = {
        "pca_fit_date":     pca_scores.get("pca_fit_date"),
        "pca_feature_date": pca_scores.get("pca_feature_date"),
        "entropy_last_bar": str(daily_df["date"].iloc[-1].date()) if not daily_df.empty else None,
    }
    leakage_audit(feature_dates, pred_date)

    # Classify
    result = classify_intraday_bias(
        entropy      = entropy,
        pca          = pca_scores,
        gamma_regime = gamma_regime,
        gamma_flip   = gamma_flip,
        nq_price     = nq_price,
        top_wall     = top_wall,
        H_GEX_norm   = H_GEX_norm,
        mm_intense   = mm_intense,
        macro_bias   = macro_bias,
        levels_regime = lvl_regime,
    )

    updated_et = now_et.strftime("%I:%M %p ET")

    output = {
        "updated":           updated_et,
        "pred_date":         str(pred_date.date()),
        # -- Entropy --
        "entropy_state":     entropy.get("entropy_state"),
        "H_returns":         entropy.get("H_returns"),
        "H_threshold":       entropy.get("H_threshold"),
        # -- PCA --
        "PC1":               pca_scores.get("PC1"),
        "PC2":               pca_scores.get("PC2"),
        "PC3":               pca_scores.get("PC3"),
        "pca_fit_date":      pca_scores.get("pca_fit_date"),
        # -- Options flow --
        "nq_price":          nq_price,
        "gamma_flip":        gamma_flip,
        "gamma_regime":      gamma_regime,
        "H_GEX_norm":        H_GEX_norm,
        "levels_regime":     lvl_regime,
        "top_wall":          top_wall,
        "mm_intensification": mm_intense,
        # -- Macro context --
        "macro_bias":        macro_bias,
        "macro_regime":      macro_regime,
        # -- Classifier output --
        "intraday_bias":     result["intraday_bias"],
        "confidence":        result["confidence"],
        "air_pocket_watch":  result["air_pocket_watch"],
        "air_pocket_type":   result["air_pocket_type"],
        "reason":            result["reason"],
    }

    if dry_run:
        print(json.dumps(output, indent=2))
        return

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Written: %s", OUT_PATH)
    log.info("Intraday bias: %s | %s | air_pocket=%s",
             result["intraday_bias"], result["confidence"], result["air_pocket_watch"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VANTA intraday bias classifier")
    parser.add_argument("--date",    metavar="YYYY-MM-DD", help="Override prediction date")
    parser.add_argument("--dry-run", action="store_true",  help="Print JSON, do not write file")
    args = parser.parse_args()
    main(pred_date_override=args.date, dry_run=args.dry_run)
