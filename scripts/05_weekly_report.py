import io
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
import requests
import pickle
from datetime import datetime
from dotenv import load_dotenv

# ── CONFIG ────────────────────────────────────────────────────
load_dotenv()
FRED_KEY  = os.getenv("FRED_API_KEY")
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

LIVE_MODE = "--live" in sys.argv

# ── LOAD SAVED MODELS ─────────────────────────────────────────
print("Loading models...")
with open(os.path.join(MODEL_DIR, "price_model.pkl"), "rb") as f:
    price_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "vol_model.pkl"), "rb") as f:
    vol_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "price_features.pkl"), "rb") as f:
    PRICE_FEATURES = pickle.load(f)
with open(os.path.join(MODEL_DIR, "all_features.pkl"), "rb") as f:
    ALL_FEATURES = pickle.load(f)

# ── DATE RANGES ───────────────────────────────────────────────
today     = datetime.today().strftime("%Y-%m-%d")
if LIVE_MODE:
    price_start = "2025-09-01"   # enough lookback for 12w momentum
    macro_start = "2025-09-01"
    data_start  = "2026-01-01"   # only show recent weeks
else:
    price_start = "2025-12-01"
    macro_start = "2025-10-01"
    data_start  = "2026-02-01"

# ── PULL PRICE DATA ───────────────────────────────────────────
print("\nPulling fresh price data...")
nq = yf.download("NQ=F", start=price_start, end=today,
                 interval="1d", progress=False)
es = yf.download("ES=F", start=price_start, end=today,
                 interval="1d", progress=False)

for df in [nq, es]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
nq.columns = [c.lower() for c in nq.columns]
es.columns = [c.lower() for c in es.columns]

print(f"  NQ: {len(nq)} bars  {nq.index.min().date()} → {nq.index.max().date()}")

# ── PULL MACRO DATA ───────────────────────────────────────────
print("\nPulling macro data...")

def fred_fetch(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(url, params={
        "series_id":         series_id,
        "api_key":           FRED_KEY,
        "file_type":         "json",
        "observation_start": macro_start,
        "observation_end":   today,
    })
    r.raise_for_status()
    df = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"]  = pd.to_datetime(df["date"])
    return df.dropna().set_index("date")["value"]

walcl = fred_fetch("WALCL")   / 1000
rrp   = fred_fetch("RRPONTSYD")
tga   = fred_fetch("WTREGEN") / 1000
us10y = fred_fetch("DGS10")

dxy = yf.download("DX-Y.NYB", start=macro_start, end=today,
                  interval="1d", progress=False)["Close"].squeeze()
vix = yf.download("^VIX",     start=macro_start, end=today,
                  interval="1d", progress=False)["Close"].squeeze()

# ── BUILD WEEKLY FEATURES ─────────────────────────────────────
print("\nBuilding features...")

def to_weekly_last(s): return s.resample("W-THU").last()
def to_weekly_mean(s): return s.resample("W-THU").mean()

net_liq = to_weekly_last(walcl) - to_weekly_mean(rrp) - to_weekly_last(tga)

macro = pd.DataFrame({
    "net_liq": net_liq,
    "vix":     to_weekly_mean(vix),
    "us10y":   to_weekly_mean(us10y),
    "dxy":     to_weekly_mean(dxy),
    "walcl":   to_weekly_last(walcl),
    "rrp":     to_weekly_mean(rrp),
    "tga":     to_weekly_last(tga),
}).ffill(limit=1).dropna()

for col in ["net_liq", "vix", "us10y", "dxy"]:
    macro[f"{col}_wow"] = macro[col].diff(1)
    macro[f"{col}_4w"]  = macro[col].diff(4)

feature_cols = [c for c in macro.columns if c not in ["walcl","rrp","tga"]]
macro[feature_cols] = macro[feature_cols].shift(1)
macro = macro.dropna()

# ── PULL COT DATA (current week) ──────────────────────────────
print("\nPulling current-week COT data...")

_COT_URL     = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"
_COT_USECOLS = [3, 7, 11, 12, 14, 15, 31, 32]
_COT_COLMAP  = {
    3:  "contract_code",
    7:  "open_interest",
    11: "asset_mgr_longs",
    12: "asset_mgr_shorts",
    14: "lev_longs",
    15: "lev_shorts",
    31: "change_lev_longs",
    32: "change_lev_shorts",
}
_COT_NUM_COLS = [v for k, v in _COT_COLMAP.items() if k != 3]

cot_feats = {
    "nq_lev_wow":           0.0,
    "es_lev_wow":           0.0,
    "nq_lev_pctile":        0.0,
    "es_asset_mgr_net_pct": 0.0,
}

try:
    r = requests.get(_COT_URL, timeout=30)
    r.raise_for_status()
    cot_raw = pd.read_csv(
        io.BytesIO(r.content), header=None,
        usecols=_COT_USECOLS, dtype=str, low_memory=False,
    )
    cot_raw.rename(columns=_COT_COLMAP, inplace=True)
    cot_raw["contract_code"] = cot_raw["contract_code"].str.strip()
    for col in _COT_NUM_COLS:
        cot_raw[col] = pd.to_numeric(
            cot_raw[col].str.replace(",", ""), errors="coerce"
        )

    # NQ — contract code 209742
    nq_cot = cot_raw[cot_raw["contract_code"] == "209742"]
    if not nq_cot.empty:
        row = nq_cot.iloc[0]
        oi  = row["open_interest"]
        if pd.notna(oi) and oi > 0:
            cot_feats["nq_lev_wow"] = (
                row["change_lev_longs"] - row["change_lev_shorts"]
            ) / oi

    # ES — contract code 13874A
    es_cot = cot_raw[cot_raw["contract_code"] == "13874A"]
    if not es_cot.empty:
        row = es_cot.iloc[0]
        oi  = row["open_interest"]
        if pd.notna(oi) and oi > 0:
            cot_feats["es_lev_wow"] = (
                row["change_lev_longs"] - row["change_lev_shorts"]
            ) / oi
            cot_feats["es_asset_mgr_net_pct"] = (
                row["asset_mgr_longs"] - row["asset_mgr_shorts"]
            ) / oi

    print(f"  nq_lev_wow={cot_feats['nq_lev_wow']:.4f}  "
          f"es_lev_wow={cot_feats['es_lev_wow']:.4f}  "
          f"es_asset_mgr_net_pct={cot_feats['es_asset_mgr_net_pct']:.4f}")
except Exception as e:
    print(f"  COT fetch failed ({e}) — using 0.0 fallbacks")

# nq_lev_pctile — most recent positioning_pctile from saved history
_cot_as_of_date = None
try:
    cot_nq_hist = pd.read_csv(
        os.path.join(PROC_DIR, "cot_NQ.csv"), index_col=0, parse_dates=True
    )
    valid_pctile = cot_nq_hist["positioning_pctile"].dropna()
    if not valid_pctile.empty:
        cot_feats["nq_lev_pctile"] = float(valid_pctile.iloc[-1])
        _cot_as_of_date = valid_pctile.index[-1].date()
        print(f"  nq_lev_pctile={cot_feats['nq_lev_pctile']:.4f} "
              f"(from cot_NQ.csv {_cot_as_of_date})")
except Exception as e:
    print(f"  cot_NQ.csv load failed ({e}) — nq_lev_pctile=0.0")

# Price features
nq_w = nq["close"].resample("W-THU").agg(
    weekly_open="first", weekly_high="max",
    weekly_low="min",    weekly_close="last"
)
es_w = es["close"].resample("W-THU").last()

daily_ret = nq["close"].pct_change()
nq_w["realized_vol"]  = daily_ret.resample("W-THU").std() * np.sqrt(5)
nq_w["nq_week_ret"]   = nq_w["weekly_close"].pct_change()
nq_w["nq_close_pos"]  = ((nq_w["weekly_close"] - nq_w["weekly_low"]) /
                          (nq_w["weekly_high"] - nq_w["weekly_low"] + 1e-9))
nq_w["nq_week_range"] = ((nq_w["weekly_high"] - nq_w["weekly_low"]) /
                           nq_w["weekly_close"])
nq_w["nq_mom_4w"]     = nq_w["weekly_close"].pct_change(4)
nq_w["nq_mom_12w"]    = nq_w["weekly_close"].pct_change(12)
nq_w["nq_vol_regime"] = (nq_w["realized_vol"] /
                          nq_w["realized_vol"].rolling(12, min_periods=1).mean())
nq_w["es_nq_spread"]  = nq_w["nq_week_ret"] - es_w.pct_change()
nq_1h = yf.download("NQ=F",
                    start=(datetime.today() - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
                    end=today, interval="1h", progress=False)
if isinstance(nq_1h.columns, pd.MultiIndex):
    nq_1h.columns = [c[0] for c in nq_1h.columns]
nq_1h.columns = [c.lower() for c in nq_1h.columns]
nq_1h["ret_1h"] = nq_1h["close"].pct_change()
nq_w["nq_1h_drift"] = nq_1h["ret_1h"].resample("W-THU").mean()
nq_w["nq_1h_vol"]   = nq_1h["ret_1h"].resample("W-THU").std()
nq_w[["nq_1h_drift", "nq_1h_vol"]] = nq_w[["nq_1h_drift", "nq_1h_vol"]].fillna(0.0)

# Key levels
nq_lv = nq.resample("W-THU").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last"})
nq_lv["pw_high"]  = nq_lv["high"].shift(1)
nq_lv["pw_low"]   = nq_lv["low"].shift(1)
nq_lv["pw_close"] = nq_lv["close"].shift(1)
nq_lv["pw_mid"]   = (nq_lv["pw_high"] + nq_lv["pw_low"]) / 2
nq_lv["4w_high"]  = nq_lv["high"].rolling(4).max().shift(1)
nq_lv["4w_low"]   = nq_lv["low"].rolling(4).min().shift(1)

nq["tr"] = np.maximum(nq["high"] - nq["low"],
           np.maximum(abs(nq["high"] - nq["close"].shift(1)),
                      abs(nq["low"]  - nq["close"].shift(1))))
nq_lv["atr_5d"]  = nq["tr"].resample("W-THU").mean()
nq_lv["atr_20d"] = nq["tr"].rolling(20).mean().resample("W-THU").last()

# Merge
price_feats = nq_w[[
    "nq_week_ret", "nq_close_pos", "nq_week_range",
    "nq_mom_4w", "nq_mom_12w", "nq_vol_regime",
    "es_nq_spread", "realized_vol", "nq_1h_drift", "nq_1h_vol",
]]
combined = macro.join(price_feats, how="inner")
combined = combined[combined.index >= data_start]
levels   = nq_lv[nq_lv.index >= data_start]

# ── JOIN COT FEATURES ─────────────────────────────────────────
# All weeks default to 0.0; latest week gets the live COT values.
for feat, val in cot_feats.items():
    combined[feat] = 0.0
if not combined.empty:
    combined.loc[combined.index[-1], list(cot_feats.keys())] = list(cot_feats.values())

# ── MACRO REGIME SCORING ──────────────────────────────────────
hist = pd.read_csv(os.path.join(PROC_DIR, "model_dataset_enriched.csv"),
                   index_col=0, parse_dates=True)
MACRO_COLS = ["net_liq_wow", "net_liq_4w", "vix_wow", "vix_4w",
              "us10y_wow", "us10y_4w", "dxy_wow", "dxy_4w"]
hist_vals = hist[MACRO_COLS].dropna()

def pct_rank(val, hist_series, as_of):
    sliced = hist_series[hist_series.index <= as_of]
    window = sliced.iloc[-156:] if len(sliced) > 156 else sliced
    if len(window) < 10:
        window = sliced
    return (window < val).mean()

def score_macro_live(row, as_of):
    p = {col: pct_rank(row[col], hist_vals[col], as_of)
         for col in MACRO_COLS if col in row and not pd.isna(row[col])}
    s = {
        "net_liq_wow": 1 if p.get("net_liq_wow",0.5)>0.65 else(-1 if p.get("net_liq_wow",0.5)<0.35 else 0),
        "net_liq_4w":  1 if p.get("net_liq_4w",0.5) >0.65 else(-1 if p.get("net_liq_4w",0.5) <0.35 else 0),
        "vix_wow":     1 if p.get("vix_wow",0.5)    <0.35 else(-1 if p.get("vix_wow",0.5)    >0.65 else 0),
        "vix_4w":      1 if p.get("vix_4w",0.5)     <0.35 else(-1 if p.get("vix_4w",0.5)     >0.65 else 0),
        "dxy_wow":     1 if p.get("dxy_wow",0.5)    <0.35 else(-1 if p.get("dxy_wow",0.5)    >0.65 else 0),
        "dxy_4w":      1 if p.get("dxy_4w",0.5)     <0.35 else(-1 if p.get("dxy_4w",0.5)     >0.65 else 0),
        "us10y_wow":   1 if p.get("us10y_wow",0.5)  <0.35 else(-1 if p.get("us10y_wow",0.5)  >0.70 else 0),
        "us10y_4w":    1 if p.get("us10y_4w",0.5)   <0.35 else(-1 if p.get("us10y_4w",0.5)   >0.70 else 0),
    }
    return sum(s.values()), s

def classify_regime(score):
    if score >= 3:    return "RISK-ON"
    elif score <= -3: return "RISK-OFF"
    elif score >= 1:  return "LEAN RISK-ON"
    elif score <= -1: return "LEAN RISK-OFF"
    else:             return "TRANSITION"

# ── REPORT FUNCTION ───────────────────────────────────────────
def print_week(date, row, lv, save_list=None):
    macro_score, macro_details = score_macro_live(row, as_of=date)
    regime     = classify_regime(macro_score)

    price_prob = price_model.predict_proba(
        pd.DataFrame([row[PRICE_FEATURES].values],
                     columns=PRICE_FEATURES))[:, 1][0]
    price_dir  = "BULLISH" if price_prob > 0.5 else "BEARISH"

    vol_input    = pd.DataFrame([row[ALL_FEATURES].values], columns=ALL_FEATURES)
    vol_forecast = vol_model.predict(vol_input)[0]

    confluence = {
        "RISK-ON":       "STRONG BULL",
        "LEAN RISK-ON":  "LEAN BULL",
        "TRANSITION":    "MIXED",
        "LEAN RISK-OFF": "LEAN BEAR",
        "RISK-OFF":      "STRONG BEAR",
    }[regime]

    bull_f = [k for k, v in macro_details.items() if v > 0]
    bear_f = [k for k, v in macro_details.items() if v < 0]

    week_start = (date - pd.Timedelta(days=6)).date()
    week_end   = date.date()

    print(f"\n  ┌─── {'CURRENT WEEK' if LIVE_MODE else 'Week of ' + str(date.date())} {'─'*33}")
    print(f"  │ Period:        {week_start} → {week_end}")
    print(f"  │ Generated:     {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"  │")
    print(f"  │ MACRO REGIME:  {regime}  (score: {macro_score:+.0f}/8)")
    if bull_f: print(f"  │   Bullish inputs: {', '.join(bull_f)}")
    if bear_f: print(f"  │   Bearish inputs: {', '.join(bear_f)}")
    print(f"  │")
    factor_str = "  ".join(f"{k}:{v:+d}" for k, v in macro_details.items())
    print(f"  │ MACRO FACTORS: {factor_str}")
    print(f"  │ VOL FORECAST:  est. {vol_forecast:.1%} weekly realized vol")
    print(f"  │")

    nq_pctile = row.get("nq_lev_pctile", 0.0)
    nq_wow    = row.get("nq_lev_wow", 0.0)
    es_am_net = row.get("es_asset_mgr_net_pct", 0.0)

    if nq_pctile < 0.20:
        lev_label = "EXTREME SHORT"
    elif nq_pctile > 0.80:
        lev_label = "EXTREME LONG"
    else:
        lev_label = "NEUTRAL"

    wow_note = "adding shorts" if nq_wow < 0 else "adding longs"
    am_dir   = "net long"  if es_am_net >= 0 else "net short"
    am_desc  = "institutions holding" if es_am_net >= 0 else "institutions reducing"

    if nq_pctile < 0.20 and nq_wow < 0:
        pos_interp = "EXTREME SHORT — adding shorts, squeeze risk elevated"
    elif nq_pctile < 0.20 and nq_wow > 0:
        pos_interp = "EXTREME SHORT — covering, potential squeeze underway"
    elif nq_pctile > 0.80 and nq_wow > 0:
        pos_interp = "EXTREME LONG — adding longs, reversal risk elevated"
    elif nq_pctile > 0.80 and nq_wow < 0:
        pos_interp = "EXTREME LONG — reducing, potential unwind underway"
    else:
        pos_interp = "NEUTRAL positioning"

    cot_as_of = _cot_as_of_date if _cot_as_of_date is not None else week_end
    print(f"  │ COT POSITIONING (as of {cot_as_of}):")
    print(f"  │   NQ Lev Funds:    {lev_label} ({nq_pctile*100:.1f}th pctile)")
    print(f"  │   NQ Lev WoW:      {nq_wow:+.1%} OI ({wow_note})")
    print(f"  │   ES Asset Mgr:    {es_am_net:+.1%} OI {am_dir} ({am_desc})")
    print(f"  │   ► Positioning:   {pos_interp}")
    print(f"  │")

    if lv is not None and not pd.isna(lv).all():
        print(f"  │ KEY LEVELS (NQ):")
        for name, key in [("Prior Week High",  "pw_high"),
                          ("Prior Week Low",   "pw_low"),
                          ("Prior Week Close", "pw_close"),
                          ("Prior Week Mid",   "pw_mid"),
                          ("4-Week High",      "4w_high"),
                          ("4-Week Low",       "4w_low"),
                          ("ATR 5-day",        "atr_5d"),
                          ("ATR 20-day",       "atr_20d")]:
            if key in lv.index and not pd.isna(lv[key]):
                print(f"  │   {name:<20}  {lv[key]:.2f}")
        print(f"  │")

    print(f"  │ ► CONFLUENCE:  {confluence}")
    print(f"  └{'─'*55}")

    if save_list is not None:
        save_list.append({
            "date": date, "regime": regime,
            "macro_score": macro_score, "price_dir": price_dir,
            "price_prob": price_prob, "vol_forecast": vol_forecast,
            "confluence": confluence,
        })

# ── RUN ───────────────────────────────────────────────────────
if LIVE_MODE:
    print("\n" + "="*60)
    print(f"  LIVE WEEKLY BIAS — {datetime.today().strftime('%Y-%m-%d')}")
    print("="*60)

    # Show only the most recent week
    latest_date = combined.index[-1]
    lv = levels.loc[latest_date] if latest_date in levels.index else None
    print_week(latest_date, combined.loc[latest_date], lv)

else:
    print("\n" + "="*60)
    print("  BLIND PREDICTIONS — FEB TO MAY 2026")
    print("  (actuals intentionally hidden)")
    print("="*60)

    predictions = []
    for date in combined.index:
        lv = levels.loc[date] if date in levels.index else None
        print_week(date, combined.loc[date], lv, save_list=predictions)

    pred_df   = pd.DataFrame(predictions).set_index("date")
    pred_path = os.path.join(PROC_DIR, "blind_predictions_feb_may_2026.csv")
    pred_df.to_csv(pred_path)
    print(f"\nPredictions saved → {pred_path}")
    print(f"Total weeks: {len(pred_df)}")
    print("\nRun 06_check_accuracy.py later to compare against actuals.")
