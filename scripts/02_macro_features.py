import os
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# ── CONFIG ────────────────────────────────────────────────────
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(BASE_DIR, "data", "processed")

# ── FRED SERIES ───────────────────────────────────────────────
# WALCL     = Fed total assets (millions)
# RRPONTSYD = Overnight RRP (billions)
# WTREGEN   = Treasury General Account (millions)
# DGS10     = 10Y yield (%)
# Note: HY OAS dropped — FRED restricted to 3 years.
#       Using VIX from local CSV instead (same risk signal)

FRED_SERIES = {
    "walcl":  "WALCL",
    "rrp":    "RRPONTSYD",
    "tga":    "WTREGEN",
    "us10y":  "DGS10",
    "us2y":   "DGS2",
}

# ── FRED FETCH ────────────────────────────────────────────────
def fred_fetch(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(url, params={
        "series_id":         series_id,
        "api_key":           FRED_KEY,
        "file_type":         "json",
        "observation_start": "2014-01-01",
        "observation_end":   "2026-01-31",
    })
    r.raise_for_status()
    df = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"]  = pd.to_datetime(df["date"])
    df = df.dropna().set_index("date")
    return df["value"]

# ── VIX FROM LOCAL CSV ────────────────────────────────────────
# We use VIX as the credit stress / risk-off proxy
# since FRED restricted HY OAS to 3 years of history.
# VIX rising = fear/risk-off = bearish for equities
# VIX falling = complacency/risk-on = bullish for equities
def fetch_vix_from_csv():
    vix_path = os.path.join(BASE_DIR, "data", "raw", "VIX", "15Min_Vix.csv")
    df = pd.read_csv(vix_path, sep=";", skiprows=1,
                     thousands=".", decimal=",", low_memory=False)
    df.columns = ["date", "symbol", "open", "high", "low", "close", "volume"]
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).set_index("date")
    df = df[(df.index >= "2014-01-01") & (df.index <= "2026-01-31")]
    return df["close"]

# ── PULL ALL DATA ─────────────────────────────────────────────
print("Pulling FRED data...")
raw = {}
for name, sid in FRED_SERIES.items():
    raw[name] = fred_fetch(sid)
    print(f"  {name}: {len(raw[name])} obs  "
          f"{raw[name].index.min().date()} → {raw[name].index.max().date()}")

print("\nLoading VIX from local CSV...")
vix_raw = fetch_vix_from_csv()
print(f"  vix: {len(vix_raw)} obs  "
      f"{vix_raw.index.min().date()} → {vix_raw.index.max().date()}")

print("\nPulling DXY, VIX3M, VVIX from yfinance...")
dxy = yf.download("DX-Y.NYB", start="2014-01-01", end="2026-01-31",
                  interval="1d", progress=False)["Close"]
dxy.index = pd.to_datetime(dxy.index)
dxy.name  = "dxy"
print(f"  dxy: {len(dxy)} obs  "
      f"{dxy.index.min().date()} → {dxy.index.max().date()}")

vix3m_raw = yf.download("^VIX3M", start="2014-01-01", end="2026-01-31",
                        interval="1d", progress=False)["Close"].squeeze()
vix3m_raw.index = pd.to_datetime(vix3m_raw.index)
vvix_raw  = yf.download("^VVIX",  start="2014-01-01", end="2026-01-31",
                        interval="1d", progress=False)["Close"].squeeze()
vvix_raw.index = pd.to_datetime(vvix_raw.index)
print(f"  vix3m: {len(vix3m_raw)} obs  "
      f"{vix3m_raw.index.min().date()} → {vix3m_raw.index.max().date()}")
print(f"  vvix:  {len(vvix_raw)} obs  "
      f"{vvix_raw.index.min().date()} → {vvix_raw.index.max().date()}")

# ── BUILD WEEKLY TABLE ────────────────────────────────────────
print("\nBuilding weekly table...")

# Unit conversions
walcl = raw["walcl"] / 1000   # millions → billions
rrp   = raw["rrp"]             # already billions
tga   = raw["tga"]   / 1000   # millions → billions

def to_weekly_last(series):
    return series.resample("W-THU").last()

def to_weekly_mean(series):
    return series.resample("W-THU").mean()

w_walcl = to_weekly_last(walcl)
w_tga   = to_weekly_last(tga)
w_rrp   = to_weekly_mean(rrp)
w_10y   = to_weekly_mean(raw["us10y"])
w_2y    = to_weekly_mean(raw["us2y"])
w_dxy   = to_weekly_mean(dxy.squeeze())
w_vix   = to_weekly_mean(vix_raw)
w_vix3m = to_weekly_mean(vix3m_raw)
w_vvix  = to_weekly_mean(vvix_raw)

# Net liquidity = Fed assets - RRP - TGA
net_liq = w_walcl - w_rrp - w_tga

# ── COMBINE ───────────────────────────────────────────────────
macro = pd.DataFrame({
    "walcl":   w_walcl,
    "rrp":     w_rrp,
    "tga":     w_tga,
    "net_liq": net_liq,
    "vix":     w_vix,
    "vix3m":   w_vix3m,
    "vvix":    w_vvix,
    "us10y":   w_10y,
    "us2y":    w_2y,
    "dxy":     w_dxy,
}).ffill(limit=1).dropna()

# Yield curve: 10Y - 2Y (2Y10Y spread)
macro["yield_curve"] = macro["us10y"] - macro["us2y"]

# VIX term structure: ratio > 1.0 = backwardation (stress), < 1.0 = contango (calm)
macro["vix_ratio"] = macro["vix"] / macro["vix3m"]

# ── COMPUTE CHANGES ───────────────────────────────────────────
# WoW = week over week, 4W = 4 week change
# These are the actual model inputs
for col in ["net_liq", "vix", "us10y", "dxy", "yield_curve", "vix_ratio"]:
    macro[f"{col}_wow"] = macro[col].diff(1)
    macro[f"{col}_4w"]  = macro[col].diff(4)

# ── LEAKAGE PROTECTION ────────────────────────────────────────
# Shift all features forward 1 week so the model only sees
# data that was published BEFORE the week it's predicting
feature_cols = [c for c in macro.columns
                if c not in ["walcl", "rrp", "tga", "us2y"]]
macro[feature_cols] = macro[feature_cols].shift(1)

macro = macro.dropna()

# ── SAVE ──────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "macro_weekly.csv")
macro.to_csv(out_path)
print(f"\nMacro table saved → {out_path}")
print(f"Shape: {macro.shape}  "
      f"({macro.index.min().date()} → {macro.index.max().date()})")
print(f"\nExpected ~620 weeks. Got {len(macro)}.")
print("\nSample (last 5 weeks):")
print(macro[["net_liq", "net_liq_wow",
             "vix", "vix3m", "vix_ratio", "vix_ratio_wow",
             "vvix", "yield_curve", "yield_curve_wow"]].tail())