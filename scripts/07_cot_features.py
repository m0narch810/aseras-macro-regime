import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")

YEARS    = list(range(2014, 2026))
HIST_URL = "https://www.cftc.gov/files/dea/history/fut_fin_xls_{year}.zip"
WEEK_URL = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"

NQ_CODES  = {"209742", "20974+"}
ES_CODES  = {"13874A", "13874+"}
ALL_CODES = NQ_CODES | ES_CODES

# Column indices (0-based, header=None) — TFF report layout
_D,   _C   = 2,  3   # Report_Date_as_MM_DD_YYYY, CFTC_Contract_Market_Code
_OI        = 7       # Open_Interest_All
_DLR_L,  _DLR_S  =  8,  9   # Dealer long/short
_AM_L,   _AM_S   = 11, 12   # Asset Manager long/short
_LM_L,   _LM_S  = 14, 15   # Leveraged Money long/short
_CLM_L,  _CLM_S = 31, 32   # Change in Leveraged Money long/short

USECOLS = [_D, _C, _OI, _DLR_L, _DLR_S, _AM_L, _AM_S, _LM_L, _LM_S, _CLM_L, _CLM_S]
COLMAP  = {
    _D:      "date",
    _C:      "contract_code",
    _OI:     "open_interest",
    _DLR_L:  "dealer_longs",
    _DLR_S:  "dealer_shorts",
    _AM_L:   "asset_mgr_longs",
    _AM_S:   "asset_mgr_shorts",
    _LM_L:   "lev_longs",
    _LM_S:   "lev_shorts",
    _CLM_L:  "change_lev_longs",
    _CLM_S:  "change_lev_shorts",
}
NUM_COLS = [v for k, v in COLMAP.items() if k not in (_D, _C)]

# Excel column names (TFF report — historical zip files use underscore headers)
EXCEL_COLS = [
    "Market_and_Exchange_Names",
    "Report_Date_as_MM_DD_YYYY",
    "CFTC_Contract_Market_Code",
    "Open_Interest_All",
    "Dealer_Positions_Long_All",
    "Dealer_Positions_Short_All",
    "Asset_Mgr_Positions_Long_All",
    "Asset_Mgr_Positions_Short_All",
    "Lev_Money_Positions_Long_All",
    "Lev_Money_Positions_Short_All",
    "Change_in_Lev_Money_Long_All",
    "Change_in_Lev_Money_Short_All",
]

EXCEL_COLMAP = {
    "Report_Date_as_MM_DD_YYYY":      "date",
    "CFTC_Contract_Market_Code":      "contract_code",
    "Open_Interest_All":              "open_interest",
    "Dealer_Positions_Long_All":      "dealer_longs",
    "Dealer_Positions_Short_All":     "dealer_shorts",
    "Asset_Mgr_Positions_Long_All":   "asset_mgr_longs",
    "Asset_Mgr_Positions_Short_All":  "asset_mgr_shorts",
    "Lev_Money_Positions_Long_All":   "lev_longs",
    "Lev_Money_Positions_Short_All":  "lev_shorts",
    "Change_in_Lev_Money_Long_All":   "change_lev_longs",
    "Change_in_Lev_Money_Short_All":  "change_lev_shorts",
}

# ── DEBUG ─────────────────────────────────────────────────────
def debug_2024():
    url = HIST_URL.format(year=2024)
    print(f"Fetching: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
        print(f"Files in zip: {names}")
        inner = zf.read(names[0])
    df = pd.read_excel(io.BytesIO(inner), header=0)
    print(f"Shape: {df.shape}")
    print(f"First 3 columns: {list(df.columns[:3])}")

# ── PARSE (current-week .txt — headerless CSV) ────────────────
def parse_cot(raw_bytes):
    df = pd.read_csv(
        io.BytesIO(raw_bytes), header=None, usecols=USECOLS,
        dtype=str, low_memory=False,
    )
    df.rename(columns=COLMAP, inplace=True)
    df["contract_code"] = df["contract_code"].str.strip()
    df["date"]          = df["date"].str.strip()
    df = df[df["contract_code"].isin(ALL_CODES)].copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
    return df

# ── PARSE (historical zip → .xls Excel files) ─────────────────
def parse_cot_xls(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes), header=0)
    missing = [c for c in EXCEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found in XLS: {missing}")
    df = df[EXCEL_COLS].copy()
    df.rename(columns=EXCEL_COLMAP, inplace=True)
    df["contract_code"] = df["contract_code"].astype(str).str.strip()
    df = df[df["contract_code"].isin(ALL_CODES)].copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for col in NUM_COLS:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ── DOWNLOAD HISTORICAL (2014–2025) ───────────────────────────
all_frames = []

print("Downloading CFTC COT historical data...")
for year in YEARS:
    url = HIST_URL.format(year=year)
    print(f"  {year}...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            inner = zf.read(zf.namelist()[0])
        df = parse_cot_xls(inner)
        print(f"{len(df)} matching rows")
        if not df.empty:
            all_frames.append(df)
    except Exception as e:
        print(f"FAILED ({e})")

# ── DOWNLOAD CURRENT WEEK ─────────────────────────────────────
print("\nDownloading current week (FinFutWk.txt)...")
try:
    r = requests.get(WEEK_URL, timeout=30)
    r.raise_for_status()
    df = parse_cot(r.content)
    print(f"  {len(df)} matching rows")
    if not df.empty:
        all_frames.append(df)
except Exception as e:
    print(f"  FAILED ({e})")

if not all_frames:
    raise RuntimeError("No data downloaded — check network and CFTC URLs.")

# ── COMBINE ───────────────────────────────────────────────────
print("\nCombining all frames...")
raw = pd.concat(all_frames, ignore_index=True)
raw = raw.sort_values(["date", "open_interest"], ascending=[True, False])
raw = raw.drop_duplicates(subset=["date", "contract_code"])
print(f"  Total rows:  {len(raw)}")
print(f"  Date range:  {raw['date'].min().date()} → {raw['date'].max().date()}")
print(f"  Codes found: {sorted(raw['contract_code'].unique())}")

# ── FEATURE BUILDER ───────────────────────────────────────────
def rolling_pctile(series, window=156):
    arr = series.values.astype(float)
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            continue
        start = max(0, i - window + 1)
        w = arr[start:i + 1]
        w = w[~np.isnan(w)]
        if len(w) >= 2:
            out[i] = (w < arr[i]).mean()
    return pd.Series(out, index=series.index)

def build_features(subset):
    df = (subset
          .sort_values(["date", "open_interest"], ascending=[True, False])
          .drop_duplicates("date")
          .set_index("date"))

    # Primary: Leveraged Money (speculative hedge funds — extremes predict reversals)
    df["lev_net"]         = df["lev_longs"] - df["lev_shorts"]
    df["lev_net_pct"]     = df["lev_net"] / df["open_interest"]
    df["lev_net_wow"]     = df["lev_net"].diff(1)
    df["lev_net_pct_wow"] = df["lev_net_pct"].diff(1)
    df["lev_net_4w"]      = df["lev_net"].diff(4)

    # Secondary: Asset Manager (institutional trend-followers)
    df["asset_mgr_net"]     = df["asset_mgr_longs"] - df["asset_mgr_shorts"]
    df["asset_mgr_net_pct"] = df["asset_mgr_net"] / df["open_interest"]

    # Dealer net (market-makers — typically opposite to lev money)
    df["dealer_net"]     = df["dealer_longs"] - df["dealer_shorts"]
    df["dealer_net_pct"] = df["dealer_net"] / df["open_interest"]

    # Positioning signal based on leveraged fund percentile (3-year rolling window)
    df["positioning_pctile"] = rolling_pctile(df["lev_net_pct"])
    df["positioning_signal"] = df["positioning_pctile"].apply(
        lambda x: "EXTREME_LONG"  if x > 0.80
        else      "EXTREME_SHORT" if x < 0.20
        else      "NEUTRAL"       if not pd.isna(x)
        else      np.nan
    )
    return df

# ── NQ ────────────────────────────────────────────────────────
print("\nBuilding NQ features...")
nq     = build_features(raw[raw["contract_code"].isin(NQ_CODES)].copy())
nq_out = os.path.join(OUT_DIR, "cot_NQ.csv")
nq.to_csv(nq_out)
print(f"  {len(nq)} weeks  →  {nq_out}")

# ── ES ────────────────────────────────────────────────────────
print("\nBuilding ES features...")
es     = build_features(raw[raw["contract_code"].isin(ES_CODES)].copy())
es_out = os.path.join(OUT_DIR, "cot_ES.csv")
es.to_csv(es_out)
print(f"  {len(es)} weeks  →  {es_out}")

# ── SUMMARY ───────────────────────────────────────────────────
print("\n" + "="*60)
print("  COT POSITIONING SUMMARY")
print("="*60)

for label, df in [("NQ (NASDAQ-100)", nq), ("ES (S&P 500)", es)]:
    valid = df.dropna(subset=["positioning_pctile"])
    if valid.empty:
        print(f"\n  {label}  — no valid rows")
        continue
    row = valid.iloc[-1]
    print(f"\n  {label}")
    print(f"    Date range:              {df.index.min().date()} → {df.index.max().date()}")
    print(f"    Weeks of data:           {len(df)}")
    print(f"    Latest date:             {row.name.date()}")
    print(f"    Open interest:           {row['open_interest']:>12,.0f}")
    print(f"    Lev Money net:           {row['lev_net']:>+12,.0f}")
    print(f"    Lev Money net % OI:      {row['lev_net_pct']:>+12.3f}")
    print(f"    Lev Money WoW change:    {row['lev_net_wow']:>+12,.0f}")
    print(f"    Lev Money 4W change:     {row['lev_net_4w']:>+12,.0f}")
    print(f"    Asset Mgr net:           {row['asset_mgr_net']:>+12,.0f}")
    print(f"    Asset Mgr net % OI:      {row['asset_mgr_net_pct']:>+12.3f}")
    print(f"    Dealer net:              {row['dealer_net']:>+12,.0f}")
    print(f"    Positioning pctile:      {row['positioning_pctile']:>12.1%}")
    print(f"    ► Signal:                {row['positioning_signal']}")
