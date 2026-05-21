"""
Netlify Python serverless function — FreeFlow levels endpoint.
GET /.netlify/functions/levels
Returns scored NQ gamma levels as JSON with a 4-minute CDN cache.
"""

import json
import os
from datetime import datetime, date, timedelta

import requests
import pandas as pd


# ── CONFIG ────────────────────────────────────────────────────
SYMBOL     = "QQQ"
BASE_URL   = "https://www.free-flow.site/api"
FILTER_PCT = 5.0
MIN_SCORE  = 20.0

HEADERS = {
    "Accept":            "*/*",
    "Accept-Language":   "en-US,en;q=0.9",
    "Connection":        "keep-alive",
    "Referer":           "https://www.free-flow.site/?auth=success",
    "Sec-Fetch-Dest":    "empty",
    "Sec-Fetch-Mode":    "cors",
    "Sec-Fetch-Site":    "same-origin",
    "User-Agent":        (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua":         '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    "sec-ch-ua-mobile":  "?0",
    "sec-ch-ua-platform": '"Windows"',
}

RESPONSE_HEADERS = {
    "Content-Type":                "application/json",
    "Access-Control-Allow-Origin": "*",
    "Cache-Control":               "public, max-age=240",
}


# ── DATE HELPERS ──────────────────────────────────────────────
def next_trading_days(n=3):
    days = []
    d = date.today()
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days


# ── SESSION ───────────────────────────────────────────────────
def make_session():
    ff_session = os.environ.get("FF_SESSION", "")
    s = requests.Session()
    s.cookies.set("ff_session", ff_session)
    s.headers.update(HEADERS)
    return s


# ── FETCHERS ──────────────────────────────────────────────────
def fetch_futures_levels(session, exp):
    url = f"{BASE_URL}/futures-levels?symbol={SYMBOL}&exp={exp}"
    r = session.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_vol_context(session):
    ctx = {
        "current_iv":  None,
        "rv_iv_ratio": None,
        "hv21":        None,
    }
    try:
        url = f"{BASE_URL}/vol/realized?symbol={SYMBOL}"
        r = session.get(url, timeout=15)
        r.raise_for_status()
        d = r.json()
        ctx["current_iv"]  = d.get("current_iv")
        ctx["rv_iv_ratio"] = d.get("rv_iv_ratio")
        ctx["hv21"]        = d.get("hv21")
    except Exception:
        pass
    return ctx


# ── AGGREGATION ───────────────────────────────────────────────
def aggregate_multi(datasets):
    dfs = []
    for data in datasets:
        rows = data.get("rows", [])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "strike_futures" not in df.columns:
            ratio = data.get("ratio", 41.14)
            df["strike_futures"] = (df["strike_etf"] * ratio).round(1)
        dfs.append(df)

    if not dfs:
        return None, 0, 0

    combined = pd.concat(dfs, ignore_index=True)
    agg = combined.groupby("strike_futures").agg(
        strike_etf  = ("strike_etf",  "first"),
        net_gex     = ("gex",         "sum"),
        abs_gex     = ("ag",          "sum"),
        net_vex     = ("vex",         "sum"),
        net_charmex = ("charmex",     "sum"),
        net_dex     = ("dex",         "sum"),
        net_vegaex  = ("vegaex",      "sum"),
        net_dag     = ("dag",         "sum"),
        total_oi    = ("oi",          "sum"),
    ).reset_index()

    first = datasets[0]
    futures_price = first.get("futures_price", 0)
    spot_etf      = first.get("etf_spot", 0)
    return agg, futures_price, spot_etf


# ── REGIME + WEIGHTS ─────────────────────────────────────────
def classify_regime(ctx):
    iv    = ctx.get("current_iv")  or 37.0
    rv_iv = ctx.get("rv_iv_ratio") or 0.46

    if iv >= 30 or rv_iv < 0.5:
        return "EXPANSION",   {"gex": 0.20, "vex": 0.38, "charmex": 0.17, "oi": 0.15, "dag": 0.10}
    if iv >= 20:
        return "NEUTRAL",     {"gex": 0.32, "vex": 0.28, "charmex": 0.15, "oi": 0.15, "dag": 0.10}
    return     "CONTRACTION", {"gex": 0.50, "vex": 0.15, "charmex": 0.15, "oi": 0.15, "dag": 0.05}


# ── SCORING ───────────────────────────────────────────────────
def normalize(series):
    s = series.abs()
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (s - mn) / (mx - mn)


def score_levels(agg, weights, futures_price):
    agg = agg.copy()
    agg["dist_nq"]  = agg["strike_futures"] - futures_price
    agg["dist_pct"] = (agg["dist_nq"] / futures_price) * 100

    nearby = agg[agg["dist_pct"].abs() <= FILTER_PCT].copy()
    if nearby.empty:
        return nearby

    nearby["gex_n"]     = normalize(nearby["net_gex"])
    nearby["vex_n"]     = normalize(nearby["net_vex"])
    nearby["charmex_n"] = normalize(nearby["net_charmex"])
    nearby["oi_n"]      = normalize(nearby["total_oi"])
    nearby["dag_n"]     = normalize(nearby["net_dag"])

    nearby["score"] = (
        nearby["gex_n"]     * weights["gex"]     +
        nearby["vex_n"]     * weights["vex"]      +
        nearby["charmex_n"] * weights["charmex"]  +
        nearby["oi_n"]      * weights["oi"]       +
        nearby["dag_n"]     * weights["dag"]
    ) * 100

    def classify(row):
        g        = row["net_gex"]
        v        = row["net_vex"]
        vol_sens = abs(v) / (abs(g) + 1e-9)
        base     = "CALL WALL" if g > 0 else "PUT WALL"
        tag      = " + VOL SENSITIVE" if vol_sens > 2.0 else ""
        return base + tag

    nearby["type"] = nearby.apply(classify, axis=1)
    nearby = nearby[nearby["score"] >= MIN_SCORE]
    return nearby.sort_values("score", ascending=False).reset_index(drop=True)


# ── HANDLER ───────────────────────────────────────────────────
def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": RESPONSE_HEADERS, "body": ""}

    try:
        session  = make_session()
        expiries = next_trading_days(3)

        datasets = []
        for exp in expiries:
            try:
                data = fetch_futures_levels(session, exp)
                datasets.append(data)
            except Exception:
                pass

        if not datasets:
            raise ValueError("All FreeFlow expiry requests failed — check FF_SESSION.")

        ctx              = fetch_vol_context(session)
        agg, nq_price, qqq_price = aggregate_multi(datasets)

        if agg is None:
            raise ValueError("No rows returned from FreeFlow API.")

        regime, weights = classify_regime(ctx)
        nearby          = score_levels(agg, weights, nq_price)

        levels = []
        if not nearby.empty:
            for _, row in nearby.iterrows():
                levels.append({
                    "strike_futures": round(float(row["strike_futures"]), 1),
                    "strike_etf":     round(float(row["strike_etf"]), 2),
                    "dist_nq":        round(float(row["dist_nq"]), 1),
                    "score":          round(float(row["score"]), 1),
                    "type":           str(row["type"]),
                    "net_gex":        int(row["net_gex"]),
                    "net_vex":        int(row["net_vex"]),
                    "net_charmex":    int(row.get("net_charmex", 0)),
                    "total_oi":       int(row["total_oi"]),
                })

        iv    = ctx.get("current_iv")
        rv_iv = ctx.get("rv_iv_ratio")
        hv21  = ctx.get("hv21")

        payload = {
            "updated":     datetime.utcnow().strftime("%Y-%m-%d %H:%M ET"),
            "nq_price":    round(float(nq_price), 1),
            "qqq_price":   round(float(qqq_price), 2),
            "regime":      regime,
            "iv":          round(float(iv),    1) if iv    is not None else None,
            "rv_iv_ratio": round(float(rv_iv), 3) if rv_iv is not None else None,
            "hv21":        round(float(hv21),  1) if hv21  is not None else None,
            "levels":      levels,
        }

        return {
            "statusCode": 200,
            "headers":    RESPONSE_HEADERS,
            "body":       json.dumps(payload),
        }

    except Exception as e:
        return {
            "statusCode": 200,
            "headers":    RESPONSE_HEADERS,
            "body":       json.dumps({"error": True, "message": str(e)}),
        }
