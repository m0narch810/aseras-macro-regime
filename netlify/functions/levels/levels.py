"""
Netlify Python function — FreeFlow levels endpoint.
GET /.netlify/functions/levels
"""

import json
import os
from collections import defaultdict
from datetime import datetime, date, timedelta

import requests


SYMBOL     = "QQQ"
BASE_URL   = "https://www.free-flow.site/api"
FILTER_PCT = 5.0
MIN_SCORE  = 20.0

HEADERS = {
    "Accept":             "*/*",
    "Accept-Language":    "en-US,en;q=0.9",
    "Connection":         "keep-alive",
    "Referer":            "https://www.free-flow.site/?auth=success",
    "Sec-Fetch-Dest":     "empty",
    "Sec-Fetch-Mode":     "cors",
    "Sec-Fetch-Site":     "same-origin",
    "User-Agent":         (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua":          '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    "sec-ch-ua-mobile":   "?0",
    "sec-ch-ua-platform": '"Windows"',
}

RESPONSE_HEADERS = {
    "Content-Type":                "application/json",
    "Access-Control-Allow-Origin": "*",
    "Cache-Control":               "public, max-age=240",
}


def next_trading_days(n=3):
    days, d = [], date.today()
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days


def make_session():
    s = requests.Session()
    s.cookies.set("ff_session", os.environ.get("FF_SESSION", ""))
    s.headers.update(HEADERS)
    return s


def fetch_futures_levels(session, exp):
    r = session.get(f"{BASE_URL}/futures-levels?symbol={SYMBOL}&exp={exp}", timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_vol_context(session):
    ctx = {"current_iv": None, "rv_iv_ratio": None, "hv21": None}
    try:
        r = session.get(f"{BASE_URL}/vol/realized?symbol={SYMBOL}", timeout=15)
        r.raise_for_status()
        d = r.json()
        ctx["current_iv"]  = d.get("current_iv")
        ctx["rv_iv_ratio"] = d.get("rv_iv_ratio")
        ctx["hv21"]        = d.get("hv21")
    except Exception:
        pass
    return ctx


def aggregate_multi(datasets):
    """Aggregate rows across multiple expiries by strike using plain dicts."""
    strikes = defaultdict(lambda: {
        "strike_etf": 0, "net_gex": 0, "abs_gex": 0,
        "net_vex": 0, "net_charmex": 0, "net_dex": 0,
        "net_vegaex": 0, "net_dag": 0, "total_oi": 0,
        "_etf_set": False,
    })

    futures_price = 0
    spot_etf      = 0
    first         = True

    for data in datasets:
        rows = data.get("rows", [])
        if not rows:
            continue
        ratio = data.get("ratio", 41.14)
        if first:
            futures_price = data.get("futures_price", 0)
            spot_etf      = data.get("etf_spot", 0)
            first         = False
        for row in rows:
            etf = row.get("strike_etf", 0)
            sf  = row.get("strike_futures")
            if sf is None:
                sf = round(etf * ratio, 1)
            else:
                sf = round(sf, 1)
            s = strikes[sf]
            if not s["_etf_set"]:
                s["strike_etf"] = etf
                s["_etf_set"]   = True
            s["net_gex"]     += row.get("gex",     0) or 0
            s["abs_gex"]     += row.get("ag",      0) or 0
            s["net_vex"]     += row.get("vex",     0) or 0
            s["net_charmex"] += row.get("charmex", 0) or 0
            s["net_dex"]     += row.get("dex",     0) or 0
            s["net_vegaex"]  += row.get("vegaex",  0) or 0
            s["net_dag"]     += row.get("dag",     0) or 0
            s["total_oi"]    += row.get("oi",      0) or 0

    return strikes, futures_price, spot_etf


def classify_regime(ctx):
    iv    = ctx.get("current_iv")  or 37.0
    rv_iv = ctx.get("rv_iv_ratio") or 0.46
    if iv >= 30 or rv_iv < 0.5:
        return "EXPANSION",   {"gex": 0.20, "vex": 0.38, "charmex": 0.17, "oi": 0.15, "dag": 0.10}
    if iv >= 20:
        return "NEUTRAL",     {"gex": 0.32, "vex": 0.28, "charmex": 0.15, "oi": 0.15, "dag": 0.10}
    return     "CONTRACTION", {"gex": 0.50, "vex": 0.15, "charmex": 0.15, "oi": 0.15, "dag": 0.05}


def normalize_col(values):
    """Min-max normalize a list of absolute values, returns list of 0-1 floats."""
    abs_vals = [abs(v) for v in values]
    mn, mx = min(abs_vals), max(abs_vals)
    if mx == mn:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn) for v in abs_vals]


def score_levels(strikes, weights, futures_price):
    if not strikes or futures_price == 0:
        return []

    # Filter to ±5% of current price
    nearby = []
    for sf, s in strikes.items():
        dist    = sf - futures_price
        dist_pct = (dist / futures_price) * 100
        if abs(dist_pct) <= FILTER_PCT:
            nearby.append({**s, "strike_futures": sf, "dist_nq": dist})

    if not nearby:
        return []

    # Normalize each metric across the nearby set
    gex_n     = normalize_col([r["net_gex"]     for r in nearby])
    vex_n     = normalize_col([r["net_vex"]     for r in nearby])
    charm_n   = normalize_col([r["net_charmex"] for r in nearby])
    oi_n      = normalize_col([r["total_oi"]    for r in nearby])
    dag_n     = normalize_col([r["net_dag"]     for r in nearby])

    scored = []
    for i, row in enumerate(nearby):
        score = (
            gex_n[i]   * weights["gex"]     +
            vex_n[i]   * weights["vex"]      +
            charm_n[i] * weights["charmex"]  +
            oi_n[i]    * weights["oi"]       +
            dag_n[i]   * weights["dag"]
        ) * 100

        if score < MIN_SCORE:
            continue

        g        = row["net_gex"]
        v        = row["net_vex"]
        vol_sens = abs(v) / (abs(g) + 1e-9)
        base     = "CALL WALL" if g > 0 else "PUT WALL"
        tag      = " + VOL SENSITIVE" if vol_sens > 2.0 else ""

        scored.append({
            "strike_futures": round(row["strike_futures"], 1),
            "strike_etf":     round(row["strike_etf"], 2),
            "dist_nq":        round(row["dist_nq"], 1),
            "score":          round(score, 1),
            "type":           base + tag,
            "net_gex":        int(row["net_gex"]),
            "net_vex":        int(row["net_vex"]),
            "net_charmex":    int(row["net_charmex"]),
            "total_oi":       int(row["total_oi"]),
        })

    return sorted(scored, key=lambda x: x["score"], reverse=True)


def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": RESPONSE_HEADERS, "body": ""}

    try:
        session  = make_session()
        expiries = next_trading_days(3)

        datasets = []
        for exp in expiries:
            try:
                datasets.append(fetch_futures_levels(session, exp))
            except Exception:
                pass

        if not datasets:
            raise ValueError("All FreeFlow requests failed — FF_SESSION may be expired.")

        ctx                          = fetch_vol_context(session)
        strikes, nq_price, qqq_price = aggregate_multi(datasets)
        regime, weights              = classify_regime(ctx)
        levels                       = score_levels(strikes, weights, nq_price)

        iv    = ctx.get("current_iv")
        rv_iv = ctx.get("rv_iv_ratio")
        hv21  = ctx.get("hv21")

        payload = {
            "updated":     datetime.utcnow().strftime("%Y-%m-%d %H:%M ET"),
            "nq_price":    round(float(nq_price),  1),
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
