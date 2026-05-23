import os
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# ── PURPOSE ───────────────────────────────────────────────────
# For each week where a prediction was made AND the week has
# completed, record:
#   week_end_date    — Thursday of the predicted week
#   week_open        — NQ Monday open of that week (first daily bar)
#   week_close       — NQ Thursday close of that week (last daily bar)
#   oc_return        — (week_close - week_open) / week_open
#   actual_dir       — +1 if oc_return > +0.3%, -1 if < -0.3%, 0 otherwise
#   predicted_conf   — confluence string from saved predictions (e.g. "LEAN BULL")
#   predicted_dir    — +1 if BULL in predicted_conf, -1 if BEAR, 0 if MIXED/NEUTRAL
#   match            — True if actual_dir != 0 and predicted_dir == actual_dir
#   base_rate_match  — True if actual_dir == +1 (naive always-bullish baseline)
#
# Output: data/processed/weekly_accuracy_log.csv (append, no duplicates)
# Display: per-week table + summary stats with BOTH nulls reported

LOG_PATH = os.path.join(PROC_DIR, 'weekly_accuracy_log.csv')
PRED_PATH = os.path.join(PROC_DIR, 'blind_predictions_feb_may_2026.csv')

# Load NQ daily bars to compute weekly OC
nq = pd.read_csv(os.path.join(PROC_DIR, 'NQ_daily_clean.csv'),
                 parse_dates=['date']).set_index('date')

# Build weekly OC bars: open = first daily open of week, close = last daily close
nq_weekly_open  = nq['open'].resample('W-THU').first()
nq_weekly_close = nq['close'].resample('W-THU').last()
nq_oc = pd.DataFrame({'week_open': nq_weekly_open, 'week_close': nq_weekly_close})
nq_oc['oc_return'] = (nq_oc['week_close'] - nq_oc['week_open']) / nq_oc['week_open']

THRESHOLD = 0.003
def oc_label(r):
    if pd.isna(r): return np.nan
    if r >  THRESHOLD: return  1
    if r < -THRESHOLD: return -1
    return 0

nq_oc['actual_dir'] = nq_oc['oc_return'].apply(oc_label)

def parse_predicted_dir(conf_str):
    if pd.isna(conf_str): return np.nan
    c = str(conf_str).upper()
    if 'BULL' in c:   return  1
    if 'BEAR' in c:   return -1
    return 0

# Load saved predictions if they exist
records = []
if os.path.exists(PRED_PATH):
    preds = pd.read_csv(PRED_PATH, parse_dates=[0], index_col=0)
    for date, row in preds.iterrows():
        if date not in nq_oc.index:
            continue
        actual   = nq_oc.loc[date, 'actual_dir']
        pred_dir = parse_predicted_dir(row.get('confluence') or row.get('predicted_confluence'))
        if np.isnan(actual):
            continue
        records.append({
            'week_end_date':   date.date(),
            'week_open':       round(nq_oc.loc[date, 'week_open'], 2),
            'week_close':      round(nq_oc.loc[date, 'week_close'], 2),
            'oc_return_pct':   round(nq_oc.loc[date, 'oc_return'] * 100, 3),
            'actual_dir':      int(actual),
            'predicted_conf':  row.get('confluence', ''),
            'predicted_dir':   int(pred_dir) if not np.isnan(pred_dir) else 0,
            'match':           bool(actual != 0 and pred_dir == actual),
            'base_rate_match': bool(actual == 1),  # naive always-bullish baseline
        })

# Also check if this week's bias_output.json should be appended
bias_json_path = os.path.join(BASE_DIR, 'bias_output.json')
if os.path.exists(bias_json_path):
    import json
    with open(bias_json_path) as f:
        b = json.load(f)
    meta   = b.get('meta', {})
    week_end_str = meta.get('week_end')
    if week_end_str:
        week_end_dt = pd.Timestamp(week_end_str)
        if week_end_dt in nq_oc.index:
            actual = nq_oc.loc[week_end_dt, 'actual_dir']
            if not np.isnan(actual):
                pred_dir = parse_predicted_dir(b.get('confluence'))
                new_rec = {
                    'week_end_date':   week_end_dt.date(),
                    'week_open':       round(nq_oc.loc[week_end_dt, 'week_open'], 2),
                    'week_close':      round(nq_oc.loc[week_end_dt, 'week_close'], 2),
                    'oc_return_pct':   round(nq_oc.loc[week_end_dt, 'oc_return'] * 100, 3),
                    'actual_dir':      int(actual),
                    'predicted_conf':  b.get('confluence', ''),
                    'predicted_dir':   int(pred_dir) if not np.isnan(pred_dir) else 0,
                    'match':           bool(actual != 0 and pred_dir == actual),
                    'base_rate_match': bool(actual == 1),
                }
                if new_rec not in records:
                    records.append(new_rec)

# Load existing log and merge (no duplicates)
if os.path.exists(LOG_PATH):
    existing = pd.read_csv(LOG_PATH)
    existing['week_end_date'] = pd.to_datetime(existing['week_end_date']).dt.date
    existing_dates = set(existing['week_end_date'])
    records = [r for r in records if r['week_end_date'] not in existing_dates]
    log = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
else:
    log = pd.DataFrame(records)

log.to_csv(LOG_PATH, index=False)

# ── DISPLAY ───────────────────────────────────────────────────
print("="*70)
print("  WEEKLY ACCURACY LOG  (open-to-close labels)")
print("="*70)
log_display = log.copy()
log_display['actual_dir']    = log_display['actual_dir'].map({1:'BULL',0:'FLAT',-1:'BEAR'})
log_display['predicted_dir'] = log_display['predicted_dir'].map({1:'BULL',0:'FLAT',-1:'BEAR'})
log_display['match']         = log_display['match'].map({True:'✓', False:'✗'})
print(log_display[['week_end_date','oc_return_pct','actual_dir',
                    'predicted_conf','predicted_dir','match']].to_string(index=False))

# ── SUMMARY STATS ─────────────────────────────────────────────
from scipy import stats as scipy_stats

tradeable = log[(log['actual_dir'] != 0) & (log['predicted_dir'] != 0)]
n = len(tradeable)
n_match = tradeable['match'].sum()
hit_rate = n_match / n if n > 0 else np.nan
bull_base_rate = (log['actual_dir'] == 1).mean()

print(f"\n  Tradeable weeks (both sides non-neutral): {n}")
print(f"  Hit rate: {n_match}/{n} = {hit_rate:.1%}")
print(f"  Historical bull base rate (full log):     {bull_base_rate:.1%}")

if n > 0:
    # Test 1: vs 50% random
    z1 = (hit_rate - 0.5) / np.sqrt(0.25 / n)
    p1 = 1 - scipy_stats.norm.cdf(z1)
    # Test 2: vs bull base rate (correct null for equity-biased dataset)
    z2 = (hit_rate - bull_base_rate) / np.sqrt(bull_base_rate * (1-bull_base_rate) / n)
    p2 = 1 - scipy_stats.norm.cdf(z2)
    print(f"\n  vs 50% null:            z={z1:.2f}, p={p1:.4f}  "
          f"{'SIGNIFICANT' if p1<0.05 else 'not significant'}")
    print(f"  vs {bull_base_rate:.0%} bull base rate: z={z2:.2f}, p={p2:.4f}  "
          f"{'SIGNIFICANT' if p2<0.05 else 'not significant (meaningful null)'}")
    print(f"\n  NOTE: The bull base rate null is the correct benchmark for a")
    print(f"  trend-following bias on an equity index. Beating 50% random is")
    print(f"  insufficient — equities drift upward.")

print(f"\n  Log saved: {LOG_PATH}")
