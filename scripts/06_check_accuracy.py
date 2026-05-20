import os
import pandas as pd
import yfinance as yf

# ── CONFIG ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# ── LOAD BLIND PREDICTIONS ────────────────────────────────────
preds = pd.read_csv(os.path.join(PROC_DIR, "blind_predictions_feb_may_2026.csv"),
                    index_col=0, parse_dates=True)
print(f"Loaded {len(preds)} predictions")
print(f"  {preds.index.min().date()} → {preds.index.max().date()}")

# ── PULL ACTUAL NQ DATA ───────────────────────────────────────
print("\nPulling actual NQ data...")
nq = yf.download("NQ=F", start="2026-01-25", end="2026-05-25",
                 interval="1d", progress=False)
if isinstance(nq.columns, pd.MultiIndex):
    nq.columns = [c[0] for c in nq.columns]
nq.columns = [c.lower() for c in nq.columns]

# Build actual weekly returns
nq_weekly = nq["close"].resample("W-THU").agg(
    weekly_open="first", weekly_close="last"
)
nq_weekly["actual_return"] = nq_weekly["weekly_close"].pct_change()

NEUTRAL_THRESHOLD = 0.003

def label_actual(ret):
    if pd.isna(ret):
        return "N/A"
    if ret > NEUTRAL_THRESHOLD:
        return "BULL"
    elif ret < -NEUTRAL_THRESHOLD:
        return "BEAR"
    else:
        return "FLAT"

nq_weekly["actual_dir"] = nq_weekly["actual_return"].apply(label_actual)

# ── MERGE AND COMPARE ─────────────────────────────────────────
results = preds.join(nq_weekly[["actual_return", "actual_dir"]], how="left")

# Determine predicted direction from confluence
def pred_direction(row):
    c = row["confluence"]
    if "BULL" in c:
        return "BULL"
    elif "BEAR" in c:
        return "BEAR"
    else:
        return "MIXED"

results["pred_dir"] = results.apply(pred_direction, axis=1)

# Score correctness
def check_correct(row):
    if row["actual_dir"] == "N/A" or row["actual_dir"] == "FLAT":
        return "FLAT"
    if row["pred_dir"] == "MIXED":
        return "SKIP"
    if row["pred_dir"] == row["actual_dir"]:
        return "✓"
    else:
        return "✗"

results["result"] = results.apply(check_correct, axis=1)

# ── PRINT RESULTS ─────────────────────────────────────────────
print("\n" + "="*70)
print("  ACCURACY CHECK — BLIND PREDICTIONS vs ACTUALS")
print("="*70)

print(f"\n{'Date':<14} {'Regime':<16} {'Confluence':<14} "
      f"{'Predicted':<10} {'Actual':<8} {'Return':<10} {'Result'}")
print("-" * 82)

for date, row in results.iterrows():
    ret_str = f"{row['actual_return']:+.2%}" if not pd.isna(row["actual_return"]) else "N/A"
    print(f"  {str(date.date()):<12} {row['regime']:<16} {row['confluence']:<14} "
          f"{row['pred_dir']:<10} {row['actual_dir']:<8} {ret_str:<10} {row['result']}")

# ── SUMMARY STATS ─────────────────────────────────────────────
tradeable = results[results["result"].isin(["✓", "✗"])]
correct   = (tradeable["result"] == "✓").sum()
total     = len(tradeable)
skipped   = (results["result"] == "SKIP").sum()
flat      = (results["result"] == "FLAT").sum()

print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  Total weeks:      {len(results)}")
print(f"  Tradeable calls:  {total}")
print(f"  Correct:          {correct}")
print(f"  Wrong:            {total - correct}")
print(f"  Accuracy:         {correct/total:.1%}" if total > 0 else "  Accuracy: N/A")
print(f"  Skipped (mixed):  {skipped}")
print(f"  Flat weeks:       {flat}")

# Accuracy by conviction level
print(f"\n  By conviction:")
for conv in ["STRONG BULL", "STRONG BEAR", "LEAN BULL", "LEAN BEAR"]:
    sub = results[(results["confluence"] == conv) &
                  (results["result"].isin(["✓", "✗"]))]
    if len(sub) > 0:
        c = (sub["result"] == "✓").sum()
        print(f"    {conv:<14}  {c}/{len(sub)} correct  ({c/len(sub):.0%})")

# By regime
print(f"\n  By macro regime:")
for reg in ["RISK-ON", "LEAN RISK-ON", "TRANSITION",
            "LEAN RISK-OFF", "RISK-OFF"]:
    sub = results[(results["regime"] == reg) &
                  (results["result"].isin(["✓", "✗"]))]
    if len(sub) > 0:
        c = (sub["result"] == "✓").sum()
        print(f"    {reg:<16}  {c}/{len(sub)} correct  ({c/len(sub):.0%})")

print(f"\n{'='*70}")