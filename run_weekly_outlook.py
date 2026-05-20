import os
import argparse
import pandas as pd

def load_predictions(pred_path):
    return pd.read_csv(pred_path, index_col=0, parse_dates=True)

def format_outlook(row, date):
    data_origin = date - pd.Timedelta(days=7)
    applies_start = date - pd.Timedelta(days=6)
    applies_end = date

    lines = []
    lines.append(f"WEEKLY OUTLOOK — Week of {date.date()}")
    lines.append(f"DATA ORIGIN (H.4.1 snapshot): {data_origin.date()}")
    lines.append(f"APPLIES TO: {applies_start.date()} → {applies_end.date()}")
    lines.append("")
    lines.append(f"Regime:      {row.get('regime','N/A')}")
    lines.append(f"Macro score: {row.get('macro_score','N/A')}")
    lines.append(f"Confluence:   {row.get('confluence','N/A')}")
    lines.append(f"Price bias:  {row.get('price_dir','N/A')}  (p={row.get('price_prob',None):.3f})" if pd.notna(row.get('price_prob', None)) else f"Price bias:  {row.get('price_dir','N/A')}")
    vol = row.get('vol_forecast', None)
    if pd.notna(vol):
        lines.append(f"Vol forecast: {vol:.2%} (weekly est.)")
    else:
        lines.append("Vol forecast: N/A")

    lines.append("")
    lines.append("Summary: ")
    # Simple interpretation
    conf = row.get('confluence','').upper()
    if 'STRONG BULL' in conf:
        lines.append("  Strong bullish bias — high conviction")
    elif 'STRONG BEAR' in conf:
        lines.append("  Strong bearish bias — high conviction")
    elif 'LEAN BULL' in conf:
        lines.append("  Mild bullish bias — low conviction")
    elif 'LEAN BEAR' in conf:
        lines.append("  Mild bearish bias — low conviction")
    else:
        lines.append("  Mixed signals — no clear directional bias")

    lines.append("")
    lines.append("Suggested handling: keep position sizes moderate, use tight risk management, and monitor news that could change the H.4.1 picture.")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Print weekly outlook from H.4.1 snapshot predictions.")
    parser.add_argument('--pred', help='Path to predictions CSV (defaults to repository data/processed file)')
    parser.add_argument('--date', help='Week date to show (YYYY-MM-DD). If omitted, shows most recent entry.')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    default_pred = os.path.join(base, 'data', 'processed', 'blind_predictions_feb_may_2026.csv')
    pred_path = args.pred if args.pred else default_pred

    if not os.path.exists(pred_path):
        print(f"Predictions file not found: {pred_path}")
        return

    preds = load_predictions(pred_path)
    if preds.empty:
        print("No predictions found in file.")
        return

    if args.date:
        try:
            sel_date = pd.to_datetime(args.date)
        except Exception:
            print("Invalid date format. Use YYYY-MM-DD.")
            return
        if sel_date not in preds.index:
            print(f"Date {sel_date.date()} not found in predictions file.")
            return
        row = preds.loc[sel_date].to_dict()
        out = format_outlook(row, sel_date)
        print(out)
    else:
        last_date = preds.index.max()
        row = preds.loc[last_date].to_dict()
        out = format_outlook(row, last_date)
        print(out)

if __name__ == '__main__':
    main()
