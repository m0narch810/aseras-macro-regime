import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

SIGNAL_MAP = {"EXTREME_SHORT": -1, "NEUTRAL": 0, "EXTREME_LONG": 1}

# COT feature columns to pull, mapped to output names
NQ_COLS = {
    "lev_net_pct":     "nq_lev_net_pct",
    "lev_net_pct_wow": "nq_lev_wow",
    "lev_net_4w":      "nq_lev_4w",
    "positioning_pctile": "nq_lev_pctile",
    "asset_mgr_net_pct":  "nq_asset_mgr_net_pct",
    "positioning_signal": "nq_positioning_signal",
}
ES_COLS = {
    "lev_net_pct":     "es_lev_net_pct",
    "lev_net_pct_wow": "es_lev_wow",
    "lev_net_4w":      "es_lev_4w",
    "positioning_pctile": "es_lev_pctile",
    "asset_mgr_net_pct":  "es_asset_mgr_net_pct",
    "positioning_signal": "es_positioning_signal",
}


def load_cot(filename, col_map, signal_col):
    path = os.path.join(PROC_DIR, filename)
    df = pd.read_csv(path, parse_dates=["date"])

    # COT is reported for Tuesday and published the following Friday.
    # Shift forward 1 week so the signal only applies to the next week's
    # Thursday close — Tuesday + 9 days lands on the following Thursday.
    df["merge_date"] = df["date"] + pd.Timedelta(days=9)

    df = df[["merge_date"] + list(col_map.keys())].rename(columns=col_map)
    df[signal_col] = df[signal_col].map(SIGNAL_MAP)
    df = df.set_index("merge_date")
    df.index.name = None
    return df


def main():
    nq_cot = load_cot("cot_NQ.csv", NQ_COLS, "nq_positioning_signal")
    es_cot = load_cot("cot_ES.csv", ES_COLS, "es_positioning_signal")

    md_path = os.path.join(PROC_DIR, "model_dataset.csv")
    md = pd.read_csv(md_path, index_col=0, parse_dates=True)

    md = md.join(nq_cot, how="left")
    md = md.join(es_cot, how="left")

    out_path = os.path.join(PROC_DIR, "model_dataset_cot.csv")
    md.to_csv(out_path)

    total = len(md)
    nq_filled = md["nq_lev_net_pct"].notna().sum()
    es_filled = md["es_lev_net_pct"].notna().sum()

    print(f"Shape:      {md.shape}")
    print(f"Date range: {md.index.min().date()} to {md.index.max().date()}")
    print()
    print(f"NQ COT: {nq_filled}/{total} weeks with data  ({total - nq_filled} missing)")
    print(f"ES COT: {es_filled}/{total} weeks with data  ({total - es_filled} missing)")
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
