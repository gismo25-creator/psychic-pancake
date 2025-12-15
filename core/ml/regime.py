import pandas as pd

def classify_regime(df, atr_pct, rv, bb_bw, adx_val) -> str:
    if pd.isna(adx_val) or pd.isna(rv) or pd.isna(bb_bw) or pd.isna(atr_pct):
        return "WARMUP"

    if adx_val >= 25 and bb_bw >= 0.01:
        return "TREND"

    if rv >= 0.01 or bb_bw >= 0.03 or atr_pct >= 0.02:
        return "CHAOS"

    return "RANGE"
