from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return float("nan")
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, np.nan)
    return float(dd.max())


def summarize_run(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, float]:
    eq = equity_curve["equity"].astype(float)
    total_pnl = float(eq.iloc[-1] - eq.iloc[0]) if len(eq) else float("nan")
    mdd = max_drawdown(eq)

    # realized pnl on SELL rows if present
    if trades is not None and not trades.empty and "pnl" in trades.columns:
        realized = float(trades.loc[trades["side"] == "SELL", "pnl"].astype(float).sum())
    else:
        realized = float("nan")

    # win-rate from SELL pnls
    win_rate = float("nan")
    if trades is not None and not trades.empty and "pnl" in trades.columns:
        sells = trades[trades["side"] == "SELL"].copy()
        if len(sells) > 0:
            w = (sells["pnl"].astype(float) > 0).sum()
            win_rate = float(w / len(sells))

    return {
        "start_equity": float(eq.iloc[0]) if len(eq) else float("nan"),
        "end_equity": float(eq.iloc[-1]) if len(eq) else float("nan"),
        "total_pnl": total_pnl,
        "max_drawdown": mdd,
        "realized_pnl": realized,
        "win_rate": win_rate,
        "num_trades": float(len(trades)) if trades is not None else 0.0,
    }
