from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.exchange.simulator import PortfolioSimulatorTrader
from core.grid.engine import GridEngine
from core.grid.linear import generate_linear_grid
from core.grid.fibonacci import generate_fibonacci_grid


def _grid_for_cfg(price: float, cfg: Dict) -> List[float]:
    range_pct = float(cfg.get("base_range_pct", 1.0))
    lower = float(price) * (1.0 - range_pct / 100.0)
    upper = float(price) * (1.0 + range_pct / 100.0)
    gtype = str(cfg.get("grid_type", "Linear"))
    if gtype == "Linear":
        levels = int(cfg.get("base_levels", 10))
        return generate_linear_grid(lower, upper, levels)
    return generate_fibonacci_grid(lower, upper)


def run_backtest(
    dfs: Dict[str, pd.DataFrame],
    pair_cfg: Dict[str, Dict],
    timeframe: str,
    start_cash: float,
    maker_fee: float,
    taker_fee: float,
    slippage: float,
    fee_mode: str,
    quote_ccy: str,
    max_exposure_quote: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, PortfolioSimulatorTrader]:
    """Multi-pair replay backtest on candle CLOSE prices.

    - Uses a fixed grid per pair based on the first available price in the test range.
    - Executes on CLOSE as mark/trigger price (simple, deterministic).
    """
    trader = PortfolioSimulatorTrader(
        cash_quote=float(start_cash),
        maker_fee=float(maker_fee),
        taker_fee=float(taker_fee),
        slippage=float(slippage),
        fee_mode=str(fee_mode),
        quote_ccy=str(quote_ccy),
        max_exposure_quote=max_exposure_quote or {},
    )

    engines: Dict[str, GridEngine] = {}
    mark_prices: Dict[str, float] = {}

    # Build an event list: (timestamp, symbol, close)
    events: List[Tuple[pd.Timestamp, str, float]] = []
    for sym, df in dfs.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            ts = row["timestamp"]
            px = float(row["close"])
            events.append((pd.Timestamp(ts), sym, px))

    events.sort(key=lambda x: x[0])

    # Initialize engines using first seen price for each pair
    first_price: Dict[str, float] = {}
    for ts, sym, px in events:
        if sym not in first_price:
            first_price[sym] = px

    for sym, px in first_price.items():
        cfg = pair_cfg.get(sym, {})
        grid = _grid_for_cfg(px, cfg)
        osize = float(cfg.get("order_size", 0.001))
        engines[sym] = GridEngine(sym, grid, osize)

    equity_rows = []
    # Replay
    for ts, sym, px in events:
        mark_prices[sym] = px
        eng = engines.get(sym)
        if eng is None:
            continue

        # Allow buys always; trader enforces cash/exposure caps.
        eng.check_price(px, trader, ts, allow_buys=True)

        eq = trader.equity(mark_prices)
        equity_rows.append({
            "timestamp": ts,
            "equity": eq,
            "cash": float(trader.cash),
        })

    equity_curve = pd.DataFrame(equity_rows)
    equity_curve = equity_curve.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Aggregate trades from engines
    all_trades = []
    for sym, eng in engines.items():
        for t in eng.trades:
            all_trades.append(t)

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("time").reset_index(drop=True)

    return trades_df, equity_curve, trader
