from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class OpenCycle:
    cash_out: float
    buy_price: float
    amount: float
    buy_time: Any


class GridEngine:
    """Simple grid engine (simulation-oriented).

    - Tracks per-level buy triggers and corresponding sell levels.
    - Maintains per-cycle accounting (cash_out / cash_in) for exact realized PnL.
    - Supports an optional per-cycle take-profit (CYCLE_TP) that exits a cycle early.
    - Supports an optional buy_guard(symbol, amount_base, limit_price, ts) -> (ok, reason).
    """

    def __init__(self, symbol: str, grid: List[float], order_size: float):
        self.symbol = symbol
        self.grid = sorted([float(x) for x in grid])
        if len(self.grid) < 2:
            raise ValueError("Grid must contain at least 2 levels")

        self.order_size = float(order_size)

        # Cycle TP (set by Streamlit per pair)
        self.enable_cycle_tp: bool = False
        self.cycle_tp_pct: float = 0.35

        # Internal state
        self.active_buys: Set[float] = set(self.grid[:-1])   # last level has no next sell
        self.active_sells: Set[float] = set()
        self.open_cycles: Dict[float, OpenCycle] = {}        # key: buy_level
        self.closed_cycles: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []

    def reset_open_cycles(self) -> None:
        """Clears all pending cycles and restores initial buy levels."""
        self.active_buys = set(self.grid[:-1])
        self.active_sells = set()
        self.open_cycles = {}

    def _next(self, level: float) -> float:
        i = self.grid.index(level)
        return self.grid[i + 1]

    def _prev(self, level: float) -> float:
        i = self.grid.index(level)
        return self.grid[i - 1]

    def check_price(self, price: float, trader, ts, allow_buys: bool = True, buy_guard=None) -> None:
        price = float(price)

        # ----------------------------
        # BUY
        # ----------------------------
        if allow_buys:
            for buy in list(self.active_buys):
                if price <= buy:
                    if buy_guard is not None:
                        ok, why = buy_guard(self.symbol, float(self.order_size), float(buy), ts)
                        if not ok:
                            if hasattr(trader, "record_blocked"):
                                trader.record_blocked("BUY", self.symbol, float(buy), float(self.order_size), ts, why)
                            continue

                    tr = trader.buy(self.symbol, float(buy), float(self.order_size), ts, reason="GRID")
                    if tr is None:
                        continue

                    self.active_buys.remove(buy)
                    sell = self._next(buy)
                    self.active_sells.add(sell)

                    cash_out = -float(tr.cash_delta_quote)  # positive
                    self.open_cycles[buy] = OpenCycle(
                        cash_out=cash_out,
                        buy_price=float(tr.price),
                        amount=float(tr.amount),
                        buy_time=tr.time,
                    )

                    self.trades.append({
                        "time": tr.time, "symbol": tr.symbol, "side": tr.side,
                        "price": float(tr.price), "amount": float(tr.amount),
                        "fee_rate": float(tr.fee_rate), "fee_paid": float(tr.fee_paid_quote),
                        "cash_delta": float(tr.cash_delta_quote),
                        "pnl": 0.0,
                        "reason": tr.reason,
                    })

        # ----------------------------
        # Per-cycle TP (optional)
        # ----------------------------
        if bool(getattr(self, "enable_cycle_tp", False)) and float(getattr(self, "cycle_tp_pct", 0.0)) > 0.0:
            tp_mult = 1.0 + (float(self.cycle_tp_pct) / 100.0)
            for buy_level, oc in list(self.open_cycles.items()):
                tp_price = float(oc.buy_price) * tp_mult
                if price >= tp_price:
                    tr = trader.sell(self.symbol, float(tp_price), float(oc.amount), ts, reason="CYCLE_TP")
                    if tr is None:
                        continue

                    cash_in = float(tr.cash_delta_quote)
                    pnl = cash_in - float(oc.cash_out)

                    self.closed_cycles.append({
                        "symbol": tr.symbol,
                        "buy_time": oc.buy_time, "sell_time": tr.time,
                        "buy_price": float(oc.buy_price), "sell_price": float(tr.price),
                        "amount": float(tr.amount),
                        "cash_out": float(oc.cash_out), "cash_in": cash_in,
                        "pnl": pnl,
                    })

                    sell_level = self._next(buy_level)
                    if sell_level in self.active_sells:
                        self.active_sells.remove(sell_level)

                    self.open_cycles.pop(buy_level, None)
                    self.active_buys.add(buy_level)

                    self.trades.append({
                        "time": tr.time, "symbol": tr.symbol, "side": tr.side,
                        "price": float(tr.price), "amount": float(tr.amount),
                        "fee_rate": float(tr.fee_rate), "fee_paid": float(tr.fee_paid_quote),
                        "cash_delta": float(tr.cash_delta_quote),
                        "pnl": pnl,
                        "reason": tr.reason,
                    })

        # ----------------------------
        # SELL (grid target exits)
        # ----------------------------
        for sell in list(self.active_sells):
            if price >= sell:
                buy_level = self._prev(sell)
                oc = self.open_cycles.pop(buy_level, None)
                if oc is None:
                    continue

                tr = trader.sell(self.symbol, float(sell), float(oc.amount), ts, reason="GRID")
                if tr is None:
                    # restore if not filled
                    self.open_cycles[buy_level] = oc
                    continue

                cash_in = float(tr.cash_delta_quote)
                pnl = cash_in - float(oc.cash_out)

                self.closed_cycles.append({
                    "symbol": tr.symbol,
                    "buy_time": oc.buy_time, "sell_time": tr.time,
                    "buy_price": float(oc.buy_price), "sell_price": float(tr.price),
                    "amount": float(tr.amount),
                    "cash_out": float(oc.cash_out), "cash_in": cash_in,
                    "pnl": pnl,
                })

                self.active_sells.remove(sell)
                self.active_buys.add(buy_level)

                self.trades.append({
                    "time": tr.time, "symbol": tr.symbol, "side": tr.side,
                    "price": float(tr.price), "amount": float(tr.amount),
                    "fee_rate": float(tr.fee_rate), "fee_paid": float(tr.fee_paid_quote),
                    "cash_delta": float(tr.cash_delta_quote),
                    "pnl": pnl,
                    "reason": tr.reason,
                })
