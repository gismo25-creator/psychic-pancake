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

        # Inventory management: time-stop per cycle (interpretable)
        self.enable_time_stop: bool = False
        self.time_stop_hours: float = 48.0
        # modes: BREAK_EVEN_NET, REDUCE_TO_TP, DECAY_TO_TP
        self.time_stop_mode: str = "BREAK_EVEN_NET"
        # floor TP (%) used for REDUCE_TO_TP / DECAY_TO_TP
        self.time_stop_floor_tp_pct: float = 0.20

        # Internal state
        self.active_buys: Set[float] = set(self.grid[:-1])   # last level has no next sell
        self.active_sells: Set[float] = set()
        self.open_cycles: Dict[float, OpenCycle] = {}        # key: buy_level
        self.closed_cycles: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        # Intrabar safety: prevent repeated fills at the same buy-level within the same bar timestamp.
        # This mitigates duplicate BUY/SELL sequences when using intrabar OHLC replay.
        self.enable_bar_fill_guard: bool = True
        self._guard_bar_ts = None
        self._guard_buys_done: Set[float] = set()
        self._guard_sells_done: Set[float] = set()

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

    def _age_hours(self, now, then) -> float:
        try:
            delta = now - then
            if hasattr(delta, "total_seconds"):
                return float(delta.total_seconds()) / 3600.0
        except Exception:
            pass
        return 0.0

    def _fee_rate(self, trader) -> float:
        try:
            fr = getattr(trader, "fee_rate", None)
            if callable(fr):
                return float(fr())
        except Exception:
            pass
        try:
            mode = str(getattr(trader, "fee_mode", "maker"))
            if mode == "taker":
                return float(getattr(trader, "taker_fee", 0.0))
            return float(getattr(trader, "maker_fee", 0.0))
        except Exception:
            return 0.0

    def _slippage(self, trader) -> float:
        try:
            return float(getattr(trader, "slippage", getattr(trader, "slippage_pct", 0.0)))
        except Exception:
            return 0.0

    def _net_breakeven_limit(self, trader, oc: OpenCycle) -> float:
        """Limit price required so that (after sell slippage + fee) cash_in >= oc.cash_out."""
        fee = max(0.0, self._fee_rate(trader))
        slip = max(0.0, self._slippage(trader))
        denom = float(oc.amount) * max(1e-12, (1.0 - slip)) * max(1e-12, (1.0 - fee))
        return float(oc.cash_out) / denom

    def check_price(self, price: float, trader, ts, allow_buys: bool = True, buy_guard=None) -> None:
        price = float(price)

        # Per-bar fill guard (keyed by ts): reset guards when moving to a new candle/bar.
        if bool(getattr(self, 'enable_bar_fill_guard', True)):
            if getattr(self, '_guard_bar_ts', None) != ts:
                self._guard_bar_ts = ts
                self._guard_buys_done = set()
                self._guard_sells_done = set()


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

                    # Bar-level guard: avoid repeated BUY at the same level within the same bar,
                    # and avoid re-buying a level that was already sold in the same bar.
                    if bool(getattr(self, 'enable_bar_fill_guard', True)):
                        if float(buy) in getattr(self, '_guard_buys_done', set()) or float(buy) in getattr(self, '_guard_sells_done', set()):
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


                    if bool(getattr(self, 'enable_bar_fill_guard', True)):
                        self._guard_buys_done.add(float(buy))

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
                oc = self.open_cycles.get(buy_level)
                if oc is None:
                    continue

                # Floor-aware grid exit (Option B): do not sell below net break-even and/or cycle TP target.
                be_limit = self._net_breakeven_limit(trader, oc)
                exit_price = max(float(sell), float(be_limit))

                if bool(getattr(self, "enable_cycle_tp", False)) and float(getattr(self, "cycle_tp_pct", 0.0)) > 0.0:
                    tp_mult = 1.0 + (float(getattr(self, "cycle_tp_pct", 0.0)) / 100.0)
                    tp_price = float(oc.buy_price) * tp_mult
                    exit_price = max(exit_price, float(tp_price))

                # Only execute once price reaches the computed floor.
                if price < float(exit_price):
                    continue

                reason = "GRID_FLOOR" if float(exit_price) > float(sell) else "GRID"
                # Bar-level guard: avoid repeated SELL for the same buy_level within the same bar.
                if bool(getattr(self, 'enable_bar_fill_guard', True)):
                    if float(buy_level) in getattr(self, '_guard_sells_done', set()):
                        continue

                tr = trader.sell(self.symbol, float(exit_price), float(oc.amount), ts, reason=reason)
                if tr is None:
                    continue

                # Remove the open cycle only after successful execution
                self.open_cycles.pop(buy_level, None)

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

                if bool(getattr(self, 'enable_bar_fill_guard', True)):
                    self._guard_sells_done.add(float(buy_level))


                self.trades.append({
                    "time": tr.time, "symbol": tr.symbol, "side": tr.side,
                    "price": float(tr.price), "amount": float(tr.amount),
                    "fee_rate": float(tr.fee_rate), "fee_paid": float(tr.fee_paid_quote),
                    "cash_delta": float(tr.cash_delta_quote),
                    "pnl": pnl,
                    "reason": tr.reason,
                })

        # Time-stop per cycle (optional)
        # ----------------------------
        if bool(getattr(self, "enable_time_stop", False)) and float(getattr(self, "time_stop_hours", 0.0)) > 0.0:
            max_h = float(getattr(self, "time_stop_hours", 0.0))
            mode = str(getattr(self, "time_stop_mode", "BREAK_EVEN_NET")).upper()
            floor_tp = float(getattr(self, "time_stop_floor_tp_pct", 0.0))

            for buy_level, oc in list(self.open_cycles.items()):
                age_h = self._age_hours(ts, oc.buy_time)
                if age_h < max_h:
                    continue

                if mode == "DECAY_TO_TP":
                    base_tp = float(getattr(self, "cycle_tp_pct", 0.0))
                    frac = min(1.0, max(0.0, age_h / max_h))
                    eff_tp = max(floor_tp, base_tp * (1.0 - frac))
                    target_price = float(oc.buy_price) * (1.0 + eff_tp / 100.0)
                elif mode == "REDUCE_TO_TP":
                    target_price = float(oc.buy_price) * (1.0 + floor_tp / 100.0)
                else:
                    target_price = float(oc.buy_price)

                be_limit = self._net_breakeven_limit(trader, oc)
                exit_price = max(float(target_price), float(be_limit))

                if price >= exit_price:
                    # Bar-level guard: avoid repeated TIME_STOP sells for same cycle within the same bar.
                    if bool(getattr(self, 'enable_bar_fill_guard', True)):
                        if float(buy_level) in getattr(self, '_guard_sells_done', set()):
                            continue

                    tr = trader.sell(self.symbol, float(exit_price), float(oc.amount), ts, reason="TIME_STOP")
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

                    if bool(getattr(self, 'enable_bar_fill_guard', True)):
                        self._guard_sells_done.add(float(buy_level))


                    self.trades.append({
                        "time": tr.time, "symbol": tr.symbol, "side": tr.side,
                        "price": float(tr.price), "amount": float(tr.amount),
                        "fee_rate": float(tr.fee_rate), "fee_paid": float(tr.fee_paid_quote),
                        "cash_delta": float(tr.cash_delta_quote),
                        "pnl": pnl,
                        "reason": tr.reason,
                    })
