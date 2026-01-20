"""Paper trading executor for the FVG strategy (standalone).

Design goals:
- Simple, interpretable, and auditable.
- Candle-based fills using OHLC (15m recommended).
- Limit entry on FVG midpoint (or edge), SL beyond candle1 extremum, TP at R-multiple.

This is NOT a brokerage/exchange integration.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Literal, Tuple

import math
import pandas as pd


Side = Literal["BUY", "SELL"]


@dataclass
class FVGOrder:
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry: float
    sl: float
    tp: float
    size_base: float
    created_time: pd.Timestamp
    zone_low: float
    zone_high: float
    status: Literal["PENDING", "OPEN", "CLOSED", "CANCELED"] = "PENDING"
    filled_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # ensure timestamps are serializable
        for k in ["created_time", "filled_time", "exit_time"]:
            if d.get(k) is not None:
                d[k] = str(d[k])
        return d


@dataclass
class TradeFill:
    time: pd.Timestamp
    symbol: str
    side: Side
    price: float
    amount_base: float
    fee_paid_quote: float
    fee_rate: float
    cash_delta_quote: float
    reason: str
    pnl_quote: float


class FVGPaperAccount:
    """Single-quote-currency paper account."""

    def __init__(
        self,
        *,
        cash_quote: float = 1000.0,
        maker_fee: float = 0.0010,
        taker_fee: float = 0.0025,
        slippage: float = 0.0001,
        fee_mode_entry: Literal["maker", "taker"] = "maker",
        fee_mode_exit_tp: Literal["maker", "taker"] = "maker",
        fee_mode_exit_sl: Literal["taker", "maker"] = "taker",
        quote_ccy: str = "EUR",
    ):
        self.quote_ccy = quote_ccy
        self.cash = float(cash_quote)
        self.positions: Dict[str, float] = {}  # base -> amount
        self.avg_entry: Dict[str, float] = {}  # base -> avg entry
        self.open_orders: Dict[str, FVGOrder] = {}  # symbol -> order
        self.fills: List[TradeFill] = []

        self.maker_fee = float(maker_fee)
        self.taker_fee = float(taker_fee)
        self.slippage = float(slippage)
        self.fee_mode_entry = fee_mode_entry
        self.fee_mode_exit_tp = fee_mode_exit_tp
        self.fee_mode_exit_sl = fee_mode_exit_sl

    def _fee_rate(self, mode: str) -> float:
        return self.maker_fee if mode == "maker" else self.taker_fee

    def equity(self, last_prices: Dict[str, float]) -> float:
        eq = self.cash
        for base, amt in self.positions.items():
            # assume last_prices keyed by full symbol; best-effort match by base
            # caller should pass symbol-specific price; we use base mapping fallback.
            px = None
            for sym, p in last_prices.items():
                if sym.split("/")[0] == base:
                    px = float(p)
                    break
            if px is None:
                continue
            eq += amt * px
        return float(eq)

    def _apply_buy(self, symbol: str, price: float, amount_base: float, ts: pd.Timestamp, reason: str, fee_mode: str) -> None:
        base = symbol.split("/")[0]
        fee_rate = self._fee_rate(fee_mode)
        # slippage on entry for taker; for maker keep it minimal but still optional
        eff_price = price * (1.0 + (self.slippage if fee_mode == "taker" else 0.0))
        cost = eff_price * amount_base
        fee = cost * fee_rate
        total = cost + fee
        if total > self.cash + 1e-12:
            # insufficient cash
            return

        prev_amt = self.positions.get(base, 0.0)
        prev_avg = self.avg_entry.get(base)
        new_amt = prev_amt + amount_base
        if new_amt <= 1e-12:
            return
        if prev_amt <= 1e-12 or prev_avg is None:
            new_avg = eff_price
        else:
            new_avg = (prev_amt * prev_avg + amount_base * eff_price) / new_amt

        self.positions[base] = new_amt
        self.avg_entry[base] = new_avg
        self.cash -= total

        self.fills.append(
            TradeFill(
                time=ts,
                symbol=symbol,
                side="BUY",
                price=float(eff_price),
                amount_base=float(amount_base),
                fee_paid_quote=float(fee),
                fee_rate=float(fee_rate),
                cash_delta_quote=float(-total),
                reason=reason,
                pnl_quote=0.0,
            )
        )

    def _apply_sell(self, symbol: str, price: float, amount_base: float, ts: pd.Timestamp, reason: str, fee_mode: str) -> float:
        base = symbol.split("/")[0]
        pos = self.positions.get(base, 0.0)
        if amount_base > pos + 1e-12:
            amount_base = pos
        if amount_base <= 1e-12:
            return 0.0

        fee_rate = self._fee_rate(fee_mode)
        eff_price = price * (1.0 - (self.slippage if fee_mode == "taker" else 0.0))
        proceeds = eff_price * amount_base
        fee = proceeds * fee_rate
        net = proceeds - fee

        avg = self.avg_entry.get(base)
        pnl = 0.0
        if avg is not None:
            pnl = (eff_price - avg) * amount_base - fee

        self.cash += net
        new_pos = pos - amount_base
        if new_pos <= 1e-12:
            self.positions.pop(base, None)
            self.avg_entry.pop(base, None)
        else:
            self.positions[base] = new_pos

        self.fills.append(
            TradeFill(
                time=ts,
                symbol=symbol,
                side="SELL",
                price=float(eff_price),
                amount_base=float(amount_base),
                fee_paid_quote=float(fee),
                fee_rate=float(fee_rate),
                cash_delta_quote=float(net),
                reason=reason,
                pnl_quote=float(pnl),
            )
        )
        return float(pnl)

    def place_order(self, order: FVGOrder) -> None:
        self.open_orders[order.symbol] = order

    def cancel_order(self, symbol: str, reason: str = "CANCELED") -> None:
        o = self.open_orders.get(symbol)
        if o is None:
            return
        o.status = "CANCELED"
        o.exit_reason = reason
        self.open_orders.pop(symbol, None)

    def on_candle(self, symbol: str, candle: dict) -> None:
        """Process a single candle (dict with timestamp/open/high/low/close)."""
        o = self.open_orders.get(symbol)
        if o is None:
            return

        ts = pd.Timestamp(candle["timestamp"])
        high = float(candle["high"])
        low = float(candle["low"])

        # Fill pending entry
        if o.status == "PENDING":
            if low <= o.entry <= high:
                if o.direction == "LONG":
                    self._apply_buy(symbol, o.entry, o.size_base, ts, reason="FVG_ENTRY", fee_mode=self.fee_mode_entry)
                    o.status = "OPEN"
                    o.filled_time = ts
                else:
                    # short simulation not implemented in spot-account; keep pending
                    return

        # Manage exits for open long
        if o.status == "OPEN" and o.direction == "LONG":
            # Conservative intrabar ordering: SL first if both hit
            sl_hit = low <= o.sl
            tp_hit = high >= o.tp

            if sl_hit:
                amt = self.positions.get(symbol.split("/")[0], 0.0)
                self._apply_sell(symbol, o.sl, amt, ts, reason="STOP_LOSS", fee_mode=self.fee_mode_exit_sl)
                o.status = "CLOSED"
                o.exit_time = ts
                o.exit_price = float(o.sl)
                o.exit_reason = "STOP_LOSS"
                self.open_orders.pop(symbol, None)
                return

            if tp_hit:
                amt = self.positions.get(symbol.split("/")[0], 0.0)
                self._apply_sell(symbol, o.tp, amt, ts, reason="TAKE_PROFIT", fee_mode=self.fee_mode_exit_tp)
                o.status = "CLOSED"
                o.exit_time = ts
                o.exit_price = float(o.tp)
                o.exit_reason = "TAKE_PROFIT"
                self.open_orders.pop(symbol, None)
                return


def size_from_risk(
    *,
    equity_quote: float,
    risk_pct: float,
    entry: float,
    sl: float,
    max_notional_quote: float,
) -> float:
    """Compute base size so that (entry-sl)*size ~= equity*risk_pct."""
    risk_quote = max(0.0, equity_quote * (risk_pct / 100.0))
    per_unit = abs(entry - sl)
    if per_unit <= 1e-12:
        return 0.0
    size = risk_quote / per_unit

    # clamp by max notional
    notional = size * entry
    if max_notional_quote > 0 and notional > max_notional_quote:
        size = max_notional_quote / entry

    return max(0.0, float(size))


def size_from_fixed_notional(*, notional_quote: float, entry: float) -> float:
    if entry <= 1e-12:
        return 0.0
    return max(0.0, float(notional_quote) / float(entry))
