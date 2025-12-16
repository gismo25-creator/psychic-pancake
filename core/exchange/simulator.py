from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class TradeResult:
    side: str              # BUY / SELL
    symbol: str
    base: str
    quote: str
    time: object           # pandas.Timestamp
    price: float           # executed price
    amount: float          # base amount
    fee_rate: float
    fee_paid_quote: float
    cash_delta_quote: float
    pos_delta_base: float
    reason: str = "OK"     # OK / RISK_LIMIT / STOPLOSS / etc.

class PortfolioSimulatorTrader:
    '''
    Portfolio-level simulator (single quote currency cash ledger, EUR by default).
    Adds:
    - Risk-limit: max exposure per base asset in quote (EUR).
    - Cost basis tracking (average cost) to enable per-asset stop logic & reporting.
    '''
    def __init__(
        self,
        cash_quote: float = 1000.0,
        maker_fee: float = 0.0015,
        taker_fee: float = 0.0025,
        slippage: float = 0.0005,
        fee_mode: str = "taker",
        quote_ccy: str = "EUR",
        max_exposure_quote: Optional[Dict[str, float]] = None,
    ):
        self.cash: float = cash_quote
        self.positions: Dict[str, float] = {}         # base -> amount
        self.cost_basis: Dict[str, float] = {}        # base -> total EUR cost basis (average cost accounting)
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.fee_mode = fee_mode
        self.quote_ccy = quote_ccy
        self.max_exposure_quote = max_exposure_quote or {}
        self.trades: list[TradeResult] = []

    def fee_rate(self) -> float:
        return self.maker_fee if self.fee_mode == "maker" else self.taker_fee

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = price * self.slippage
        return price + slip if side == "BUY" else price - slip

    def _split_symbol(self, symbol: str) -> Tuple[str, str]:
        base, quote = symbol.split("/")
        return base, quote

    def avg_entry_price(self, base: str) -> Optional[float]:
        amt = self.positions.get(base, 0.0)
        if amt <= 0:
            return None
        cb = self.cost_basis.get(base, 0.0)
        return cb / amt if cb > 0 else None

    def _exposure_ok(self, base: str, new_amount: float, mark_price: float) -> Tuple[bool, str]:
        cap = self.max_exposure_quote.get(base)
        if cap is None:
            return True, "OK"
        current_amt = self.positions.get(base, 0.0)
        new_exposure = (current_amt + new_amount) * mark_price
        if new_exposure > cap + 1e-12:
            return False, f"RISK_LIMIT: exposure {new_exposure:.2f} > cap {cap:.2f} {self.quote_ccy}"
        return True, "OK"

    def buy(self, symbol: str, limit_price: float, amount_base: float, ts, reason: str = "OK") -> Optional[TradeResult]:
        base, quote = self._split_symbol(symbol)
        if quote != self.quote_ccy:
            raise ValueError(f"Quote currency {quote} not supported; expected {self.quote_ccy}")

        price = self._apply_slippage(limit_price, "BUY")

        ok, why = self._exposure_ok(base, amount_base, price)
        if not ok:
            self.trades.append(TradeResult("BUY", symbol, base, quote, ts, price, amount_base, self.fee_rate(), 0.0, 0.0, 0.0, why))
            return None

        notional = price * amount_base
        fee_rate = self.fee_rate()
        fee_paid = notional * fee_rate
        cash_out = notional + fee_paid

        if self.cash < cash_out:
            self.trades.append(TradeResult("BUY", symbol, base, quote, ts, price, amount_base, fee_rate, fee_paid, 0.0, 0.0, "INSUFFICIENT_CASH"))
            return None

        self.cash -= cash_out
        self.positions[base] = self.positions.get(base, 0.0) + amount_base
        # average-cost basis: add total cash out (incl fee)
        self.cost_basis[base] = self.cost_basis.get(base, 0.0) + cash_out

        tr = TradeResult("BUY", symbol, base, quote, ts, price, amount_base, fee_rate, fee_paid, -cash_out, amount_base, reason)
        self.trades.append(tr)
        return tr

    def sell(self, symbol: str, limit_price: float, amount_base: float, ts, reason: str = "OK") -> Optional[TradeResult]:
        base, quote = self._split_symbol(symbol)
        if quote != self.quote_ccy:
            raise ValueError(f"Quote currency {quote} not supported; expected {self.quote_ccy}")

        pos = self.positions.get(base, 0.0)
        if pos < amount_base - 1e-12:
            self.trades.append(TradeResult("SELL", symbol, base, quote, ts, limit_price, amount_base, self.fee_rate(), 0.0, 0.0, 0.0, "INSUFFICIENT_POSITION"))
            return None

        price = self._apply_slippage(limit_price, "SELL")
        notional = price * amount_base
        fee_rate = self.fee_rate()
        fee_paid = notional * fee_rate
        cash_in = notional - fee_paid

        self.cash += cash_in
        self.positions[base] = pos - amount_base

        # reduce cost basis using average-cost accounting
        cb = self.cost_basis.get(base, 0.0)
        if pos > 0:
            avg_cost = cb / pos
            self.cost_basis[base] = max(0.0, cb - avg_cost * amount_base)
        else:
            self.cost_basis[base] = 0.0

        tr = TradeResult("SELL", symbol, base, quote, ts, price, amount_base, fee_rate, fee_paid, cash_in, -amount_base, reason)
        self.trades.append(tr)
        return tr

    def close_all(self, mark_prices: Dict[str, float], ts, reason: str = "STOPLOSS_PORTFOLIO") -> list[TradeResult]:
        results = []
        for base, amt in list(self.positions.items()):
            if amt <= 1e-12:
                continue
            sym = f"{base}/{self.quote_ccy}"
            px = mark_prices.get(sym)
            if px is None:
                continue
            tr = self.sell(sym, px, amt, ts, reason=reason)
            if tr is not None:
                results.append(tr)
        return results
    def record_blocked(self, side: str, symbol: str, limit_price: float, amount_base: float, ts, reason: str):
        """Record a blocked order intent for UI transparency (no cash/position change)."""
        base, quote = self._split_symbol(symbol)
        tr = TradeResult(
            side=side, symbol=symbol, base=base, quote=quote, time=ts,
            price=float(limit_price), amount=float(amount_base),
            fee_rate=self.fee_rate(), fee_paid_quote=0.0,
            cash_delta_quote=0.0, pos_delta_base=0.0,
            reason=reason
        )
        self.trades.append(tr)

    def equity(self, mark_prices: Dict[str, float]) -> float:
        eq = float(self.cash)
        for base, amt in self.positions.items():
            if abs(amt) < 1e-12:
                continue
            sym = f"{base}/{self.quote_ccy}"
            px = mark_prices.get(sym)
            if px is None:
                continue
            eq += float(amt) * float(px)
        return float(eq)
