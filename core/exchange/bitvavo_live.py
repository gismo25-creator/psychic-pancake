"""
Bitvavo Live Trader (v1)

This module provides a *minimal* live executor compatible with GridEngine:
- buy(symbol, price, amount, ts, reason) -> TradeResult | None
- sell(symbol, price, amount, ts, reason) -> TradeResult | None
- equity(last_prices) -> float (derived from balances + mark prices)
- close_all(last_prices, ts, reason) -> closes inventory using market sells (optional gating in UI)

Important design choice (v1):
- Uses MARKET orders so fills are immediate and GridEngine can keep its synchronous assumptions.
- Limit-order lifecycle management (open orders, cancels, partial fills) is intentionally out-of-scope here.

Rate-limit handling:
- Retries transient errors (HTTP 429/503) with exponential backoff + jitter.
- If Bitvavo blocks the API key or IP (errorCode 105), the trader enters a "banned" state until expiry.

API credentials:
- Supply via Streamlit secrets: BITVAVO_API_KEY, BITVAVO_API_SECRET (and optionally BITVAVO_ACCESS_WINDOW_MS).
"""
from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import ccxt  # already used in this project

from .simulator import TradeResult


class BitvavoRateLimitBanned(RuntimeError):
    def __init__(self, ban_until_ms: int, message: str):
        super().__init__(message)
        self.ban_until_ms = int(ban_until_ms)


def _split_symbol(symbol: str) -> Tuple[str, str]:
    base, quote = symbol.replace("-", "/").split("/")
    return base.upper(), quote.upper()


def _to_market(symbol: str) -> str:
    return symbol.replace("/", "-").upper()


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class _CostBasis:
    qty: float = 0.0
    avg: float = 0.0


class BitvavoLiveTrader:
    """
    Minimal synchronous live executor for Bitvavo.

    The class maintains a light internal ledger for:
    - positions[base] : base quantity (synced from balance periodically)
    - cash : quote currency available (synced from balance periodically)
    - avg_entry_price(base) : weighted average of fills executed by this bot (best-effort)

    If you start with existing inventory, avg_entry_price will be "unknown" until the bot buys.
    """
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        maker_fee: float = 0.0010,
        taker_fee: float = 0.0025,
        slippage: float = 0.0,
        fee_mode: str = "taker",
        quote_ccy: str = "EUR",
        max_exposure_quote: Optional[Dict[str, float]] = None,
        max_order_quote: float = 0.0,
        sandbox: bool = False,
        enable_market_orders: bool = True,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.quote_ccy = quote_ccy.upper()
        self.maker_fee = float(maker_fee)
        self.taker_fee = float(taker_fee)
        self.slippage = float(slippage)
        self.fee_mode = str(fee_mode)
        self.max_exposure_quote = max_exposure_quote or {}
        self.max_order_quote = float(max_order_quote or 0.0)

        self.enable_market_orders = bool(enable_market_orders)
        self.max_retries = int(max_retries)

        self.exchange = ccxt.bitvavo({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        # Note: Bitvavo doesn't provide a public "sandbox" in the same way as some exchanges.
        # This flag is kept for future extensibility.
        self.sandbox = bool(sandbox)

        self.positions: Dict[str, float] = {}  # base -> amount
        self.cash: float = 0.0  # quote currency
        self._basis: Dict[str, _CostBasis] = {}  # base -> cost basis
        self._balance_cache = None
        self._balance_cache_ts = 0.0
        self._balance_ttl_s = 5.0

        self.ban_until_ms: int = 0
        self.trades = []  # optional list of TradeResult-like records for UI parity

        # initial sync
        self._sync_balances()

    def fee_rate(self) -> float:
        return self.maker_fee if self.fee_mode == "maker" else self.taker_fee

    def _is_banned(self) -> bool:
        return self.ban_until_ms > _now_ms()

    def _parse_error_code(self, err: Exception) -> Tuple[Optional[int], Optional[int], str]:
        """
        Try to extract Bitvavo errorCode and ban expiryInMs from exception message.
        """
        msg = str(err)
        code = None
        expiry = None
        # ccxt sometimes embeds JSON
        try:
            j = json.loads(msg[msg.find("{"):]) if "{" in msg and "}" in msg else None
            if isinstance(j, dict):
                if "errorCode" in j:
                    code = int(j.get("errorCode"))
                # Support/Help center uses expiryInMs / ban expires at ...
                for k in ["expiryInMs", "banUntil", "banExpiresAt", "expiresAt", "expiry"]:
                    if k in j:
                        expiry = int(float(j[k]))
        except Exception:
            pass
        # fallback: look for '"errorCode":105'
        if code is None and "errorCode" in msg:
            import re
            m = re.search(r"errorCode[\"']?\s*:\s*(\d+)", msg)
            if m:
                code = int(m.group(1))
        # expiry can appear as a raw int
        if expiry is None:
            import re
            m = re.search(r"ban expires at\s+(\d+)", msg, flags=re.IGNORECASE)
            if m:
                expiry = int(m.group(1))
        return code, expiry, msg

    def _with_backoff(self, fn, *args, **kwargs):
        if self._is_banned():
            raise BitvavoRateLimitBanned(self.ban_until_ms, f"Bitvavo rate-limit ban active until {self.ban_until_ms} ms")

        base = 0.35
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                code, expiry, msg = self._parse_error_code(e)

                # Hard ban (429 errorCode 105) — store expiry and stop retrying
                if code == 105 and expiry:
                    self.ban_until_ms = int(expiry)
                    raise BitvavoRateLimitBanned(self.ban_until_ms, msg)

                # 107/108/111/503: recommended retry after ~500ms
                retryable = False
                if code in (107, 108, 111, 112):
                    retryable = True
                    delay = 0.5
                else:
                    # Network / rate-limit transient
                    # ccxt uses NetworkError / ExchangeNotAvailable / DDoSProtection, etc.
                    cls = e.__class__.__name__.lower()
                    if "network" in cls or "timeout" in cls or "temporar" in cls or "overload" in cls:
                        retryable = True
                        delay = base * (2 ** attempt)

                # Bitvavo rate limit exceeded can also be 429 with other codes; treat as retryable with reset window
                if code in (110,):  # per docs: 110 for exceeding limit (authenticated block 1 min)
                    retryable = True
                    delay = max(1.0, base * (2 ** attempt))

                # Uncertain outcome: do not blindly retry market order placement unless we can reconcile by clientOrderId
                if code == 109:
                    raise

                if (not retryable) or attempt >= self.max_retries:
                    raise

                # jitter
                delay = float(delay) + random.random() * 0.25
                time.sleep(min(delay, 3.0))

    def _fetch_balance_cached(self):
        now = time.time()
        if self._balance_cache is not None and (now - self._balance_cache_ts) <= self._balance_ttl_s:
            return self._balance_cache
        bal = self._with_backoff(self.exchange.fetch_balance)
        self._balance_cache = bal
        self._balance_cache_ts = now
        return bal

    def _sync_balances(self):
        bal = self._fetch_balance_cached()
        free = bal.get("free", {}) or {}
        # quote cash
        self.cash = float(free.get(self.quote_ccy, 0.0) or 0.0)
        # positions: include non-zero assets
        for asset, amt in free.items():
            try:
                a = float(amt or 0.0)
            except Exception:
                continue
            if a <= 0:
                continue
            self.positions[asset.upper()] = a

    def avg_entry_price(self, base: str) -> Optional[float]:
        b = base.upper()
        cb = self._basis.get(b)
        if cb and cb.qty > 1e-12:
            return float(cb.avg)
        return None

    def _exposure_ok(self, base: str, last_prices: Dict[str, float], buy_cost_quote: float) -> bool:
        cap = float(self.max_exposure_quote.get(base.upper(), 0.0) or 0.0)
        if cap <= 0:
            return True


    def _order_notional_ok(self, price: float, amount: float) -> bool:
        cap = float(self.max_order_quote or 0.0)
        if cap <= 0:
            return True
        return (float(price) * float(amount)) <= cap + 1e-9
        # exposure approx: current position valued at mark + new buy cost
        # last_prices map is symbol->price
        px = None
        for sym, p in last_prices.items():
            if sym.split("/")[0].upper() == base.upper():
                px = float(p)
                break
        pos = float(self.positions.get(base.upper(), 0.0) or 0.0)
        exposure = (pos * px) if (px is not None) else 0.0
        return (exposure + float(buy_cost_quote)) <= cap + 1e-9

    def _record_blocked(self, side: str, symbol: str, price: float, amount: float, ts, reason: str):
        # parity with simulator UI: create a TradeResult with zero deltas
        base, quote = _split_symbol(symbol)
        tr = TradeResult(
            side=side, symbol=symbol, base=base, quote=quote,
            time=ts, price=float(price), amount=float(amount),
            fee_rate=float(self.fee_rate()),
            fee_paid_quote=0.0, cash_delta_quote=0.0, pos_delta_base=0.0,
        )
        # attach reason attribute for UI (best-effort)
        setattr(tr, "reason", reason)
        self.trades.append(tr)

    def record_blocked(self, side: str, symbol: str, price: float, amount: float, ts, reason: str):
        self._record_blocked(side, symbol, price, amount, ts, reason)

    def equity(self, last_prices: Dict[str, float]) -> float:
        # refresh balances
        self._sync_balances()
        eq = float(self.cash)
        for sym, px in last_prices.items():
            base = sym.split("/")[0].upper()
            pos = float(self.positions.get(base, 0.0) or 0.0)
            if pos > 0:
                eq += pos * float(px)
        return float(eq)

    def buy(self, symbol: str, price: float, amount: float, ts, reason: str = "LIVE") -> Optional[TradeResult]:
        if not self.enable_market_orders:
            self._record_blocked("BUY", symbol, price, amount, ts, "LIVE_DISABLED")
            return None

        self._sync_balances()
        base, quote = _split_symbol(symbol)
        # conservative: include slippage cushion
        est_px = float(price) * (1.0 + float(self.slippage))
        est_cost = est_px * float(amount)
        if self.cash + 1e-9 < est_cost:
            self._record_blocked("BUY", symbol, price, amount, ts, "INSUFFICIENT_QUOTE_BALANCE")
            return None

        # per-order notional cap
        if not self._order_notional_ok(est_px, float(amount)):
            self._record_blocked("BUY", symbol, price, amount, ts, "LIVE_ORDER_CAP")
            return None

        # exposure limit
        if not self._exposure_ok(base, {symbol: float(price)}, est_cost):
            self._record_blocked("BUY", symbol, price, amount, ts, "MAX_EXPOSURE_CAP_REACHED")
            return None

        # per-order notional cap
        if not self._order_notional_ok(float(price), float(amount)):
            self._record_blocked("SELL", symbol, price, amount, ts, "LIVE_ORDER_CAP")
            return None

        market = _to_market(symbol)
        # Place MARKET order
        resp = self._with_backoff(self.exchange.create_market_buy_order, market, float(amount))
        # Try to infer average fill price
        avg = float(resp.get("average") or resp.get("price") or est_px)
        fee = resp.get("fee") or {}
        fee_cost = float(fee.get("cost") or 0.0)
        fee_ccy = str(fee.get("currency") or quote)
        fee_paid_quote = fee_cost if fee_ccy.upper() == quote.upper() else 0.0

        cash_delta = -(avg * float(amount) + fee_paid_quote)
        pos_delta = float(amount)

        # update cost basis (best-effort)
        cb = self._basis.get(base, _CostBasis())
        new_qty = cb.qty + pos_delta
        new_avg = ((cb.avg * cb.qty) + (avg * pos_delta)) / new_qty if new_qty > 1e-12 else avg
        self._basis[base] = _CostBasis(qty=new_qty, avg=new_avg)

        # refresh balances after fill
        self._balance_cache = None
        self._sync_balances()

        tr = TradeResult(
            side="BUY", symbol=symbol, base=base, quote=quote,
            time=ts, price=float(avg), amount=float(amount),
            fee_rate=float(self.fee_rate()),
            fee_paid_quote=float(fee_paid_quote),
            cash_delta_quote=float(cash_delta),
            pos_delta_base=float(pos_delta),
        )
        setattr(tr, "reason", reason)
        self.trades.append(tr)
        return tr

    def sell(self, symbol: str, price: float, amount: float, ts, reason: str = "LIVE") -> Optional[TradeResult]:
        if not self.enable_market_orders:
            self._record_blocked("SELL", symbol, price, amount, ts, "LIVE_DISABLED")
            return None

        self._sync_balances()
        base, quote = _split_symbol(symbol)
        pos = float(self.positions.get(base, 0.0) or 0.0)
        if pos + 1e-12 < float(amount):
            self._record_blocked("SELL", symbol, price, amount, ts, "INSUFFICIENT_BASE_BALANCE")
            return None

        # per-order notional cap
        if not self._order_notional_ok(float(price), float(amount)):
            self._record_blocked("SELL", symbol, price, amount, ts, "LIVE_ORDER_CAP")
            return None

        market = _to_market(symbol)
        resp = self._with_backoff(self.exchange.create_market_sell_order, market, float(amount))
        avg = float(resp.get("average") or resp.get("price") or float(price) * (1.0 - float(self.slippage)))
        fee = resp.get("fee") or {}
        fee_cost = float(fee.get("cost") or 0.0)
        fee_ccy = str(fee.get("currency") or quote)
        fee_paid_quote = fee_cost if fee_ccy.upper() == quote.upper() else 0.0

        cash_delta = (avg * float(amount) - fee_paid_quote)
        pos_delta = -float(amount)

        # update basis: reduce qty; keep avg
        cb = self._basis.get(base)
        if cb:
            cb.qty = max(0.0, cb.qty + pos_delta)

        # refresh balances
        self._balance_cache = None
        self._sync_balances()

        tr = TradeResult(
            side="SELL", symbol=symbol, base=base, quote=quote,
            time=ts, price=float(avg), amount=float(amount),
            fee_rate=float(self.fee_rate()),
            fee_paid_quote=float(fee_paid_quote),
            cash_delta_quote=float(cash_delta),
            pos_delta_base=float(pos_delta),
        )
        setattr(tr, "reason", reason)
        self.trades.append(tr)
        return tr

    def close_all(self, last_prices: Dict[str, float], ts, reason: str = "PANIC_FLATTEN"):
        """
        Close all non-quote balances using market sells for markets in last_prices.
        This is used by STOP & FLATTEN. It is intentionally conservative (no buys).
        """
        self._sync_balances()
        for sym, px in last_prices.items():
            base = sym.split("/")[0].upper()
            amt = float(self.positions.get(base, 0.0) or 0.0)
            if amt <= 1e-12:
                continue
            try:
                self.sell(sym, float(px), amt, ts, reason=reason)
            except Exception:
                # In panic mode, best-effort; continue
                continue
