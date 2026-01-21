from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class DCAConfig:
    # ladder
    step_pct: float = 1.0
    max_safety_orders: int = 8
    base_order_quote: float = 6.0
    safety_order_quote: float = 6.0
    volume_scale: float = 1.0  # martingale: safety_n_quote = safety_quote * volume_scale**(n-1)

    # sizing / allocation
    sizing_mode: str = "fixed"  # fixed | scaled | budget
    budget_quote: float = 200.0
    reserve_funds: bool = True  # show + enforce budget cap

    # exits
    tp_pct: float = 1.7
    enable_stop: bool = False
    stop_pct: float = 20.0
    sell_all_on_stop: bool = True
    sell_all_on_manual_stop: bool = False  # when user hits STOP button

    # execution assumptions
    fee_mode: str = "maker"  # maker|taker
    maker_fee: float = 0.0010
    taker_fee: float = 0.0025

    # slippage assumptions (fractions)
    slippage_buy: float = 0.0001
    slippage_tp: float = 0.0
    slippage_stop: float = 0.0005

    # lifecycle
    auto_restart: bool = True

    # live-ticker behaviour
    max_one_safety_per_tick: bool = True


@dataclass
class Fill:
    time: pd.Timestamp
    side: str  # BUY/SELL
    price: float
    amount_base: float
    fee_paid_quote: float
    fee_rate: float
    cash_delta_quote: float
    reason: str


@dataclass
class CycleState:
    cycle_id: int = 1
    active: bool = False
    entry_price0: Optional[float] = None  # initial reference for ladder
    tp_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_safety: int = 0


class DCAPaperAccount:
    def __init__(self, cash_quote: float = 200.0, quote_ccy: str = "EUR"):
        self.cash_quote = float(cash_quote)
        self.quote_ccy = quote_ccy
        self.pos_base: float = 0.0
        self.cost_quote: float = 0.0  # total cost basis in quote (excl. fees)
        self.realized_pnl_quote: float = 0.0
        self.fills: List[Fill] = []
        self.spent_total_quote: float = 0.0  # incl fees (for budget accounting)

    def avg_entry(self) -> Optional[float]:
        if self.pos_base <= 1e-12:
            return None
        return self.cost_quote / self.pos_base

    def equity(self, mark_price: float) -> float:
        return self.cash_quote + self.pos_base * float(mark_price)

    def _fee_rate(self, cfg: DCAConfig) -> float:
        return cfg.maker_fee if cfg.fee_mode == "maker" else cfg.taker_fee

    def _can_spend(self, cfg: DCAConfig, spend_incl_fee: float) -> bool:
        if not cfg.reserve_funds:
            return spend_incl_fee <= self.cash_quote + 1e-9
        # budget cap is "allocated capital for this bot"
        if (self.spent_total_quote + spend_incl_fee) > (cfg.budget_quote + 1e-9):
            return False
        return spend_incl_fee <= self.cash_quote + 1e-9

    def buy_quote(
        self,
        t: pd.Timestamp,
        price: float,
        quote_amount: float,
        cfg: DCAConfig,
        reason: str,
        *,
        slippage_override: Optional[float] = None,
    ) -> bool:
        if quote_amount <= 0:
            return False
        sl = cfg.slippage_buy if slippage_override is None else float(slippage_override)
        price_eff = float(price) * (1.0 + sl)
        fee_rate = self._fee_rate(cfg)
        fee = quote_amount * fee_rate
        total = quote_amount + fee
        if not self._can_spend(cfg, total):
            return False
        base = quote_amount / price_eff
        self.cash_quote -= total
        self.pos_base += base
        self.cost_quote += quote_amount
        self.spent_total_quote += total
        self.fills.append(Fill(t, "BUY", price_eff, base, fee, fee_rate, -total, reason))
        return True

    def sell_all(
        self,
        t: pd.Timestamp,
        price: float,
        cfg: DCAConfig,
        reason: str,
        *,
        slippage_override: Optional[float] = None,
    ) -> Optional[float]:
        if self.pos_base <= 1e-12:
            return None
        sl = cfg.slippage_stop if slippage_override is None else float(slippage_override)
        price_eff = float(price) * (1.0 - sl)
        proceeds = self.pos_base * price_eff
        fee_rate = self._fee_rate(cfg)
        fee = proceeds * fee_rate
        net = proceeds - fee
        pnl = net - self.cost_quote
        self.cash_quote += net
        self.realized_pnl_quote += pnl
        self.fills.append(Fill(t, "SELL", price_eff, self.pos_base, fee, fee_rate, net, reason))
        self.pos_base = 0.0
        self.cost_quote = 0.0
        return pnl


def compute_safety_quote_for_budget(base_quote: float, max_safety: int, vol_scale: float, budget: float) -> float:
    """Return safety_order_quote such that base + sum(safety*vol_scale^(n-1)) ~= budget."""
    if max_safety <= 0:
        return 0.0
    remain = max(0.0, float(budget) - float(base_quote))
    denom = 0.0
    for n in range(1, max_safety + 1):
        denom += (float(vol_scale) ** (n - 1))
    if denom <= 0:
        return 0.0
    return remain / denom


class DCAStrategy:
    def __init__(self, symbol: str, cfg: DCAConfig, account: DCAPaperAccount):
        self.symbol = symbol
        self.cfg = cfg
        self.acct = account
        self.state = CycleState()
        self.completed_cycles: int = 0
        self.cycle_pnls: List[float] = []

    def planned_total_cost_excl_fee(self) -> float:
        total = float(self.cfg.base_order_quote)
        for n in range(1, int(self.cfg.max_safety_orders) + 1):
            q = float(self.cfg.safety_order_quote) * (float(self.cfg.volume_scale) ** (n - 1))
            total += q
        return total

    def planned_total_cost_incl_fee(self) -> float:
        fee_rate = self.cfg.maker_fee if self.cfg.fee_mode == "maker" else self.cfg.taker_fee
        return self.planned_total_cost_excl_fee() * (1.0 + fee_rate)

    def funds_reserved(self) -> float:
        # Show an OKX-like "reserved" number: remaining planned (incl fee approximation)
        if not self.cfg.reserve_funds:
            return 0.0
        planned = self.planned_total_cost_incl_fee()
        used = min(self.acct.spent_total_quote, float(self.cfg.budget_quote))
        remaining = max(0.0, planned - used)
        return remaining

    def ladder_levels(self) -> List[float]:
        if self.state.entry_price0 is None:
            return []
        levels = []
        for n in range(1, int(self.cfg.max_safety_orders) + 1):
            lvl = float(self.state.entry_price0) * (1.0 - (float(self.cfg.step_pct) / 100.0) * n)
            levels.append(lvl)
        return levels

    def next_safety_level(self) -> Optional[float]:
        levels = self.ladder_levels()
        if not levels:
            return None
        idx = int(self.state.filled_safety)
        if idx >= len(levels):
            return None
        return float(levels[idx])

    def _update_targets(self):
        avg = self.acct.avg_entry()
        if avg is None:
            self.state.tp_price = None
            self.state.stop_price = None
            return
        self.state.tp_price = float(avg) * (1.0 + float(self.cfg.tp_pct) / 100.0)
        if bool(self.cfg.enable_stop):
            self.state.stop_price = float(avg) * (1.0 - float(self.cfg.stop_pct) / 100.0)
        else:
            self.state.stop_price = None

    def start_cycle_if_needed(self, t: pd.Timestamp, current_price: float) -> None:
        if self.state.active:
            return
        ok = self.acct.buy_quote(t, current_price, float(self.cfg.base_order_quote), self.cfg, reason="BASE_ORDER")
        if not ok:
            self.state.active = False
            return
        self.state.active = True
        self.state.entry_price0 = float(current_price)
        self.state.filled_safety = 0
        self._update_targets()

    def _try_fill_safety(self, t: pd.Timestamp, trigger_price: float) -> bool:
        levels = self.ladder_levels()
        if not levels:
            return False
        if self.state.filled_safety >= int(self.cfg.max_safety_orders):
            return False
        lvl = float(levels[int(self.state.filled_safety)])
        if float(trigger_price) > lvl:
            return False
        n = int(self.state.filled_safety) + 1
        quote_amt = float(self.cfg.safety_order_quote) * (float(self.cfg.volume_scale) ** (n - 1))
        ok = self.acct.buy_quote(t, lvl, quote_amt, self.cfg, reason=f"SAFETY_{n}")
        if ok:
            self.state.filled_safety += 1
            self._update_targets()
        return ok

    def on_bar(self, bar: Dict, *, allow_new_cycle: bool = True) -> None:
        # bar dict: timestamp, open, high, low, close
        t = pd.Timestamp(bar["timestamp"])
        h = float(bar["high"]); l = float(bar["low"]); c = float(bar["close"])

        if allow_new_cycle and (not self.state.active) and bool(self.cfg.auto_restart):
            self.start_cycle_if_needed(t, c)

        if not self.state.active:
            return

        self._update_targets()

        # Priority (conservative): Stop -> TP -> Safety
        if bool(self.cfg.enable_stop) and self.state.stop_price is not None and l <= float(self.state.stop_price):
            pnl = self.acct.sell_all(t, float(self.state.stop_price), self.cfg, reason="STOP_LOSS", slippage_override=self.cfg.slippage_stop)
            self._end_cycle(pnl if pnl is not None else 0.0)
            return

        if self.state.tp_price is not None and h >= float(self.state.tp_price):
            pnl = self.acct.sell_all(t, float(self.state.tp_price), self.cfg, reason="TAKE_PROFIT", slippage_override=self.cfg.slippage_tp)
            self._end_cycle(pnl if pnl is not None else 0.0)
            return

        # Safety fills: allow multiple intrabar fills if candle wicks far below multiple levels.
        while self.state.filled_safety < int(self.cfg.max_safety_orders):
            lvl = self.next_safety_level()
            if lvl is None:
                break
            if l <= float(lvl):
                ok = self._try_fill_safety(t, float(lvl))
                if ok:
                    continue
            break

    def on_tick(self, t: pd.Timestamp, price: float, *, allow_new_cycle: bool = True) -> None:
        """Live-ticker (paper) mode: evaluate fills on a single price sample."""
        px = float(price)

        if allow_new_cycle and (not self.state.active) and bool(self.cfg.auto_restart):
            self.start_cycle_if_needed(t, px)

        if not self.state.active:
            return

        self._update_targets()

        # Stop -> TP -> Safety
        if bool(self.cfg.enable_stop) and self.state.stop_price is not None and px <= float(self.state.stop_price):
            pnl = self.acct.sell_all(t, float(self.state.stop_price), self.cfg, reason="STOP_LOSS", slippage_override=self.cfg.slippage_stop)
            self._end_cycle(pnl if pnl is not None else 0.0)
            return

        if self.state.tp_price is not None and px >= float(self.state.tp_price):
            pnl = self.acct.sell_all(t, float(self.state.tp_price), self.cfg, reason="TAKE_PROFIT", slippage_override=self.cfg.slippage_tp)
            self._end_cycle(pnl if pnl is not None else 0.0)
            return

        # safety: optionally only one per tick (to avoid "instant ladder fill" on a single print)
        filled = False
        while self.state.filled_safety < int(self.cfg.max_safety_orders):
            lvl = self.next_safety_level()
            if lvl is None:
                break
            if px <= float(lvl):
                ok = self._try_fill_safety(t, px)
                if ok:
                    filled = True
                    if bool(self.cfg.max_one_safety_per_tick):
                        break
                    continue
            break

    def manual_stop(self, t: pd.Timestamp, mark_price: float) -> None:
        """User STOP button. Optional 'sell all' behavior."""
        if bool(self.cfg.sell_all_on_manual_stop) and self.acct.pos_base > 1e-12:
            pnl = self.acct.sell_all(t, float(mark_price), self.cfg, reason="MANUAL_STOP_SELL", slippage_override=self.cfg.slippage_stop)
            self._end_cycle(pnl if pnl is not None else 0.0)

    def _end_cycle(self, pnl: float) -> None:
        self.completed_cycles += 1
        self.cycle_pnls.append(float(pnl))
        self.state = CycleState(cycle_id=self.state.cycle_id + 1)

    def status_snapshot(self, mark_price: float) -> Dict:
        avg = self.acct.avg_entry()
        eq = self.acct.equity(mark_price)
        invested = self.acct.cost_quote
        unreal = (self.acct.pos_base * mark_price - invested) if (avg is not None) else 0.0

        fee_rate = self.cfg.maker_fee if self.cfg.fee_mode == "maker" else self.cfg.taker_fee
        planned_total_incl_fee = self.planned_total_cost_incl_fee()
        used = self.acct.spent_total_quote
        budget = float(self.cfg.budget_quote) if self.cfg.reserve_funds else float("nan")
        budget_left = (budget - used) if self.cfg.reserve_funds else float("nan")

        return {
            "symbol": self.symbol,
            "cycle": self.state.cycle_id,
            "active": self.state.active,
            "filled_safety": self.state.filled_safety,
            "max_safety": self.cfg.max_safety_orders,
            "pos_base": self.acct.pos_base,
            "avg_entry": avg,
            "tp_price": self.state.tp_price,
            "stop_price": self.state.stop_price,
            "next_safety": self.next_safety_level(),
            "cash": self.acct.cash_quote,
            "equity": eq,
            "realized_pnl": self.acct.realized_pnl_quote,
            "unrealized_pnl": unreal,
            "invested": invested,
            "completed_cycles": self.completed_cycles,
            "fee_rate": fee_rate,
            "planned_total_incl_fee": planned_total_incl_fee,
            "spent_total": used,
            "budget_quote": budget,
            "budget_left": budget_left,
            "funds_reserved": self.funds_reserved(),
        }
