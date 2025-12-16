from dataclasses import dataclass
from typing import Dict

@dataclass
class OpenCycle:
    cash_out: float   # positive EUR spent (includes buy fee)
    buy_price: float
    amount: float
    buy_time: object

class GridEngine:
    '''
    Exact realized PnL:
    - Buy stores exact cash_out from portfolio ledger
    - Sell closes and realized pnl = cash_in - cash_out (matches portfolio exactly)
    '''
    def __init__(self, symbol: str, grid_levels, order_size):
        self.symbol = symbol
        self.grid = sorted(grid_levels)
        self.order_size = order_size
        self.active_buys = set(self.grid[:-1])
        self.active_sells = set()
        self.open_cycles: Dict[float, OpenCycle] = {}  # buy_level -> OpenCycle
        self.closed_cycles = []
        self.trades = []  # executed trades only

    def reset_open_cycles(self):
        # Used when we forced a flat (stop-loss) so the grid state doesn't refer to stale entries.
        self.open_cycles = {}
        self.active_buys = set(self.grid[:-1])
        self.active_sells = set()
def check_price(self, price: float, trader, ts, allow_buys: bool = True, buy_guard=None):
    # BUY
    if allow_buys:
        for buy in list(self.active_buys):
            if price <= buy:
                if buy_guard is not None:
                    ok, why = buy_guard(self.symbol, self.order_size, buy, ts)
                    if not ok:
                        if hasattr(trader, "record_blocked"):
                            trader.record_blocked("BUY", self.symbol, buy, self.order_size, ts, why)
                        continue

                tr = trader.buy(self.symbol, buy, self.order_size, ts, reason="GRID")
                if tr is None:
                    continue

                self.active_buys.remove(buy)
                sell = self._next(buy)
                self.active_sells.add(sell)

                cash_out = -tr.cash_delta_quote  # positive
                self.open_cycles[buy] = OpenCycle(
                    cash_out=cash_out,
                    buy_price=tr.price,
                    amount=tr.amount,
                    buy_time=tr.time,
                )

                self.trades.append({
                    "time": tr.time, "symbol": tr.symbol, "side": tr.side,
                    "price": tr.price, "amount": tr.amount,
                    "fee_rate": tr.fee_rate, "fee_paid": tr.fee_paid_quote,
                    "cash_delta": tr.cash_delta_quote,
                    "pnl": 0.0,
                    "reason": tr.reason,
                })

    # SELL (always allowed)
    for sell in list(self.active_sells):
        if price >= sell:
            buy_level = self._prev(sell)
            oc = self.open_cycles.pop(buy_level, None)
            if oc is None:
                continue

            tr = trader.sell(self.symbol, sell, self.order_size, ts, reason="GRID")
            if tr is None:
                self.open_cycles[buy_level] = oc
                continue

            cash_in = tr.cash_delta_quote
            pnl = cash_in - oc.cash_out

            self.closed_cycles.append({
                "symbol": tr.symbol,
                "buy_time": oc.buy_time, "sell_time": tr.time,
                "buy_price": oc.buy_price, "sell_price": tr.price,
                "amount": tr.amount,
                "cash_out": oc.cash_out, "cash_in": cash_in,
                "pnl": pnl,
            })

            self.active_sells.remove(sell)
            self.active_buys.add(buy_level)

            self.trades.append({
                "time": tr.time, "symbol": tr.symbol, "side": tr.side,
                "price": tr.price, "amount": tr.amount,
                "fee_rate": tr.fee_rate, "fee_paid": tr.fee_paid_quote,
                "cash_delta": tr.cash_delta_quote,
                "pnl": pnl,
                "reason": tr.reason,
            })

    def _next(self, level):
        return self.grid[self.grid.index(level) + 1]

    def _prev(self, level):
        return self.grid[self.grid.index(level) - 1]
