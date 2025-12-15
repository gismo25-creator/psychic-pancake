class GridEngine:
    def __init__(self, grid_levels, order_size):
        self.grid = sorted(grid_levels)
        self.order_size = order_size
        self.active_buys = set(self.grid[:-1])
        self.active_sells = set()
        self.open_positions = {}
        self.closed_grids = []
        self.trades = []

    def check_price(self, price, trader, symbol):
        for buy in list(self.active_buys):
            if price <= buy:
                exec_price = trader.buy(symbol, buy, self.order_size)
                if exec_price is None:
                    continue
                self.active_buys.remove(buy)
                sell = self._next(buy)
                self.active_sells.add(sell)
                self.open_positions[buy] = {"buy": exec_price, "amount": self.order_size}
                self.trades.append({"side":"BUY","price":exec_price})

        for sell in list(self.active_sells):
            if price >= sell:
                buy_level = self._prev(sell)
                pos = self.open_positions.pop(buy_level, None)
                if pos is None:
                    continue
                exec_price = trader.sell(symbol, sell, self.order_size)
                if exec_price is None:
                    continue
                pnl = (exec_price - pos["buy"]) * pos["amount"]
                self.closed_grids.append({"buy": pos["buy"], "sell": exec_price, "pnl": pnl})
                self.active_sells.remove(sell)
                self.active_buys.add(buy_level)
                self.trades.append({"side":"SELL","price":exec_price})

    def _next(self, level):
        return self.grid[self.grid.index(level) + 1]

    def _prev(self, level):
        return self.grid[self.grid.index(level) - 1]
