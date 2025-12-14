class GridEngine:
    def __init__(self, grid_levels, order_size):
        self.grid = sorted(grid_levels)
        self.order_size = order_size
        self.active_buys = set(self.grid[:-1])
        self.active_sells = set()
        self.trades = []

    def check_price(self, price, trader, symbol):
        for buy in list(self.active_buys):
            if price <= buy:
                trader.buy(symbol, buy, self.order_size)
                self.active_buys.remove(buy)
                self.active_sells.add(self._next(buy))
                self.trades.append({"side":"BUY","price":buy})

        for sell in list(self.active_sells):
            if price >= sell:
                trader.sell(symbol, sell, self.order_size)
                self.active_sells.remove(sell)
                self.active_buys.add(self._prev(sell))
                self.trades.append({"side":"SELL","price":sell})

    def _next(self, level):
        return self.grid[self.grid.index(level)+1]

    def _prev(self, level):
        return self.grid[self.grid.index(level)-1]
