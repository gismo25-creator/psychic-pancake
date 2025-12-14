class SimulatorTrader:
    def __init__(self, balance=1000, fee=0.001, slippage=0.0005):
        self.balance = balance
        self.position = 0.0
        self.fee = fee
        self.slippage = slippage
        self.trades = []

    def _apply_slippage(self, price, side):
        slip = price * self.slippage
        return price + slip if side == "BUY" else price - slip

    def buy(self, symbol, price, amount):
        exec_price = self._apply_slippage(price, "BUY")
        cost = exec_price * amount * (1 + self.fee)
        if self.balance >= cost:
            self.balance -= cost
            self.position += amount
            self.trades.append(("BUY", exec_price, amount))
            return exec_price
        return None

    def sell(self, symbol, price, amount):
        exec_price = self._apply_slippage(price, "SELL")
        if self.position >= amount:
            revenue = exec_price * amount * (1 - self.fee)
            self.balance += revenue
            self.position -= amount
            self.trades.append(("SELL", exec_price, amount))
            return exec_price
        return None

    def equity(self, price):
        return self.balance + self.position * price
