class SimulatorTrader:
    def __init__(self, balance=1000, fee=0.001):
        self.start_balance = balance
        self.balance = balance
        self.position = 0.0
        self.fee = fee
        self.trades = []

    def buy(self, symbol, price, amount):
        cost = price * amount * (1 + self.fee)
        if self.balance >= cost:
            self.balance -= cost
            self.position += amount
            self.trades.append(("BUY", price, amount))

    def sell(self, symbol, price, amount):
        if self.position >= amount:
            revenue = price * amount * (1 - self.fee)
            self.balance += revenue
            self.position -= amount
            self.trades.append(("SELL", price, amount))

    def equity(self, price):
        return self.balance + self.position * price
