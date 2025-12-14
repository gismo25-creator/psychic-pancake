import ccxt

class LiveTrader(BaseTrader):
    def __init__(self, exchange_name, api_key, secret):
        if exchange_name == "Binance":
            self.exchange = ccxt.binance({
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True
            })
        elif exchange_name == "Bitvavo":
            self.exchange = ccxt.bitvavo({
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True
            })
        else:
            raise ValueError("Unsupported exchange")

    def buy(self, symbol, price, amount):
        return self.exchange.create_limit_buy_order(
            symbol, amount, price
        )

    def sell(self, symbol, price, amount):
        return self.exchange.create_limit_sell_order(
            symbol, amount, price
        )
