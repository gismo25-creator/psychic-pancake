import ccxt
import pandas as pd

def fetch_ohlcv(exchange_name, symbol, timeframe="5m", limit=200):
    if exchange_name == "Binance":
        exchange = ccxt.binance()
    elif exchange_name == "Bitvavo":
        exchange = ccxt.bitvavo()
    else:
        raise ValueError("Unsupported exchange")

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
