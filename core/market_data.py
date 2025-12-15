import ccxt
import pandas as pd

def fetch_ohlcv_bitvavo(symbol: str, timeframe: str = "5m", limit: int = 300) -> pd.DataFrame:
    exchange = ccxt.bitvavo({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
