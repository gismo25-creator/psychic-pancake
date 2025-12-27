import ccxt
import pandas as pd

def fetch_ohlcv_bitvavo(symbol: str, timeframe: str = "5m", limit: int = 300) -> pd.DataFrame:
    exchange = ccxt.bitvavo({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def fetch_ticker_bitvavo(symbol: str) -> dict:
    """Return a lightweight ticker dict: {'last','bid','ask','timestamp'} using CCXT public endpoints."""
    exchange = ccxt.bitvavo({"enableRateLimit": True})
    t = exchange.fetch_ticker(symbol)
    return {
        "last": float(t.get("last")) if t.get("last") is not None else None,
        "bid": float(t.get("bid")) if t.get("bid") is not None else None,
        "ask": float(t.get("ask")) if t.get("ask") is not None else None,
        "timestamp": t.get("timestamp"),
        "datetime": t.get("datetime"),
    }
