import re
from dataclasses import dataclass
from typing import Optional

import ccxt
import pandas as pd

try:
    import streamlit as st  # optional; used for caching in Streamlit apps
except Exception:  # pragma: no cover
    st = None


@dataclass
class BitvavoRateLimitBan(Exception):
    banned_until_ms: int
    message: str

    def __str__(self) -> str:
        return f"Bitvavo rate-limit ban until {self.banned_until_ms}: {self.message}"


_BAN_EXPIRES_RE = re.compile(r"(?:expires\s+at\s+)(\d{13})")


def _extract_ban_until_ms(msg: str) -> Optional[int]:
    """Extract Bitvavo ban expiry epoch-ms from an exception or response message."""
    if not msg:
        return None
    m = _BAN_EXPIRES_RE.search(msg)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # Sometimes embedded JSON-like: {"errorCode":105, ...}
    m2 = re.search(r'"errorCode"\s*:\s*105', msg)
    if m2:
        # no expiry found; still a ban-like message, return short backoff (60s)
        return int(pd.Timestamp.utcnow().timestamp() * 1000) + 60_000
    return None


def fetch_ohlcv_bitvavo(symbol: str, timeframe: str = "5m", limit: int = 300) -> pd.DataFrame:
    """Fetch OHLCV via CCXT. Raises BitvavoRateLimitBan on Bitvavo errorCode 105 bans."""
    exchange = ccxt.bitvavo({"enableRateLimit": True})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        msg = str(e)
        until = _extract_ban_until_ms(msg)
        if until is not None:
            raise BitvavoRateLimitBan(banned_until_ms=until, message=msg) from e
        raise

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # Keep tz-aware UTC timestamps for consistent comparisons across modules
    return df


def fetch_ticker_bitvavo(symbol: str) -> dict:
    """Return a lightweight ticker dict: {'last','bid','ask','timestamp'} using CCXT public endpoints."""
    exchange = ccxt.bitvavo({"enableRateLimit": True})
    try:
        t = exchange.fetch_ticker(symbol)
    except Exception as e:
        msg = str(e)
        until = _extract_ban_until_ms(msg)
        if until is not None:
            raise BitvavoRateLimitBan(banned_until_ms=until, message=msg) from e
        raise

    return {
        "last": float(t.get("last")) if t.get("last") is not None else None,
        "bid": float(t.get("bid")) if t.get("bid") is not None else None,
        "ask": float(t.get("ask")) if t.get("ask") is not None else None,
        "timestamp": t.get("timestamp"),
        "datetime": t.get("datetime"),
    }


# ----------------------------
# Streamlit-friendly cached wrappers
# ----------------------------
if st is not None:
    @st.cache_data(ttl=60, show_spinner=False)
    def fetch_ohlcv_bitvavo_cached(symbol: str, timeframe: str = "5m", limit: int = 300) -> pd.DataFrame:
        return fetch_ohlcv_bitvavo(symbol, timeframe=timeframe, limit=limit)

    @st.cache_data(ttl=60, show_spinner=False)
    def fetch_ticker_bitvavo_cached(symbol: str) -> dict:
        return fetch_ticker_bitvavo(symbol)
else:
    # Fallbacks when Streamlit is not available (e.g., unit tests)
    def fetch_ohlcv_bitvavo_cached(symbol: str, timeframe: str = "5m", limit: int = 300) -> pd.DataFrame:  # type: ignore
        return fetch_ohlcv_bitvavo(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker_bitvavo_cached(symbol: str) -> dict:  # type: ignore
        return fetch_ticker_bitvavo(symbol)
