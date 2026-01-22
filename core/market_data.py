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
    exchange = _public_exchange_bitvavo()
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



# ----------------------------
# Bitvavo CCXT singleton (public)
# ----------------------------
_PUBLIC_EXCHANGE = None

def _public_exchange_bitvavo():
    """Reuse one CCXT Bitvavo instance for public endpoints to reduce overhead and rate-limit pressure."""
    global _PUBLIC_EXCHANGE
    if _PUBLIC_EXCHANGE is None:
        _PUBLIC_EXCHANGE = ccxt.bitvavo({"enableRateLimit": True})
    return _PUBLIC_EXCHANGE

def fetch_markets_bitvavo(quote: str = "EUR") -> list[dict]:
    """Fetch Bitvavo markets via CCXT and return unified market dicts."""
    ex = _public_exchange_bitvavo()
    markets = ex.load_markets()
    out = []
    for sym, m in markets.items():
        try:
            if quote and str(m.get("quote")).upper() != quote.upper():
                continue
        except Exception:
            pass
        out.append(m)
    return out

def fetch_tickers_bitvavo(symbols: list[str] | None = None) -> dict:
    """Fetch tickers for symbols (or all if supported).

    Returns CCXT unified ticker dict keyed by symbol. Raises BitvavoRateLimitBan on Bitvavo errorCode 105 bans.
    """
    ex = _public_exchange_bitvavo()

    def _handle_exc(e: Exception):
        msg = str(e)
        until = _extract_ban_until_ms(msg)
        if until is not None:
            raise BitvavoRateLimitBan(banned_until_ms=until, message=msg) from e
        raise e

    if symbols:
        # CCXT may not support bulk tickers; attempt and fallback to per-symbol.
        try:
            if hasattr(ex, "fetch_tickers"):
                return ex.fetch_tickers(symbols)
        except Exception as e:
            _handle_exc(e)

        res = {}
        for s in symbols:
            try:
                res[s] = ex.fetch_ticker(s)
            except Exception as e:
                _handle_exc(e)
        return res

    # all
    if hasattr(ex, "fetch_tickers"):
        try:
            return ex.fetch_tickers()
        except Exception as e:
            _handle_exc(e)

    # fallback: no all-tickers support
    return {}

def fetch_ticker_bitvavo(symbol: str) -> dict:
    """Return a lightweight ticker dict: {'last','bid','ask','timestamp'} using CCXT public endpoints."""
    exchange = _public_exchange_bitvavo()
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

# ----------------------------
# Cached wrappers for heavier public calls
# ----------------------------
if st is not None:
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_markets_bitvavo_cached(quote: str = "EUR") -> list[dict]:
        return fetch_markets_bitvavo(quote=quote)

    @st.cache_data(ttl=30, show_spinner=False)
    def fetch_tickers_bitvavo_cached(symbols: list[str] | None = None) -> dict:
        return fetch_tickers_bitvavo(symbols=symbols)
else:
    def fetch_markets_bitvavo_cached(quote: str = "EUR") -> list[dict]:  # type: ignore
        return fetch_markets_bitvavo(quote=quote)

    def fetch_tickers_bitvavo_cached(symbols: list[str] | None = None) -> dict:  # type: ignore
        return fetch_tickers_bitvavo(symbols=symbols)

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
