from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


def _to_utc_ts(x) -> pd.Timestamp:
    """Normalize any timestamp-like input to UTC tz-aware pandas Timestamp."""
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        return ts.tz_localize('UTC')
    return ts.tz_convert('UTC')

import ccxt


def _safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_").upper()


def cache_dir() -> str:
    d = os.path.join(os.getcwd(), ".cache", "ohlcv")
    os.makedirs(d, exist_ok=True)
    return d


def cache_path(symbol: str, timeframe: str) -> str:
    return os.path.join(cache_dir(), f"{_safe_name(symbol)}_{timeframe}.csv")


def load_cached(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    path = cache_path(symbol, timeframe)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def save_cache(symbol: str, timeframe: str, df: pd.DataFrame) -> str:
    path = cache_path(symbol, timeframe)
    df2 = df.copy()
    df2.to_csv(path, index=False)
    return path


def fetch_ohlcv_range_bitvavo(
    symbol: str,
    timeframe: str,
    since: Optional[pd.Timestamp] = None,
    until: Optional[pd.Timestamp] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV from Bitvavo via ccxt over a time range using pagination.

    Notes:
    - Bitvavo/ccxt typically supports `since` in ms.
    - This function paginates forward in time.
    """
    exchange = ccxt.bitvavo({"enableRateLimit": True})

    tf_s = exchange.parse_timeframe(timeframe)  # seconds
    tf_ms = int(tf_s * 1000)

    since_ms = int(pd.Timestamp(since).value // 1_000_000) if since is not None else None
    until_ms = int(pd.Timestamp(until).value // 1_000_000) if until is not None else None

    all_rows = []
    next_since = since_ms

    # Safety max pages to avoid infinite loops
    max_pages = 5000
    pages = 0

    while pages < max_pages:
        pages += 1
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=next_since, limit=limit)
        if not rows:
            break

        all_rows.extend(rows)
        last_ts = rows[-1][0]

        # Stop if we reached until
        if until_ms is not None and last_ts >= until_ms:
            break

        # Advance since to next candle
        next_since = last_ts + tf_ms

        # Stop if we didn't move forward (paranoia)
        if next_since <= (next_since or 0):
            break

    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Filter range (timezone-safe)
    if since is not None:
        since_ts = _to_utc_ts(since)
        df = df[df["timestamp"] >= since_ts]
    if until is not None:
        until_ts = _to_utc_ts(until)
        df = df[df["timestamp"] <= until_ts]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_or_fetch(
    symbol: str,
    timeframe: str,
    since: Optional[pd.Timestamp],
    until: Optional[pd.Timestamp],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load from local cache; fetch missing range if needed (simplified)."""
    cached = None if force_refresh else load_cached(symbol, timeframe)
    if cached is not None and (not cached.empty):
        # Normalize cached timestamps to UTC (handles older cache formats)
        cached = cached.copy()
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
    if cached is None or cached.empty:
        df = fetch_ohlcv_range_bitvavo(symbol, timeframe=timeframe, since=since, until=until)
        save_cache(symbol, timeframe, df)
        return df

    # If cached covers requested range, return slice
    cmin = cached["timestamp"].min()
    cmax = cached["timestamp"].max()
    need_fetch = False
    if since is not None and _to_utc_ts(since) < _to_utc_ts(cmin):
        need_fetch = True
    if until is not None and _to_utc_ts(until) > _to_utc_ts(cmax):
        need_fetch = True

    if not need_fetch:
        df = cached.copy()
        if since is not None:
            df = df[df["timestamp"] >= _to_utc_ts(since)]
        if until is not None:
            df = df[df["timestamp"] <= _to_utc_ts(until)]
        return df.reset_index(drop=True)

    # Fetch full requested range and overwrite cache (keep it simple/robust)
    df = fetch_ohlcv_range_bitvavo(symbol, timeframe=timeframe, since=since, until=until)
    save_cache(symbol, timeframe, df)
    return df
