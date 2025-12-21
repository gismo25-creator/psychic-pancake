from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests


def _to_utc_ts(x) -> pd.Timestamp:
    """Normalize any timestamp-like input to UTC tz-aware pandas Timestamp.
    Works for tz-naive and tz-aware timestamps.
    """
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        return ts.tz_localize('UTC')
    return ts.tz_convert('UTC')


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
    limit: int = 1440,
) -> pd.DataFrame:
    """Fetch OHLCV candles from Bitvavo REST API over a time range using pagination.

    Why: ccxt/Bitvavo can behave as 'latest-only' for candles (effectively returning max `limit` rows),
    which makes long lookbacks impossible. This implementation paginates using `start`/`end`.

    Notes:
    - Endpoint: GET /v2/{market}/candles with query params: interval, limit, start, end.
    - Docs indicate `limit` for candles is <= 1440.
    - Candles are returned from latest to earliest; we dedupe and sort ascending.
    """
    # Normalize timestamps to UTC tz-aware
    if since is not None:
        since_ts = _to_utc_ts(since)
    else:
        since_ts = None

    if until is not None:
        until_ts = _to_utc_ts(until)
    else:
        until_ts = pd.Timestamp.now(tz="UTC")

    # Bitvavo expects market like BTC-EUR
    market = symbol.replace("/", "-").upper()

    # Limit clamp (Bitvavo candles limit max 1440)
    limit = int(max(1, min(int(limit), 1440)))

    # Interval ms helper
    tf = timeframe.strip()
    tf_map_ms = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "8h": 28_800_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
    }
    if tf not in tf_map_ms:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    tf_ms = tf_map_ms[tf]

    end_ms = int(until_ts.value // 1_000_000)
    since_ms = int(since_ts.value // 1_000_000) if since_ts is not None else (end_ms - limit * tf_ms)

    url = f"https://api.bitvavo.com/v2/{market}/candles"

    rows = []
    safety = 0
    max_pages = 20000  # plenty even for 1m over months

    while end_ms > since_ms and safety < max_pages:
        start_ms = max(since_ms, end_ms - limit * tf_ms)

        params = {
            "interval": tf,
            "limit": limit,
            "start": start_ms,
            "end": end_ms,
        }

        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        # data is list of [timestamp, open, high, low, close, volume]
        rows.extend(data)

        # Move window backwards using oldest candle returned
        try:
            oldest = min(int(c[0]) for c in data)
        except Exception:
            break

        # Prevent infinite loop (no progress)
        if oldest >= end_ms:
            break

        end_ms = oldest - tf_ms
        safety += 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter requested range (timezone-safe)
    if since_ts is not None:
        df = df[df["timestamp"] >= since_ts]
    if until_ts is not None:
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
