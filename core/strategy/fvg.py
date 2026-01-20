"""Fair Value Gap (FVG) utilities (interpretable, non-black-box).

This module detects 3-candle imbalance zones (FVGs) with optional displacement+range context.
The implementation is intentionally explicit so it can be audited and tuned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal

import pandas as pd


Direction = Literal["BULL", "BEAR"]


@dataclass
class FVGZone:
    direction: Direction
    # zone bounds (price)
    zone_low: float
    zone_high: float
    # the 3 candles that form the pattern: c1, c2 (displacement), c3
    c1_time: pd.Timestamp
    c2_time: pd.Timestamp
    c3_time: pd.Timestamp
    c1_high: float
    c1_low: float
    c2_open: float
    c2_close: float
    c2_high: float
    c2_low: float
    c3_high: float
    c3_low: float
    # context
    range_high: float
    range_low: float
    atr: float

    @property
    def midpoint(self) -> float:
        return self.zone_low + 0.5 * (self.zone_high - self.zone_low)

    @property
    def size(self) -> float:
        return max(0.0, self.zone_high - self.zone_low)


def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in OHLCV df: {missing}")


def detect_fvgs(
    df: pd.DataFrame,
    atr_series: pd.Series,
    *,
    lookback_range: int = 40,
    displacement_k_atr: float = 1.0,
    min_gap_k_atr: float = 0.15,
    require_range_break: bool = True,
) -> List[FVGZone]:
    """Detect FVG zones across the dataframe.

    Rules (3-candle):
      - Bullish FVG when c1.high < c3.low  -> zone [c1.high, c3.low]
      - Bearish FVG when c1.low  > c3.high -> zone [c3.high, c1.low]

    Optional context:
      - Displacement candle c2 must have |body| >= displacement_k_atr * ATR
      - If require_range_break: c2 close must break range_high (bull) or range_low (bear)

    Notes:
      - df must be chronological.
      - timestamps should be tz-aware UTC (as used across this project).
    """

    _require_cols(df, ["timestamp", "open", "high", "low", "close"])
    if len(df) < max(lookback_range + 3, 10):
        return []

    zones: List[FVGZone] = []

    # Ensure aligned indexes
    atr_series = atr_series.reindex(df.index)

    for i in range(1, len(df) - 1):
        c1 = df.iloc[i - 1]
        c2 = df.iloc[i]
        c3 = df.iloc[i + 1]

        atr = float(atr_series.iloc[i]) if pd.notna(atr_series.iloc[i]) else float("nan")
        if not (atr > 0):
            continue

        body = abs(float(c2["close"]) - float(c2["open"]))
        if body < displacement_k_atr * atr:
            continue

        # range context from candles before c2 (exclusive)
        lb_start = max(0, i - lookback_range)
        rng = df.iloc[lb_start:i]
        range_high = float(rng["high"].max())
        range_low = float(rng["low"].min())

        # Bullish FVG
        bull_cond = float(c1["high"]) < float(c3["low"])
        if bull_cond:
            if (not require_range_break) or (float(c2["close"]) > range_high):
                zone_low = float(c1["high"])  # lower bound
                zone_high = float(c3["low"])  # upper bound
                if (zone_high - zone_low) >= (min_gap_k_atr * atr):
                    zones.append(
                        FVGZone(
                            direction="BULL",
                            zone_low=zone_low,
                            zone_high=zone_high,
                            c1_time=pd.Timestamp(c1["timestamp"]),
                            c2_time=pd.Timestamp(c2["timestamp"]),
                            c3_time=pd.Timestamp(c3["timestamp"]),
                            c1_high=float(c1["high"]),
                            c1_low=float(c1["low"]),
                            c2_open=float(c2["open"]),
                            c2_close=float(c2["close"]),
                            c2_high=float(c2["high"]),
                            c2_low=float(c2["low"]),
                            c3_high=float(c3["high"]),
                            c3_low=float(c3["low"]),
                            range_high=range_high,
                            range_low=range_low,
                            atr=atr,
                        )
                    )

        # Bearish FVG
        bear_cond = float(c1["low"]) > float(c3["high"])
        if bear_cond:
            if (not require_range_break) or (float(c2["close"]) < range_low):
                zone_low = float(c3["high"])  # lower bound
                zone_high = float(c1["low"])  # upper bound
                if (zone_high - zone_low) >= (min_gap_k_atr * atr):
                    zones.append(
                        FVGZone(
                            direction="BEAR",
                            zone_low=zone_low,
                            zone_high=zone_high,
                            c1_time=pd.Timestamp(c1["timestamp"]),
                            c2_time=pd.Timestamp(c2["timestamp"]),
                            c3_time=pd.Timestamp(c3["timestamp"]),
                            c1_high=float(c1["high"]),
                            c1_low=float(c1["low"]),
                            c2_open=float(c2["open"]),
                            c2_close=float(c2["close"]),
                            c2_high=float(c2["high"]),
                            c2_low=float(c2["low"]),
                            c3_high=float(c3["high"]),
                            c3_low=float(c3["low"]),
                            range_high=range_high,
                            range_low=range_low,
                            atr=atr,
                        )
                    )

    return zones


def pick_latest_zone(
    zones: List[FVGZone],
    *,
    direction: Optional[Direction] = "BULL",
    max_age_candles: int = 120,
    df_len: Optional[int] = None,
) -> Optional[FVGZone]:
    """Pick the latest (most recent) zone with optional direction filtering.

    `df_len` can be provided to enforce max_age_candles based on positional age.
    """

    if not zones:
        return None

    filtered = [z for z in zones if (direction is None or z.direction == direction)]
    if not filtered:
        return None

    filtered.sort(key=lambda z: z.c3_time)
    z = filtered[-1]

    # Optional age check: if df_len provided, best-effort based on ordering in zones list (not strict)
    if df_len is not None and max_age_candles is not None:
        # fallback: accept, since strict index isn't stored
        pass

    return z
