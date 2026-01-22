import numpy as np
import pandas as pd

def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = _true_range(df)
    return tr.rolling(period, min_periods=period).mean()

def realized_vol(df: pd.DataFrame, period: int = 30) -> pd.Series:
    r = np.log(df["close"]).diff()
    return r.rolling(period, min_periods=period).std()

def bollinger_bandwidth(df: pd.DataFrame, period: int = 20, nstd: float = 2.0) -> pd.Series:
    ma = df["close"].rolling(period, min_periods=period).mean()
    sd = df["close"].rolling(period, min_periods=period).std()
    upper = ma + nstd * sd
    lower = ma - nstd * sd
    return (upper - lower) / ma

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(df).to_numpy()
    tr_s = pd.Series(tr).rolling(period, min_periods=period).sum()
    plus_s = pd.Series(plus_dm).rolling(period, min_periods=period).sum()
    minus_s = pd.Series(minus_dm).rolling(period, min_periods=period).sum()

    plus_di = 100 * (plus_s / tr_s)
    minus_di = 100 * (minus_s / tr_s)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period, min_periods=period).mean()


def vol_cluster_acf1(df, window: int = 120) -> float:
    """Volatility clustering proxy: autocorrelation of squared log-returns (lag=1) over a rolling window.
    Returns the most recent ACF1 value (nan if insufficient data).
    """
    import math
    import pandas as pd

    if df is None or len(df) < max(3, window + 2):
        return float("nan")

    c = pd.Series(df["close"]).astype(float)
    lr = (c.apply(lambda x: math.log(x))).diff()
    sq = lr * lr
    sq = sq.dropna()
    if len(sq) < window + 2:
        return float("nan")

    w = sq.tail(window + 1)  # need lag1
    x = w.iloc[1:].reset_index(drop=True)
    y = w.iloc[:-1].reset_index(drop=True)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(x.corr(y))
