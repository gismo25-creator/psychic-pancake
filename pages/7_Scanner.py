import math
import pandas as pd
import streamlit as st

from core.market_data import fetch_markets_bitvavo_cached, fetch_tickers_bitvavo_cached, fetch_ohlcv_bitvavo_cached, BitvavoRateLimitBan
from core.ml.volatility import atr, realized_vol, bollinger_bandwidth, adx

st.set_page_config(layout="wide")
st.title("Bitvavo Scanner (EUR markets)")

st.caption("Doel: snel EUR-paren vinden met voldoende liquiditeit, redelijke spreads, en geschikte volatiliteit voor grid/mean-reversion. "
           "Dit is een *scanner*, geen advies of automatische uitvoering.")

with st.sidebar:
    st.subheader("Universe")
    quote = st.selectbox("Quote", ["EUR"], index=0)
    exclude = st.text_input("Exclude (contains, comma)", "UP,DOWN,BULL,BEAR,3L,3S")
    q = st.text_input("Filter (symbol contains)", "")
    max_rows = st.slider("Max rows", 20, 300, 120, step=10)
    st.subheader("Metrics")
    timeframe = st.selectbox("Timeframe (vol metrics)", ["5m","15m","1h"], index=1)
    ohlcv_limit = st.slider("OHLCV candles", 120, 600, 300, step=30)
    compute_ohlcv = st.checkbox("Compute ATR/RV/BB/ADX (slower)", value=False, help="Dit doet per symbol een OHLCV call. Zet dit aan voor alleen top-N om rate-limits te vermijden.")
    top_for_ohlcv = st.slider("OHLCV for top-N by volume (fast mode)", 5, 120, 20, step=10, disabled=not compute_ohlcv)
    sleep_between_ohlcv = st.slider("Min delay between OHLCV calls (sec)", 0.0, 2.0, 0.25, step=0.05, disabled=not compute_ohlcv)
    max_ohlcv_calls = st.slider("Hard cap OHLCV calls per run", 5, 120, 25, step=5, disabled=not compute_ohlcv)
    st.subheader("Ranking")
    rank_mode = st.selectbox("Rank by", ["Volume", "Spread", "Volatility (ATR%)", "Grid score"], index=3)
    run = st.button("Run scan", type="primary")

if not run:
    st.info("Klik **Run scan** om resultaten te laden.")
    st.stop()

# Markets
mkts = fetch_markets_bitvavo_cached(quote=quote)
syms = []
for m in mkts:
    s = m.get("symbol")
    if not s:
        continue
    syms.append(str(s).upper())
syms = sorted(set(syms))

# Excludes
ex_parts = [x.strip().upper() for x in exclude.split(",") if x.strip()]
if ex_parts:
    syms = [s for s in syms if all(p not in s for p in ex_parts)]
if q.strip():
    syms = [s for s in syms if q.strip().upper() in s]

if not syms:
    st.warning("Geen symbolen na filtering.")
    st.stop()

# Tickers
tickers = fetch_tickers_bitvavo_cached(syms)
rows = []
for s in syms:
    t = tickers.get(s) or {}
    bid = t.get("bid")
    ask = t.get("ask")
    last = t.get("last")
    mid = None
    if bid and ask:
        mid = (float(bid) + float(ask)) / 2.0
    elif last:
        mid = float(last)
    spread = None
    if bid and ask and mid and mid > 0:
        spread = (float(ask) - float(bid)) / mid * 100.0
    qvol = t.get("quoteVolume") or t.get("baseVolume")
    try:
        qvol = float(qvol) if qvol is not None else float("nan")
    except Exception:
        qvol = float("nan")
    rows.append({
        "symbol": s,
        "mid": float(mid) if mid is not None else float("nan"),
        "bid": float(bid) if bid is not None else float("nan"),
        "ask": float(ask) if ask is not None else float("nan"),
        "spread_pct": float(spread) if spread is not None else float("nan"),
        "quote_volume_24h": qvol,
    })

df = pd.DataFrame(rows)
if df.empty:
    st.warning("Geen ticker data.")
    st.stop()

# Pre-sort by volume to pick top-N for OHLCV metrics
df["quote_volume_24h"] = pd.to_numeric(df["quote_volume_24h"], errors="coerce")
df = df.sort_values("quote_volume_24h", ascending=False)

if compute_ohlcv:
    df_top = df.head(int(top_for_ohlcv)).copy()
    atr_pcts, rv_vals, bb_vals, adx_vals = [], [], [], []
    ohlcv_calls = 0
    for s in df_top["symbol"].tolist():
        try:
            if ohlcv_calls >= int(max_ohlcv_calls):
                raise RuntimeError("OHLCV_CALL_CAP_REACHED")
            ohlcv_calls += 1
            if float(sleep_between_ohlcv) > 0:
                import time as _time
                _time.sleep(float(sleep_between_ohlcv))
            o = fetch_ohlcv_bitvavo_cached(s, timeframe=timeframe, limit=int(ohlcv_limit))
            px = float(o["close"].iloc[-1])
            o2 = o.copy()
            o2["atr"] = atr(o2, 14)
            o2["rv"] = realized_vol(o2, 30)
            o2["bb"] = bollinger_bandwidth(o2, 20, 2.0)
            o2["adx"] = adx(o2, 14)
            atr_v = float(o2["atr"].iloc[-1])
            atr_p = (atr_v / px * 100.0) if (px > 0 and not math.isnan(atr_v)) else float("nan")
            atr_pcts.append(atr_p)
            rv_vals.append(float(o2["rv"].iloc[-1]))
            bb_vals.append(float(o2["bb"].iloc[-1]))
            adx_vals.append(float(o2["adx"].iloc[-1]))
        except BitvavoRateLimitBan as e:
            st.error(f"Rate-limit ban door Bitvavo. Wacht tot ban verloopt. Details: {e}")
            break
        except RuntimeError as e:
            if str(e) == "OHLCV_CALL_CAP_REACHED":
                st.warning("OHLCV call cap bereikt; toon ranking op basis van volume/spread. Verhoog cap of verlaag top-N.")
                break
            raise
        except Exception:
            atr_pcts.append(float("nan"))
            rv_vals.append(float("nan"))
            bb_vals.append(float("nan"))
            adx_vals.append(float("nan"))
    df_top["atr_pct"] = atr_pcts
    df_top["rv"] = rv_vals
    df_top["bb_bw"] = bb_vals
    df_top["adx"] = adx_vals

    # Merge back
    df = df.merge(df_top[["symbol","atr_pct","rv","bb_bw","adx"]], on="symbol", how="left")
else:
    df["atr_pct"] = float("nan")
    df["rv"] = float("nan")
    df["bb_bw"] = float("nan")
    df["adx"] = float("nan")

# Score: prefer high volume, low spread, moderate volatility
# Grid score = log(volume) * (atr_pct+0.01) / (spread_pct+0.01)
df["spread_pct"] = pd.to_numeric(df["spread_pct"], errors="coerce")
df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
df["atr_pct"] = pd.to_numeric(df["atr_pct"], errors="coerce")
df["grid_score"] = (df["quote_volume_24h"].clip(lower=1.0).apply(lambda x: math.log10(x+1.0)) *
                    (df["atr_pct"].fillna(0.0) + 0.01) /
                    (df["spread_pct"].fillna(999.0) + 0.01))

if rank_mode == "Volume":
    df = df.sort_values("quote_volume_24h", ascending=False)
elif rank_mode == "Spread":
    df = df.sort_values("spread_pct", ascending=True)
elif rank_mode == "Volatility (ATR%)":
    df = df.sort_values("atr_pct", ascending=False)
else:
    df = df.sort_values("grid_score", ascending=False)

df_show = df.head(int(max_rows)).copy()
df_show["mid"] = df_show["mid"].round(6)
df_show["spread_pct"] = df_show["spread_pct"].round(4)
df_show["quote_volume_24h"] = df_show["quote_volume_24h"].round(2)
df_show["atr_pct"] = df_show["atr_pct"].round(3)
df_show["rv"] = df_show["rv"].round(4)
df_show["bb_bw"] = df_show["bb_bw"].round(4)
df_show["adx"] = df_show["adx"].round(2)
df_show["grid_score"] = df_show["grid_score"].round(4)

st.subheader("Results")
st.dataframe(df_show, width="stretch", height=560)

st.caption("Tip: kies paren met voldoende 24h volume, lage spread, en ATR% die bij je grid-range past. "
           "Zeer lage ATR% geeft weinig fills, zeer hoge ATR% vraagt grotere ranges/risico.")