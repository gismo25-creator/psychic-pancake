import math
from typing import Dict, Optional

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from core.market_data import fetch_ohlcv_bitvavo_cached
from core.ml.volatility import atr
from core.strategy.fvg import detect_fvgs, pick_latest_zone, FVGZone
from core.strategy.fvg_paper import (
    FVGPaperAccount,
    FVGOrder,
    size_from_fixed_notional,
    size_from_risk,
)

st.set_page_config(layout="wide")
st.title("Fair Value Gap (FVG) Strategy – 15m (Paper trading)")

st.caption(
    "Standalone, interpretable strategy: detect displacement + FVG, place a LIMIT entry on the gap, "
    "stop beyond the first candle, target at 2R (configurable). This page does not modify the Grid live page."
)

# ----------------------------
# State namespace
# ----------------------------
NS = "fvg_strategy_v1"

def ns(key: str) -> str:
    return f"{NS}:{key}"

if ns("account") not in st.session_state:
    st.session_state[ns("account")] = FVGPaperAccount(cash_quote=1000.0)

if ns("last_processed") not in st.session_state:
    st.session_state[ns("last_processed")] = {}  # symbol -> timestamp

acct: FVGPaperAccount = st.session_state[ns("account")]
last_processed: Dict[str, pd.Timestamp] = st.session_state[ns("last_processed")]

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.subheader("Market")

symbol = st.sidebar.text_input("Symbol", "ETH/EUR", key=ns("symbol")).strip().upper()
if not symbol:
    st.stop()

timeframe = "15m"  # per request
limit = st.sidebar.slider("History candles", 150, 800, 350, step=50, key=ns("limit"))

refresh_s = st.sidebar.slider("Auto-refresh (seconds)", 5, 120, 30, key=ns("refresh"))
st_autorefresh(interval=int(refresh_s * 1000), key=ns("autorefresh"))

st.sidebar.subheader("Account (paper)")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    maker_fee = st.number_input("Maker fee (%)", 0.0, 1.0, 0.10, step=0.01, key=ns("maker_fee")) / 100.0
with col_b:
    taker_fee = st.number_input("Taker fee (%)", 0.0, 2.0, 0.25, step=0.01, key=ns("taker_fee")) / 100.0

slippage = st.sidebar.number_input("Slippage (%)", 0.0, 0.5, 0.01, step=0.01, key=ns("slippage")) / 100.0

acct.maker_fee = float(maker_fee)
acct.taker_fee = float(taker_fee)
acct.slippage = float(slippage)

st.sidebar.subheader("Setup definition (interpretable)")
lookback_range = st.sidebar.slider("Range lookback candles", 10, 200, 40, step=5, key=ns("lb_range"))
displacement_k_atr = st.sidebar.slider("Displacement body >= k * ATR", 0.5, 3.0, 1.0, step=0.1, key=ns("disp_k"))
min_gap_k_atr = st.sidebar.slider("Min gap size >= k * ATR", 0.05, 1.0, 0.15, step=0.05, key=ns("gap_k"))
require_range_break = st.sidebar.checkbox(
    "Require range break by displacement close",
    value=True,
    help="If enabled, c2 close must close above (bull) / below (bear) the prior range high/low.",
    key=ns("require_break"),
)

st.sidebar.subheader("Entry / SL / TP")
entry_mode = st.sidebar.selectbox(
    "Entry within gap",
    ["Midpoint (50%)", "Upper edge (aggressive fill)", "Lower edge (better price)"],
    index=0,
    key=ns("entry_mode"),
)
sl_buffer_atr = st.sidebar.slider("SL buffer (ATR units)", 0.0, 0.5, 0.05, step=0.01, key=ns("sl_buf"))
r_multiple = st.sidebar.slider("Target R multiple", 1.0, 5.0, 2.0, step=0.25, key=ns("r_mult"))
expire_candles = st.sidebar.slider("Pending order expiry (candles)", 5, 300, 120, step=5, key=ns("expiry"))

st.sidebar.subheader("Position sizing")
size_mode = st.sidebar.selectbox(
    "Sizing mode",
    ["Fixed notional (EUR)", "Risk % of equity"],
    index=0,
    key=ns("size_mode"),
)

if size_mode == "Fixed notional (EUR)":
    notional_eur = st.sidebar.number_input("Notional per trade (EUR)", 10.0, 5000.0, 100.0, step=10.0, key=ns("notional"))
    risk_pct = None
else:
    risk_pct = st.sidebar.number_input("Risk per trade (% of equity)", 0.1, 5.0, 0.5, step=0.1, key=ns("risk_pct"))
    notional_eur = None

max_notional_eur = st.sidebar.number_input(
    "Max notional cap (EUR)", 0.0, 50_000.0, 1000.0, step=100.0, help="0 = no cap", key=ns("max_notional")
)

st.sidebar.subheader("Controls")
run_enabled = st.sidebar.toggle("Enable strategy", value=False, key=ns("run"))
warmup_replay = st.sidebar.checkbox(
    "Warmup: replay recent candles into simulator",
    value=True,
    help="On first run or after reset, process historical candles sequentially so limit fills/SL/TP can trigger.",
    key=ns("warmup"),
)

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Reset paper account", key=ns("reset"), width="stretch"):
        st.session_state[ns("account")] = FVGPaperAccount(
            cash_quote=1000.0,
            maker_fee=float(maker_fee),
            taker_fee=float(taker_fee),
            slippage=float(slippage),
        )
        st.session_state[ns("last_processed")] = {}
        st.rerun()
with c2:
    if st.button("Cancel pending order", key=ns("cancel"), width="stretch"):
        acct.cancel_order(symbol, reason="USER_CANCEL")
        st.rerun()

# ----------------------------
# Fetch data
# ----------------------------
try:
    df = fetch_ohlcv_bitvavo_cached(symbol, timeframe=timeframe, limit=int(limit))
except Exception as e:
    st.error(f"Data error for {symbol}: {e}")
    st.stop()

if df is None or df.empty:
    st.warning("No OHLCV data returned.")
    st.stop()

# ensure sorted
df = df.sort_values("timestamp").reset_index(drop=True)
price = float(df["close"].iloc[-1])

atr_s = atr(df, 14)

# ----------------------------
# Detect zones
# ----------------------------
zones = detect_fvgs(
    df,
    atr_s,
    lookback_range=int(lookback_range),
    displacement_k_atr=float(displacement_k_atr),
    min_gap_k_atr=float(min_gap_k_atr),
    require_range_break=bool(require_range_break),
)

eligible_zones = [z for z in zones if z.direction == "BULL" and float(z.zone_high) < price]
latest: Optional[FVGZone] = pick_latest_zone(eligible_zones, direction="BULL")

# ----------------------------
# Strategy loop (paper)
# ----------------------------

def _entry_from_zone(z: FVGZone) -> float:
    if entry_mode.startswith("Midpoint"):
        return float(z.midpoint)
    if entry_mode.startswith("Upper"):
        return float(z.zone_high)
    return float(z.zone_low)


def _build_order(z: FVGZone, now_ts: pd.Timestamp, last_px: float) -> Optional[FVGOrder]:
    entry = _entry_from_zone(z)
    # Eligibility: only place bullish retracement orders if the market is currently ABOVE the gap.
    # If price is already below the zone, the retrace/mitigation already happened and this setup is invalid.
    if not (float(last_px) > float(z.zone_high)):
        return None
    # Entry must lie inside the zone and below current price (non-marketable limit).
    if not (float(z.zone_low) <= float(entry) <= float(z.zone_high)):
        return None
    if not (float(entry) < float(last_px)):
        return None
    # Long-only spot
    sl = float(z.c1_low) - float(sl_buffer_atr) * float(z.atr)
    if not (sl < entry < last_px * 10):
        return None

    tp = entry + float(r_multiple) * (entry - sl)

    # Size
    eq = acct.equity({symbol: last_px})
    if size_mode == "Fixed notional (EUR)":
        size_base = size_from_fixed_notional(notional_quote=float(notional_eur), entry=entry)
        if max_notional_eur and max_notional_eur > 0:
            size_base = min(size_base, float(max_notional_eur) / entry)
    else:
        size_base = size_from_risk(
            equity_quote=float(eq),
            risk_pct=float(risk_pct),
            entry=entry,
            sl=sl,
            max_notional_quote=float(max_notional_eur) if max_notional_eur else 0.0,
        )

    if size_base <= 1e-12:
        return None

    return FVGOrder(
        symbol=symbol,
        direction="LONG",
        entry=float(entry),
        sl=float(sl),
        tp=float(tp),
        size_base=float(size_base),
        created_time=now_ts,
        zone_low=float(z.zone_low),
        zone_high=float(z.zone_high),
    )


# Feed candles into simulator
# Determine candles to process
last_ts = last_processed.get(symbol)

# Warmup on first run (or after reset) so strategy can show trades quickly
candles_to_process = []
if warmup_replay and last_ts is None:
    # process last ~200 candles (bounded)
    start_idx = max(0, len(df) - 220)
    candles_to_process = df.iloc[start_idx:].to_dict("records")
elif last_ts is None:
    candles_to_process = [df.iloc[-1].to_dict()]
else:
    # process all new candles since last processed
    candles_to_process = df[df["timestamp"] > last_ts].to_dict("records")
    if not candles_to_process:
        candles_to_process = [df.iloc[-1].to_dict()]

# Run engine
if run_enabled:
    for c in candles_to_process:
        acct.on_candle(symbol, c)
        last_processed[symbol] = pd.Timestamp(c["timestamp"])


    # If market is already below the pending bullish zone, the retrace/mitigation likely happened via a gap.
    # In this candle-based simulator a pending LIMIT might not fill on a gap-down (high < entry), so we cancel it
    # to avoid "stale" orders sitting far above the market.
    o = acct.open_orders.get(symbol)
    if o is not None and o.status == "PENDING":
        if float(price) < float(o.zone_low):
            acct.cancel_order(symbol, reason="INVALIDATED_BELOW_ZONE")
            o = None

    # expire pending order
    o = acct.open_orders.get(symbol)
    if o is not None and o.status == "PENDING":
        # expire by candle count age, approximated by timestamp distance
        # (15m timeframe) -> candles = minutes/15
        age_min = (pd.Timestamp(df["timestamp"].iloc[-1]) - pd.Timestamp(o.created_time)).total_seconds() / 60.0
        age_candles = age_min / 15.0
        if age_candles >= float(expire_candles):
            acct.cancel_order(symbol, reason="EXPIRED")

    # If no active order, place a new pending order from latest zone
    if acct.open_orders.get(symbol) is None and latest is not None:
        now_ts = pd.Timestamp(df["timestamp"].iloc[-1])
        new_order = _build_order(latest, now_ts, price)
        if new_order is not None:
            acct.place_order(new_order)

# persist
st.session_state[ns("last_processed")] = last_processed

# ----------------------------
# Header metrics
# ----------------------------

col1, col2, col3, col4 = st.columns(4)
eq = acct.equity({symbol: price})
col1.metric("Price", f"{price:.2f}")
col2.metric(f"Cash ({acct.quote_ccy})", f"{acct.cash:.2f}")
col3.metric(f"Equity ({acct.quote_ccy})", f"{eq:.2f}")
col4.metric("Open order", "YES" if acct.open_orders.get(symbol) is not None else "—")

# ----------------------------
# Explain panel
# ----------------------------

left, right = st.columns([2, 1])

with right:
    st.subheader("Active setup")
    o = acct.open_orders.get(symbol)
    if o is None:
        st.info("No active order. Enable strategy to auto-place a pending LIMIT on the latest bullish FVG.")
    else:
        st.json(o.to_dict())

    st.subheader("Latest eligible bullish FVG (below price)")
    if latest is None:
        st.warning("No eligible bullish FVG found (must be below current price).")
    else:
        st.write(
            {
                "zone_low": float(latest.zone_low),
                "zone_high": float(latest.zone_high),
                "midpoint": float(latest.midpoint),
                "gap_size": float(latest.size),
                "atr": float(latest.atr),
                "range_high": float(latest.range_high),
                "range_low": float(latest.range_low),
                "c1_time": str(latest.c1_time),
                "c2_time": str(latest.c2_time),
                "c3_time": str(latest.c3_time),
            }
        )

with left:
    st.subheader("Chart")

    fig = go.Figure(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )

    # Range levels + FVG zone
    if latest is not None:
        fig.add_hline(y=float(latest.range_high), line_dash="dash", opacity=0.6)
        fig.add_hline(y=float(latest.range_low), line_dash="dash", opacity=0.6)

        # Zone rectangle across a recent window
        x0 = df["timestamp"].iloc[max(0, len(df) - 220)]
        x1 = df["timestamp"].iloc[-1]
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=float(latest.zone_low),
            y1=float(latest.zone_high),
            opacity=0.20,
            line_width=0,
            fillcolor="rgba(0, 200, 0, 0.2)",
        )

    # Active order lines
    o = acct.open_orders.get(symbol)
    if o is not None:
        fig.add_hline(y=float(o.entry), line_dash="dot")
        fig.add_hline(y=float(o.sl), line_dash="dot")
        fig.add_hline(y=float(o.tp), line_dash="dot")

    # Trade markers
    if acct.fills:
        for f in acct.fills[-250:]:
            if f.symbol != symbol:
                continue
            marker = "triangle-up" if f.side == "BUY" else "triangle-down"
            fig.add_scatter(
                x=[f.time],
                y=[f.price],
                mode="markers",
                marker=dict(
                    color="green" if f.side == "BUY" else "red",
                    symbol=marker,
                    size=10,
                ),
                name=f.side,
            )

    fig.update_layout(height=620, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")

# ----------------------------
# Trades / fills table
# ----------------------------

st.subheader("Fills & realized PnL")
if acct.fills:
    fdf = pd.DataFrame([f.__dict__ for f in acct.fills])
    fdf = fdf.sort_values("time", ascending=False)
    fdf["fee_rate_pct"] = (fdf["fee_rate"] * 100).round(3)
    fdf["price"] = fdf["price"].round(4)
    fdf["amount_base"] = fdf["amount_base"].round(6)
    fdf["fee_paid_quote"] = fdf["fee_paid_quote"].round(4)
    fdf["cash_delta_quote"] = fdf["cash_delta_quote"].round(4)
    fdf["pnl_quote"] = fdf["pnl_quote"].round(4)

    # cumulative realized pnl (SELL only)
    fdf_ch = fdf.iloc[::-1].copy()
    run_pnl = 0.0
    cum = []
    for _, r in fdf_ch.iterrows():
        if r["side"] == "SELL":
            run_pnl += float(r["pnl_quote"])
        cum.append(run_pnl)
    fdf_ch["cum_realized_pnl"] = pd.Series(cum, index=fdf_ch.index)
    fdf = fdf_ch.iloc[::-1]

    st.dataframe(
        fdf[[
            "time",
            "symbol",
            "side",
            "price",
            "amount_base",
            "fee_rate_pct",
            "fee_paid_quote",
            "cash_delta_quote",
            "pnl_quote",
            "cum_realized_pnl",
            "reason",
        ]].rename(
            columns={
                "fee_rate_pct": "fee (%)",
                "fee_paid_quote": f"fee paid ({acct.quote_ccy})",
                "cash_delta_quote": f"cash Δ ({acct.quote_ccy})",
                "pnl_quote": f"realized PnL ({acct.quote_ccy})",
                "cum_realized_pnl": f"cum PnL ({acct.quote_ccy})",
            }
        ),
        width="stretch",
        height=340,
    )
else:
    st.info("No fills yet. Enable strategy and allow time for price to retrace into the FVG zone.")

st.caption(
    "Note: This is a paper simulator using candle OHLC for fills. If both TP and SL hit in the same candle, "
    "the simulator assumes SL hits first (conservative)."
)
