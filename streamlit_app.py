import math
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from core.market_data import fetch_ohlcv
from core.grid.linear import generate_linear_grid
from core.grid.fibonacci import generate_fibonacci_grid
from core.grid.engine import GridEngine
from core.exchange.simulator import SimulatorTrader

from core.ml.volatility import atr, realized_vol, bollinger_bandwidth, adx
from core.ml.regime import classify_regime

st.set_page_config(layout="wide")
st.title("Grid Trading Bot – Simulation (Dynamic Spacing)")

if st.sidebar.button("Reset session"):
    st.session_state.clear()
    st.rerun()

exchange = st.sidebar.selectbox("Exchange", ["Binance", "Bitvavo"])
symbol = st.sidebar.text_input("Pair", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m"])

refresh = st.sidebar.slider("Refresh sec", 5, 60, 15)
st_autorefresh(interval=refresh * 1000, key="refresh")

df = fetch_ohlcv(exchange, symbol, timeframe)
price = float(df["close"].iloc[-1])

# --- ML Step 1 metrics
dfm = df.copy()
dfm["atr"] = atr(dfm, 14)
dfm["rv"] = realized_vol(dfm, 30)
dfm["bb"] = bollinger_bandwidth(dfm, 20, 2.0)
dfm["adx"] = adx(dfm, 14)

atr_val = float(dfm["atr"].iloc[-1]) if not math.isnan(float(dfm["atr"].iloc[-1])) else float("nan")
rv_val = float(dfm["rv"].iloc[-1]) if not math.isnan(float(dfm["rv"].iloc[-1])) else float("nan")
bb_val = float(dfm["bb"].iloc[-1]) if not math.isnan(float(dfm["bb"].iloc[-1])) else float("nan")
adx_val = float(dfm["adx"].iloc[-1]) if not math.isnan(float(dfm["adx"].iloc[-1])) else float("nan")

atr_pct = (atr_val / price) if not math.isnan(atr_val) else float("nan")
regime = classify_regime(dfm, atr_pct, rv_val, bb_val, adx_val)

# --- Sidebar: grid settings
st.sidebar.markdown("### Grid range (auto)")
base_range_pct = st.sidebar.slider("Base range ± (%)", 0.1, 20.0, 1.0, step=0.1)

dynamic_spacing = st.sidebar.checkbox("Regime → dynamic spacing", value=True)

# Optional tuning knobs
st.sidebar.markdown("### Dynamic spacing tuning")
k_range = st.sidebar.slider("Range multiplier strength", 0.5, 3.0, 1.5, step=0.1)
k_levels = st.sidebar.slider("Levels reduction strength", 0.3, 1.0, 0.7, step=0.05)

grid_type = st.sidebar.selectbox("Grid type", ["Linear", "Fibonacci"])
order_size = st.sidebar.number_input("Order size", value=0.001, min_value=0.0, format="%.6f")

base_levels = None
if grid_type == "Linear":
    base_levels = st.sidebar.slider("Base levels", 3, 20, 10)

# --- Regime-based adjustments
range_mult = 1.0
levels_mult = 1.0

if dynamic_spacing and regime != "WARMUP":
    if regime == "RANGE":
        range_mult = 1.0
        levels_mult = 1.0
    elif regime == "TREND":
        range_mult = k_range
        levels_mult = k_levels
    elif regime == "CHAOS":
        range_mult = k_range * 1.5
        levels_mult = max(0.3, k_levels * 0.7)

# Volatility-aware floor (keeps range from being too tight when ATR is elevated)
# This sets a minimum range based on ATR percentage.
atr_floor_pct = 0.0
if dynamic_spacing and atr_pct == atr_pct:  # not NaN
    # floor is 3x ATR% (e.g., ATR% 0.2% => floor 0.6%)
    atr_floor_pct = max(0.0, 3.0 * atr_pct * 100.0)

eff_range_pct = max(base_range_pct * range_mult, atr_floor_pct) if dynamic_spacing else base_range_pct

lower = price * (1 - eff_range_pct / 100.0)
upper = price * (1 + eff_range_pct / 100.0)

eff_levels = None
if grid_type == "Linear":
    eff_levels = base_levels if not dynamic_spacing else max(3, int(round(base_levels * levels_mult)))

st.sidebar.caption(f"Regime: {regime}")
st.sidebar.caption(f"Current price: {price:.2f}")
st.sidebar.caption(f"Effective range ±: {eff_range_pct:.2f}%")
if grid_type == "Linear":
    st.sidebar.caption(f"Effective levels: {eff_levels}")

# --- Build grid
if grid_type == "Linear":
    grid = generate_linear_grid(lower, upper, eff_levels)
else:
    grid = generate_fibonacci_grid(lower, upper)

# --- Init trader/engine
if "trader" not in st.session_state:
    st.session_state.trader = SimulatorTrader()

grid_signature = (
    exchange, symbol, timeframe, grid_type,
    round(lower, 2), round(upper, 2),
    len(grid), float(order_size),
)
if "grid_signature" not in st.session_state or st.session_state.grid_signature != grid_signature:
    st.session_state.grid_signature = grid_signature
    st.session_state.engine = GridEngine(grid, order_size)

st.session_state.engine.check_price(price, st.session_state.trader, symbol)

# --- Layout: chart + metrics panel
left, right = st.columns([3, 1], gap="large")

with left:
    fig = go.Figure(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
    ))
    for lvl in grid:
        fig.add_hline(y=lvl, line_dash="dot")

    last_ts = df["timestamp"].iloc[-1]
    for t in st.session_state.engine.trades:
        fig.add_scatter(
            x=[last_ts], y=[t["price"]],
            mode="markers",
            marker=dict(
                color="green" if t["side"] == "BUY" else "red",
                symbol="triangle-up" if t["side"] == "BUY" else "triangle-down",
                size=10
            ),
            name=t["side"]
        )
    fig.update_layout(height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Regime")
    st.metric("Regime", regime)

    # Dynamic spacing summary
    st.markdown("### Spacing")
    step_price = (upper - lower) / eff_levels if grid_type == "Linear" and eff_levels else None
    if step_price is not None:
        st.metric("Grid step", f"{step_price:.2f}")
        st.metric("Step %", f"{(step_price / price) * 100:.3f}%")
    st.metric("Eff range ±", f"{eff_range_pct:.2f}%")
    if eff_levels is not None:
        st.metric("Eff levels", f"{eff_levels}")

    st.markdown("### Volatility")
    st.metric("ATR (14)", f"{atr_val:.2f}" if not math.isnan(atr_val) else "—")
    st.metric("ATR %", f"{atr_pct*100:.2f}%" if not math.isnan(atr_pct) else "—")
    st.metric("Realized vol (30)", f"{rv_val*100:.2f}%" if not math.isnan(rv_val) else "—")
    st.metric("BB bandwidth (20)", f"{bb_val*100:.2f}%" if not math.isnan(bb_val) else "—")

    st.markdown("### Trend strength")
    st.metric("ADX (14)", f"{adx_val:.1f}" if not math.isnan(adx_val) else "—")

    st.caption("Dynamic spacing widens range and reduces levels in TREND/CHAOS. ATR floor prevents too-tight ranges.")

# --- Account
st.markdown("### Account")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Balance", f"{st.session_state.trader.balance:.2f}")
col2.metric("Position", f"{st.session_state.trader.position:.6f}")
col3.metric("Equity", f"{st.session_state.trader.equity(price):.2f}")
closed = st.session_state.engine.closed_grids
wins = [g for g in closed if g["pnl"] > 0]
winrate = (len(wins) / len(closed) * 100) if closed else 0.0
col4.metric("Winrate", f"{winrate:.1f}%")
st.metric("Grid PnL", f"{sum(g['pnl'] for g in closed):.2f}")
