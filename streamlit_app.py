import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from core.market_data import fetch_ohlcv
from core.grid.linear import generate_linear_grid
from core.grid.fibonacci import generate_fibonacci_grid
from core.grid.engine import GridEngine
from core.exchange.simulator import SimulatorTrader

st.set_page_config(layout="wide")
st.title("Grid Trading Bot – Simulation (Range Slider)")

if st.sidebar.button("Reset session"):
    st.session_state.clear()
    st.rerun()

exchange = st.sidebar.selectbox("Exchange", ["Bitvavo"])
symbol = st.sidebar.text_input("Pair", "BTC/EUR")
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m"])

refresh = st.sidebar.slider("Refresh sec", 5, 60, 15)
st_autorefresh(interval=refresh * 1000, key="refresh")

df = fetch_ohlcv(exchange, symbol, timeframe)
price = float(df["close"].iloc[-1])

st.sidebar.markdown("### Grid range (auto)")
range_pct = st.sidebar.slider("Range ± (%)", 0.1, 20.0, 1.0, step=0.1)

lower = price * (1 - range_pct / 100)
upper = price * (1 + range_pct / 100)

st.sidebar.caption(f"Current price: {price:.2f}")
st.sidebar.caption(f"Auto lower: {lower:.2f}")
st.sidebar.caption(f"Auto upper: {upper:.2f}")

grid_type = st.sidebar.selectbox("Grid type", ["Linear", "Fibonacci"])
order_size = st.sidebar.number_input("Order size", value=0.001, min_value=0.0, format="%.6f")

if grid_type == "Linear":
    levels = st.sidebar.slider("Levels", 3, 20, 10)
    grid = generate_linear_grid(lower, upper, levels)
else:
    grid = generate_fibonacci_grid(lower, upper)

if "trader" not in st.session_state:
    st.session_state.trader = SimulatorTrader()

grid_signature = (
    exchange,
    symbol,
    timeframe,
    grid_type,
    round(lower, 2),
    round(upper, 2),
    len(grid),
    float(order_size),
)

if "grid_signature" not in st.session_state or st.session_state.grid_signature != grid_signature:
    st.session_state.grid_signature = grid_signature
    st.session_state.engine = GridEngine(grid, order_size)

st.session_state.engine.check_price(price, st.session_state.trader, symbol)

fig = go.Figure(go.Candlestick(
    x=df["timestamp"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="Price"
))

for lvl in grid:
    fig.add_hline(y=lvl, line_dash="dot")

last_ts = df["timestamp"].iloc[-1]
for t in st.session_state.engine.trades:
    fig.add_scatter(
        x=[last_ts],
        y=[t["price"]],
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
