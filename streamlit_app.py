import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from core.market_data import fetch_ohlcv
from core.grid.linear import generate_linear_grid
from core.grid.fibonacci import generate_fibonacci_grid

st.set_page_config(layout="wide")
st.title("Grid Trading Bot – Live Chart & Grid")

exchange = st.sidebar.selectbox("Exchange", ["Binance", "Bitvavo"])
symbol = st.sidebar.text_input("Trading pair", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m"])

grid_type = st.sidebar.selectbox("Grid type", ["Linear", "Fibonacci"])
lower = st.sidebar.number_input("Lower price", value=50000.0)
upper = st.sidebar.number_input("Upper price", value=60000.0)

if grid_type == "Linear":
    levels = st.sidebar.slider("Aantal grids", 3, 20, 10)
    grid_levels = generate_linear_grid(lower, upper, levels)
else:
    grid_levels = generate_fibonacci_grid(lower, upper)

refresh_seconds = st.sidebar.slider("Auto-refresh (seconden)", 5, 60, 15)

st_autorefresh(interval=refresh_seconds * 1000, key="price_refresh")

df = fetch_ohlcv(exchange, symbol, timeframe)

fig = go.Figure(
    data=[
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price"
        )
    ]
)

for level in grid_levels:
    fig.add_hline(y=level, line_width=1, line_dash="dot", line_color="gray")

fig.update_layout(
    height=700,
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

last_price = df["close"].iloc[-1]
st.caption(f"Laatst bekende prijs: {last_price:.2f}")
