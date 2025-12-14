import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from core.market_data import fetch_ohlcv
from core.grid.linear import generate_linear_grid
from core.grid.fibonacci import generate_fibonacci_grid
from core.grid.engine import GridEngine
from core.exchange.simulator import SimulatorTrader

st.set_page_config(layout="wide")
st.title("Grid Trading Bot – Simulation (PnL + Slippage)")

exchange = st.sidebar.selectbox("Exchange", ["Bitvavo"])
symbol = st.sidebar.text_input("Pair","BTC/EUR")
timeframe = st.sidebar.selectbox("Timeframe",["1m","5m","15m"])

grid_type = st.sidebar.selectbox("Grid type",["Linear","Fibonacci"])
lower = st.sidebar.number_input("Lower price",50000.0)
upper = st.sidebar.number_input("Upper price",60000.0)
order_size = st.sidebar.number_input("Order size",0.001)

if grid_type=="Linear":
    levels = st.sidebar.slider("Levels",3,20,10)
    grid = generate_linear_grid(lower,upper,levels)
else:
    grid = generate_fibonacci_grid(lower,upper)

refresh = st.sidebar.slider("Refresh sec",5,60,15)
st_autorefresh(interval=refresh*1000,key="refresh")

if "trader" not in st.session_state:
    st.session_state.trader = SimulatorTrader()

if "engine" not in st.session_state:
    st.session_state.engine = GridEngine(grid,order_size)

df = fetch_ohlcv(exchange,symbol,timeframe)
price = df["close"].iloc[-1]

st.session_state.engine.check_price(price,st.session_state.trader,symbol)

fig = go.Figure(go.Candlestick(
    x=df["timestamp"],open=df["open"],high=df["high"],
    low=df["low"],close=df["close"]
))

for lvl in grid:
    fig.add_hline(y=lvl,line_dash="dot")

for t in st.session_state.engine.trades:
    fig.add_scatter(
        x=[df["timestamp"].iloc[-1]],
        y=[t["price"]],
        mode="markers",
        marker=dict(
            color="green" if t["side"]=="BUY" else "red",
            symbol="triangle-up" if t["side"]=="BUY" else "triangle-down",
            size=10
        )
    )

fig.update_layout(height=700,xaxis_rangeslider_visible=False)
st.plotly_chart(fig,use_container_width=True)

st.markdown("### Account")
col1,col2,col3,col4 = st.columns(4)

col1.metric("Balance",f"{st.session_state.trader.balance:.2f}")
col2.metric("Position",f"{st.session_state.trader.position:.6f}")
col3.metric("Equity",f"{st.session_state.trader.equity(price):.2f}")

closed = st.session_state.engine.closed_grids
wins = [g for g in closed if g["pnl"] > 0]
winrate = (len(wins)/len(closed)*100) if closed else 0

col4.metric("Winrate",f"{winrate:.1f}%")

st.metric("Grid PnL",f"{sum(g['pnl'] for g in closed):.2f}")
