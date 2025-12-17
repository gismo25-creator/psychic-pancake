import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.backtest.data_store import load_or_fetch
from core.backtest.replay import run_backtest
from core.backtest.metrics import summarize_run

from core.grid.linear import generate_linear_grid
from core.grid.fibonacci import generate_fibonacci_grid


st.set_page_config(layout="wide")
st.title("Backtest – Offline replay (Bitvavo / ccxt)")

st.info(
    "Step 1: offline replay backtest op candle CLOSE prijzen, met vaste grid per pair (gebaseerd op startprijs). "
    "Dit is bewust simpel en deterministisch, zodat we daarna regime-conditional sets kunnen toevoegen."
)

# --- Inputs
st.sidebar.subheader("Markets")
symbols_input = st.sidebar.text_input("Pairs (comma-separated)", "BTC/EUR, ETH/EUR")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=1)

st.sidebar.subheader("Period")
end = st.sidebar.date_input("End date", value=pd.Timestamp.utcnow().date())
days = st.sidebar.slider("Lookback days", 1, 90, 14)
since = pd.Timestamp(end) - pd.Timedelta(days=int(days))
until = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

st.sidebar.subheader("Simulation")
start_cash = st.sidebar.number_input("Start cash (EUR)", min_value=10.0, value=1000.0, step=100.0)

st.sidebar.subheader("Fees & slippage")
fee_mode = st.sidebar.selectbox("Assume fills as", ["taker", "maker"], index=0)
maker_fee = st.sidebar.number_input("Maker fee (%)", 0.0, 1.0, 0.15, step=0.01) / 100.0
taker_fee = st.sidebar.number_input("Taker fee (%)", 0.0, 1.0, 0.25, step=0.01) / 100.0
slippage = st.sidebar.number_input("Slippage (%)", 0.0, 1.0, 0.05, step=0.01) / 100.0

st.sidebar.subheader("Grid params (backtest)")
grid_type = st.sidebar.selectbox("Grid type", ["Linear", "Fibonacci"], index=0)
base_range_pct = st.sidebar.slider("Base range ± (%)", 0.1, 20.0, 1.0, step=0.1)
base_levels = st.sidebar.slider("Base levels (Linear)", 3, 30, 12) if grid_type == "Linear" else None
order_size = st.sidebar.number_input("Order size (base)", value=0.001, min_value=0.0, format="%.6f")

st.sidebar.subheader("Risk limits")
cap_eur = st.sidebar.number_input("Max exposure per asset (EUR)", min_value=0.0, value=300.0, step=50.0)

force_refresh = st.sidebar.checkbox("Force refresh OHLCV cache", value=False)

run = st.sidebar.button("▶ Run backtest", use_container_width=True)

if not symbols:
    st.stop()

if run:
    # Load data
    dfs = {}
    for sym in symbols:
        with st.spinner(f"Loading {sym} OHLCV..."):
            df = load_or_fetch(sym, timeframe=timeframe, since=since, until=until, force_refresh=force_refresh)
        if df is None or df.empty:
            st.warning(f"No data for {sym}")
            continue
        dfs[sym] = df

    if not dfs:
        st.error("No data loaded.")
        st.stop()

    # Pair cfg for backtest
    pair_cfg = {}
    for sym in dfs.keys():
        pair_cfg[sym] = {
            "grid_type": grid_type,
            "base_range_pct": base_range_pct,
            "base_levels": base_levels if base_levels is not None else 10,
            "order_size": order_size,
        }

    # exposure caps: keyed by base asset
    caps = {}
    for sym in dfs.keys():
        base = sym.split("/")[0]
        caps[base] = float(cap_eur)

    trades_df, equity_curve, trader = run_backtest(
        dfs=dfs,
        pair_cfg=pair_cfg,
        timeframe=timeframe,
        start_cash=float(start_cash),
        maker_fee=float(maker_fee),
        taker_fee=float(taker_fee),
        slippage=float(slippage),
        fee_mode=str(fee_mode),
        quote_ccy="EUR",
        max_exposure_quote=caps,
    )

    summ = summarize_run(equity_curve, trades_df)

    st.subheader("Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Start equity", f"{summ['start_equity']:.2f}")
    c2.metric("End equity", f"{summ['end_equity']:.2f}")
    c3.metric("Total PnL", f"{summ['total_pnl']:.2f}")
    c4.metric("Max drawdown", f"{summ['max_drawdown']*100:.2f}%")
    c5.metric("Win-rate (SELL)", f"{summ['win_rate']*100:.1f}%" if summ["win_rate"] == summ["win_rate"] else "—")

    st.subheader("Equity curve")
    fig = go.Figure()
    fig.add_scatter(x=equity_curve["timestamp"], y=equity_curve["equity"], mode="lines", name="Equity")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trades")
    if trades_df is None or trades_df.empty:
        st.info("No trades executed in this backtest window.")
    else:
        show = trades_df.copy()
        show["price"] = show["price"].astype(float).round(2)
        show["amount"] = show["amount"].astype(float).round(6)
        if "pnl" in show.columns:
            show["pnl"] = show["pnl"].astype(float).round(2)
        st.dataframe(show, use_container_width=True, height=420)

    # Downloads
    st.subheader("Exports")
    if trades_df is not None and not trades_df.empty:
        st.download_button("Download trades.csv", data=trades_df.to_csv(index=False), file_name="trades.csv")
    st.download_button("Download equity_curve.csv", data=equity_curve.to_csv(index=False), file_name="equity_curve.csv")
