import math
import time
from collections import deque
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from core.market_data import fetch_ohlcv_bitvavo
from core.grid.linear import generate_linear_grid
from core.grid.fibonacci import generate_fibonacci_grid
from core.grid.engine import GridEngine
from core.exchange.simulator import PortfolioSimulatorTrader

from core.ml.volatility import atr, realized_vol, bollinger_bandwidth, adx
from core.ml.regime import classify_regime

FEE_TIERS_CAT_A = [
    ("€0+",        0.0015, 0.0025),
    ("€100k+",     0.0010, 0.0020),
    ("€250k+",     0.0008, 0.0016),
    ("€500k+",     0.0006, 0.0012),
    ("€1M+",       0.0005, 0.0010),
    ("€2.5M+",     0.0004, 0.0008),
    ("€5M+",       0.0004, 0.0006),
    ("€10M+",      0.0000, 0.0005),
    ("€25M+",      0.0000, 0.0002),
    ("€100M+",     0.0000, 0.0001),
    ("€500M+",     0.0000, 0.0001),
]

st.set_page_config(layout="wide")
st.title("Grid Trading Bot – Bitvavo (Stop-loss testing in simulation)")

if st.sidebar.button("Reset session"):
    st.session_state.clear()
    st.rerun()

# --- Market selection (multi-pair)
st.sidebar.subheader("Market")
symbols_input = st.sidebar.text_input("Pairs (comma-separated)", "BTC/EUR, ETH/EUR")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
if not symbols:
    st.stop()

timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)

refresh = st.sidebar.slider("Refresh sec", 5, 60, 15)
st_autorefresh(interval=refresh * 1000, key="refresh")

# --- Portfolio fees & slippage
st.sidebar.subheader("Fees & slippage")
fee_mode = st.sidebar.selectbox("Assume fills as", ["taker", "maker"], index=0)
tier_labels = [t[0] for t in FEE_TIERS_CAT_A]
tier_map = {t[0]: (t[1], t[2]) for t in FEE_TIERS_CAT_A}
tier_label = st.sidebar.selectbox("30d volume tier (Category A)", tier_labels, index=0)
maker_fee, taker_fee = tier_map[tier_label]

custom_fees = st.sidebar.checkbox("Override fees (custom)", value=False)
if custom_fees:
    maker_fee = st.sidebar.number_input("Maker fee (%)", 0.0, 1.0, float(maker_fee * 100), step=0.01) / 100.0
    taker_fee = st.sidebar.number_input("Taker fee (%)", 0.0, 1.0, float(taker_fee * 100), step=0.01) / 100.0

slippage_pct = st.sidebar.number_input("Slippage (%)", 0.0, 1.0, 0.05, step=0.01) / 100.0

# --- Risk limits
st.sidebar.subheader("Risk limits")
default_cap = st.sidebar.number_input("Max exposure per asset (EUR)", min_value=0.0, value=300.0, step=50.0)
per_asset_caps = {}
for sym in symbols:
    base = sym.split("/")[0]
    if base in per_asset_caps:
        continue
    per_asset_caps[base] = st.sidebar.number_input(f"Cap {base} (EUR)", min_value=0.0, value=float(default_cap), step=50.0)

# --- Stop-loss testing (simulation)
st.sidebar.subheader("Stop-loss testing (simulation)")
enable_portfolio_dd = st.sidebar.checkbox("Enable portfolio drawdown stop", value=True)
max_dd_pct = st.sidebar.slider("Max drawdown (%)", 1.0, 50.0, 10.0, step=0.5, disabled=not enable_portfolio_dd)
dd_action_flatten = st.sidebar.checkbox("On portfolio stop: flatten all positions", value=True, disabled=not enable_portfolio_dd)

enable_asset_stop = st.sidebar.checkbox("Enable per-asset stop", value=True)
asset_stop_pct = st.sidebar.slider("Per-asset stop from avg entry (%)", 0.5, 50.0, 8.0, step=0.5, disabled=not enable_asset_stop)

use_atr_stop = st.sidebar.checkbox("Also enable ATR-based stop", value=False, disabled=not enable_asset_stop)
atr_mult = st.sidebar.slider("ATR multiple (stop = entry - m*ATR)", 0.5, 10.0, 3.0, step=0.5, disabled=(not enable_asset_stop or not use_atr_stop))

# --- Regime stability
st.sidebar.subheader("Regime stability")
cooldown_s = st.sidebar.slider("Cooldown (seconds)", 0, 3600, 300, step=30)
confirm_n = st.sidebar.slider("Confirmations required", 1, 10, 3)

# --- Per-pair grid settings
st.sidebar.subheader("Per-pair grid settings")
if "pair_cfg" not in st.session_state:
    st.session_state.pair_cfg = {}

def default_cfg(sym: str):
    return {
        "grid_type": "Linear",
        "base_range_pct": 1.0,
        "dynamic_spacing": True,
        "k_range": 1.5,
        "k_levels": 0.7,
        "base_levels": 10,
        "order_size": 0.001,
    }

for sym in symbols:
    if sym not in st.session_state.pair_cfg:
        st.session_state.pair_cfg[sym] = default_cfg(sym)
    cfg = st.session_state.pair_cfg[sym]
    with st.sidebar.expander(sym, expanded=False):
        cfg["grid_type"] = st.selectbox(f"{sym} grid type", ["Linear", "Fibonacci"], index=0 if cfg["grid_type"]=="Linear" else 1, key=f"{sym}_grid_type")
        cfg["base_range_pct"] = st.slider(f"{sym} base range ± (%)", 0.1, 20.0, float(cfg["base_range_pct"]), step=0.1, key=f"{sym}_range")
        cfg["dynamic_spacing"] = st.checkbox(f"{sym} regime → dynamic spacing", value=bool(cfg["dynamic_spacing"]), key=f"{sym}_dyn")
        cfg["k_range"] = st.slider(f"{sym} range multiplier strength", 0.5, 3.0, float(cfg["k_range"]), step=0.1, key=f"{sym}_krange")
        cfg["k_levels"] = st.slider(f"{sym} levels reduction strength", 0.3, 1.0, float(cfg["k_levels"]), step=0.05, key=f"{sym}_klevels")
        if cfg["grid_type"] == "Linear":
            cfg["base_levels"] = st.slider(f"{sym} base levels", 3, 30, int(cfg["base_levels"]), key=f"{sym}_levels")
        cfg["order_size"] = st.number_input(f"{sym} order size (base)", value=float(cfg["order_size"]), min_value=0.0, format="%.6f", key=f"{sym}_osize")

# --- Initialize portfolio trader
trader_signature = (maker_fee, taker_fee, slippage_pct, fee_mode, tuple(sorted(per_asset_caps.items())))
if "trader_signature" not in st.session_state or st.session_state.trader_signature != trader_signature:
    st.session_state.trader_signature = trader_signature
    st.session_state.trader = PortfolioSimulatorTrader(
        cash_quote=1000.0,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        slippage=slippage_pct,
        fee_mode=fee_mode,
        quote_ccy="EUR",
        max_exposure_quote=per_asset_caps,
    )
trader: PortfolioSimulatorTrader = st.session_state.trader

# --- State dicts
if "engines" not in st.session_state:
    st.session_state.engines = {}
if "regime_state" not in st.session_state:
    st.session_state.regime_state = {}
if "portfolio_peak_eq" not in st.session_state:
    st.session_state.portfolio_peak_eq = None
if "portfolio_stop_active" not in st.session_state:
    st.session_state.portfolio_stop_active = False
if "asset_halt" not in st.session_state:
    st.session_state.asset_halt = set()  # base assets halted due to stop

# --- Fetch data per pair
dfs: Dict[str, pd.DataFrame] = {}
last_prices: Dict[str, float] = {}
last_ts_map: Dict[str, pd.Timestamp] = {}
atr_abs: Dict[str, float] = {}  # per symbol

for sym in symbols:
    try:
        df = fetch_ohlcv_bitvavo(sym, timeframe=timeframe, limit=300)
    except Exception as e:
        st.error(f"Data error for {sym}: {e}")
        continue
    dfs[sym] = df
    last_prices[sym] = float(df["close"].iloc[-1])
    last_ts_map[sym] = df["timestamp"].iloc[-1]

def compute_metrics(df: pd.DataFrame, price: float) -> Tuple[float, float, float, float, float, str]:
    dfm = df.copy()
    dfm["atr"] = atr(dfm, 14)
    dfm["rv"] = realized_vol(dfm, 30)
    dfm["bb"] = bollinger_bandwidth(dfm, 20, 2.0)
    dfm["adx"] = adx(dfm, 14)

    def last_val(col):
        v = float(dfm[col].iloc[-1])
        return v if not math.isnan(v) else float("nan")

    atr_val = last_val("atr")
    rv_val = last_val("rv")
    bb_val = last_val("bb")
    adx_val = last_val("adx")
    atr_pct = (atr_val / price) if not math.isnan(atr_val) else float("nan")
    regime = classify_regime(dfm, atr_pct, rv_val, bb_val, adx_val)
    return atr_val, atr_pct, rv_val, bb_val, adx_val, regime

def apply_hysteresis(symbol: str, raw_regime: str) -> str:
    if symbol not in st.session_state.regime_state:
        st.session_state.regime_state[symbol] = {"hist": deque(maxlen=confirm_n), "effective": raw_regime, "last_change": 0.0}
    state = st.session_state.regime_state[symbol]
    if state["hist"].maxlen != confirm_n:
        state["hist"] = deque(list(state["hist"]), maxlen=confirm_n)

    state["hist"].append(raw_regime)
    now = time.time()
    hist = list(state["hist"])
    confirmed = (len(hist) == confirm_n) and all(r == hist[0] for r in hist)
    if confirmed:
        candidate = hist[0]
        if candidate != state["effective"] and (now - state["last_change"]) >= cooldown_s:
            state["effective"] = candidate
            state["last_change"] = now
    return state["effective"]

# --- STOP-LOSS CHECKS
# Portfolio equity and drawdown
ts_any = next(iter(last_ts_map.values())) if last_ts_map else pd.Timestamp.utcnow()
eq = trader.equity(last_prices)

if st.session_state.portfolio_peak_eq is None:
    st.session_state.portfolio_peak_eq = eq
else:
    st.session_state.portfolio_peak_eq = max(st.session_state.portfolio_peak_eq, eq)

peak = st.session_state.portfolio_peak_eq or eq
dd = (peak - eq) / peak if peak > 0 else 0.0

portfolio_stop_triggered = False
if enable_portfolio_dd and (dd * 100.0) >= max_dd_pct:
    st.session_state.portfolio_stop_active = True
    portfolio_stop_triggered = True
else:
    # do not auto-clear; user can reset session
    pass

# Per-asset stop: if price below avg entry * (1 - pct) and/or entry - m*ATR
asset_stops_triggered = []
if enable_asset_stop:
    for sym, px in last_prices.items():
        base = sym.split("/")[0]
        pos = trader.positions.get(base, 0.0)
        if pos <= 1e-12:
            continue
        avg_entry = trader.avg_entry_price(base)
        if avg_entry is None:
            continue

        stop_by_pct = px <= avg_entry * (1.0 - asset_stop_pct / 100.0)
        atr_val, _, _, _, _, _ = compute_metrics(dfs[sym], px)
        atr_abs[sym] = atr_val
        stop_by_atr = False
        if use_atr_stop and not math.isnan(atr_val):
            stop_by_atr = px <= (avg_entry - atr_mult * atr_val)

        if stop_by_pct or stop_by_atr:
            asset_stops_triggered.append((sym, base, px, avg_entry, atr_val))

# Execute stop actions
if portfolio_stop_triggered and dd_action_flatten:
    trader.close_all(last_prices, ts_any, reason="STOPLOSS_PORTFOLIO")
    # reset engines (positions flattened)
    for eng in st.session_state.engines.values():
        eng.reset_open_cycles()

if asset_stops_triggered:
    for sym, base, px, avg_entry, atr_val in asset_stops_triggered:
        amt = trader.positions.get(base, 0.0)
        if amt <= 1e-12:
            continue
        trader.sell(sym, px, amt, last_ts_map.get(sym, ts_any), reason="STOPLOSS_ASSET")
        st.session_state.asset_halt.add(base)
        # reset that engine’s open cycles
        if sym in st.session_state.engines:
            st.session_state.engines[sym].reset_open_cycles()

# If portfolio stop active: disallow new buys globally
global_allow_buys = not st.session_state.portfolio_stop_active

# --- Run engines per pair
pair_summaries = {}

for sym, df in dfs.items():
    price = last_prices[sym]
    ts = last_ts_map[sym]
    cfg = st.session_state.pair_cfg[sym]

    atr_val, atr_pct, rv_val, bb_val, adx_val, raw_regime = compute_metrics(df, price)
    atr_abs[sym] = atr_val
    eff_regime = apply_hysteresis(sym, raw_regime)

    range_mult = 1.0
    levels_mult = 1.0
    if cfg["dynamic_spacing"] and eff_regime != "WARMUP":
        if eff_regime == "TREND":
            range_mult = cfg["k_range"]
            levels_mult = cfg["k_levels"]
        elif eff_regime == "CHAOS":
            range_mult = cfg["k_range"] * 1.5
            levels_mult = max(0.3, cfg["k_levels"] * 0.7)

    atr_floor_pct = 0.0
    if cfg["dynamic_spacing"] and not math.isnan(atr_pct):
        atr_floor_pct = max(0.0, 3.0 * atr_pct * 100.0)

    eff_range_pct = max(cfg["base_range_pct"] * range_mult, atr_floor_pct) if cfg["dynamic_spacing"] else cfg["base_range_pct"]
    lower = price * (1 - eff_range_pct / 100.0)
    upper = price * (1 + eff_range_pct / 100.0)

    if cfg["grid_type"] == "Linear":
        eff_levels = cfg["base_levels"] if not cfg["dynamic_spacing"] else max(3, int(round(cfg["base_levels"] * levels_mult)))
        grid = generate_linear_grid(lower, upper, eff_levels)
    else:
        eff_levels = None
        grid = generate_fibonacci_grid(lower, upper)

    sig = (sym, timeframe, cfg["grid_type"], round(lower,2), round(upper,2), len(grid), float(cfg["order_size"]))
    if sym not in st.session_state.engines or getattr(st.session_state.engines[sym], "_signature", None) != sig:
        eng = GridEngine(sym, grid, cfg["order_size"])
        eng._signature = sig
        st.session_state.engines[sym] = eng

    eng: GridEngine = st.session_state.engines[sym]

    base = sym.split("/")[0]
    allow_buys = global_allow_buys and (base not in st.session_state.asset_halt)
    eng.check_price(price, trader, ts, allow_buys=allow_buys)

    pair_summaries[sym] = {
        "price": price,
        "raw_regime": raw_regime,
        "eff_regime": eff_regime,
        "eff_range_pct": eff_range_pct,
        "levels": eff_levels,
        "order_size": cfg["order_size"],
        "pos_base": trader.positions.get(base, 0.0),
        "avg_entry": trader.avg_entry_price(base),
        "closed_pnl": sum(c["pnl"] for c in eng.closed_cycles),
        "trades": len(eng.trades),
        "halted": base in st.session_state.asset_halt,
    }

# --- Portfolio header
st.subheader("Portfolio")
colA, colB, colC, colD, colE = st.columns(5)
colA.metric("Cash (EUR)", f"{trader.cash:.2f}")
colB.metric("Equity (EUR)", f"{eq:.2f}")
colC.metric("Peak equity (EUR)", f"{peak:.2f}")
colD.metric("Drawdown", f"{dd*100:.2f}%")
colE.metric("Portfolio stop", "ACTIVE" if st.session_state.portfolio_stop_active else "—")

if st.session_state.asset_halt:
    st.warning(f"Asset halt active (no new buys): {', '.join(sorted(st.session_state.asset_halt))}")

summary_df = pd.DataFrame([{"symbol": k, **v} for k,v in pair_summaries.items()]).sort_values("symbol")
if not summary_df.empty:
    show = summary_df[["symbol","price","eff_regime","eff_range_pct","levels","order_size","pos_base","avg_entry","halted","closed_pnl","trades"]].copy()
    show["price"] = show["price"].round(2)
    show["eff_range_pct"] = show["eff_range_pct"].round(2)
    show["pos_base"] = show["pos_base"].astype(float).round(6)
    show["avg_entry"] = show["avg_entry"].astype(float).round(2)
    show["closed_pnl"] = show["closed_pnl"].round(2)
    st.dataframe(show.rename(columns={"eff_range_pct":"Eff range ± (%)","closed_pnl":"Realized PnL (EUR)","pos_base":"Pos (base)"}), width="stretch", height=240)

# --- Tabs per pair
tabs = st.tabs(list(dfs.keys()))
for i, sym in enumerate(dfs.keys()):
    with tabs[i]:
        df = dfs[sym]
        price = last_prices[sym]
        eng: GridEngine = st.session_state.engines[sym]
        grid = eng.grid
        base = sym.split("/")[0]

        fig = go.Figure(go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
        ))
        for lvl in grid:
            fig.add_hline(y=lvl, line_dash="dot")

        # trade markers at actual timestamps; include stop-loss markers (reason)
        for t in eng.trades[-400:]:
            symbol = "triangle-up" if t["side"]=="BUY" else "triangle-down"
            fig.add_scatter(
                x=[t["time"]], y=[t["price"]],
                mode="markers",
                marker=dict(
                    color="green" if t["side"]=="BUY" else "red",
                    symbol=symbol,
                    size=10
                ),
                name=f'{t["side"]}'
            )

        fig.update_layout(height=580, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Trades (executed, exact realized PnL on SELL)")
        if eng.trades:
            tdf = pd.DataFrame(eng.trades).sort_values("time", ascending=False)
            tdf["fee_rate_pct"] = (tdf["fee_rate"]*100).round(3)
            tdf["price"] = tdf["price"].round(2)
            tdf["amount"] = tdf["amount"].round(6)
            tdf["fee_paid"] = tdf["fee_paid"].round(2)
            tdf["cash_delta"] = tdf["cash_delta"].round(2)
            tdf["pnl"] = tdf["pnl"].round(2)

            # cumulative realized pnl (SELL rows only, chronological)
            tdf_ch = tdf.iloc[::-1].copy()
            running = 0.0
            cum = []
            for _, row in tdf_ch.iterrows():
                if row["side"] == "SELL":
                    running += float(row["pnl"])
                cum.append(running)
            tdf_ch["cum_realized_pnl"] = pd.Series(cum, index=tdf_ch.index).round(2)
            tdf = tdf_ch.iloc[::-1]

            show_cols = ["time","side","price","amount","fee_rate_pct","fee_paid","cash_delta","pnl","cum_realized_pnl","reason"]
            st.dataframe(
                tdf[show_cols].rename(columns={
                    "fee_rate_pct":"fee (%)",
                    "fee_paid":"fee paid (EUR)",
                    "cash_delta":"cash Δ (EUR)",
                    "pnl":"realized PnL (EUR)",
                    "cum_realized_pnl":"cum PnL (EUR)",
                }),
                width="stretch",
                height=320
            )
        else:
            st.info("Nog geen grid trades.")

        # Also show stop-loss / risk rejections from trader ledger for this symbol
        st.subheader("Order attempts blocked (risk/insufficient)")
        blocked = [t for t in trader.trades if t.symbol == sym and t.reason != "OK" and t.cash_delta_quote == 0.0]
        if blocked:
            bdf = pd.DataFrame([{
                "time": t.time, "side": t.side, "price": t.price, "amount": t.amount, "reason": t.reason
            } for t in blocked]).sort_values("time", ascending=False)
            st.dataframe(bdf, width="stretch", height=180)
        else:
            st.caption("Geen geblokkeerde orders voor deze pair.")

        avg_entry = trader.avg_entry_price(base)
        st.caption(f"Position: {trader.positions.get(base,0.0):.6f} {base} | Avg entry: {avg_entry:.2f} EUR" if avg_entry else f"Position: {trader.positions.get(base,0.0):.6f} {base}")

st.caption("Stop-loss in simulatie: portfolio drawdown stop (optioneel flatten) + per-asset stop (avg-entry % en optioneel ATR). Reset session om stops te clearen.")
