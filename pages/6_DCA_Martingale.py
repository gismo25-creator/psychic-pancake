import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from core.market_data import fetch_ohlcv_bitvavo
from core.strategy.dca_martingale import DCAConfig, DCAPaperAccount, DCAStrategy, compute_safety_quote_for_budget

st.set_page_config(layout="wide")
st.title("DCA / Martingale (Safety Orders) – Standalone Strategy (OKX-like)")

NS = "dca_mg"
def k(key: str) -> str:
    return f"{NS}:{key}"

# ----------------------------
# Presets (OKX-like)
# ----------------------------
OKX_PRESETS = {
    "Custom": None,
    "OKX Conservative": {
        "step_pct": 1.2, "max_safety": 6, "base_quote": 6.0, "safety_quote": 6.0, "vol_scale": 1.0,
        "tp_pct": 1.3, "enable_stop": False, "stop_pct": 20.0,
    },
    "OKX Balanced": {
        "step_pct": 1.0, "max_safety": 8, "base_quote": 6.05, "safety_quote": 6.05, "vol_scale": 1.0,
        "tp_pct": 1.7, "enable_stop": False, "stop_pct": 20.0,
    },
    "OKX Aggressive": {
        "step_pct": 0.8, "max_safety": 10, "base_quote": 6.0, "safety_quote": 6.0, "vol_scale": 1.2,
        "tp_pct": 1.7, "enable_stop": True, "stop_pct": 20.0,
    },
}

st.sidebar.subheader("Market")
symbol = st.sidebar.text_input("Symbol", st.session_state.get(k("symbol"), "AXS/EUR"), key=k("symbol")).strip().upper()
timeframe = st.sidebar.selectbox("Timeframe", ["15m", "5m", "1m"], index=0, key=k("tf"))
limit = st.sidebar.slider("History candles", 200, 2000, int(st.session_state.get(k("limit"), 800)), step=50, key=k("limit"))

refresh = st.sidebar.slider("Refresh sec", 5, 120, int(st.session_state.get(k("refresh"), 30)), key=k("refresh"))
st_autorefresh(interval=refresh * 1000, key=k("refresh_key"))

st.sidebar.subheader("Mode")
mode = st.sidebar.selectbox("Execution mode", ["Replay candles (paper)", "Live ticker (paper)"], index=0, key=k("mode"))
max_one_safety_per_tick = st.sidebar.checkbox("Live ticker: max 1 safety fill per tick", value=True, key=k("one_safety_tick"),
                                             help="Avoid filling multiple safety orders on a single price print.")

st.sidebar.subheader("Fees & slippage (assumptions)")
fee_mode = st.sidebar.selectbox("Assume fills as", ["maker", "taker"], index=0, key=k("fee_mode"))
maker_fee = st.sidebar.number_input("Maker fee (%)", 0.0, 1.0, float(st.session_state.get(k("maker_fee"), 0.10)), step=0.01, key=k("maker_fee")) / 100.0
taker_fee = st.sidebar.number_input("Taker fee (%)", 0.0, 1.0, float(st.session_state.get(k("taker_fee"), 0.25)), step=0.01, key=k("taker_fee")) / 100.0

st.sidebar.markdown("**Slippage (separate assumptions)**")
slip_buy = st.sidebar.number_input("BUY slippage (%)", 0.0, 1.0, float(st.session_state.get(k("slip_buy"), 0.01)), step=0.01, key=k("slip_buy")) / 100.0
slip_tp = st.sidebar.number_input("TP (limit) slippage (%)", 0.0, 1.0, float(st.session_state.get(k("slip_tp"), 0.00)), step=0.01, key=k("slip_tp")) / 100.0
slip_stop = st.sidebar.number_input("STOP (market-ish) slippage (%)", 0.0, 2.0, float(st.session_state.get(k("slip_stop"), 0.05)), step=0.01, key=k("slip_stop")) / 100.0

st.sidebar.subheader("OKX preset (optional)")
preset = st.sidebar.selectbox("Preset", list(OKX_PRESETS.keys()), index=0, key=k("preset"))
apply_preset = st.sidebar.button("Apply preset", use_container_width=True)
if apply_preset and preset != "Custom":
    p = OKX_PRESETS[preset]
    for kk, vv in p.items():
        st.session_state[k(kk)] = vv
    st.rerun()

st.sidebar.subheader("Allocation / reserved funds (OKX-like)")
reserve_funds = st.sidebar.checkbox("Reserve funds + enforce budget cap", value=True, key=k("reserve_funds"),
                                    help="Shows 'Funds reserved' and prevents buys beyond the allocated budget for this bot.")
budget_quote = st.sidebar.number_input("Allocated budget (EUR)", min_value=10.0, value=float(st.session_state.get(k("budget"), 200.0)), step=10.0, key=k("budget"))

st.sidebar.subheader("DCA ladder (OKX-like)")
sizing_mode = st.sidebar.selectbox("Safety sizing mode", ["Fixed (OKX-like)", "Scaled (martingale)", "Budget-based ladder"], index=0, key=k("sizing_mode"),
                                   help="Budget-based computes safety amount so base + all safeties ≈ budget.")
step_pct = st.sidebar.slider("Price steps (%)", 0.2, 5.0, float(st.session_state.get(k("step_pct"), 1.0)), step=0.1, key=k("step_pct"))
max_safety = st.sidebar.slider("Max safety orders", 0, 30, int(st.session_state.get(k("max_safety"), 8)), step=1, key=k("max_safety"))
base_quote = st.sidebar.number_input("Initial order amount (EUR)", min_value=0.0, value=float(st.session_state.get(k("base_quote"), 6.05)), step=1.0, key=k("base_quote"))
vol_scale = st.sidebar.slider("Volume scale (martingale)", 1.0, 3.0, float(st.session_state.get(k("vol_scale"), 1.0)), step=0.05, key=k("vol_scale"))

if sizing_mode == "Budget-based ladder":
    computed_safety = compute_safety_quote_for_budget(base_quote, max_safety, vol_scale, budget_quote)
    st.sidebar.number_input("Safety order amount (computed)", min_value=0.0, value=float(computed_safety), step=0.01, key=k("safety_quote"), disabled=True)
    safety_quote = float(computed_safety)
elif sizing_mode == "Fixed (OKX-like)":
    safety_quote = st.sidebar.number_input("Safety order amount (EUR)", min_value=0.0, value=float(st.session_state.get(k("safety_quote"), 6.05)), step=1.0, key=k("safety_quote"))
else:
    safety_quote = st.sidebar.number_input("Safety order base (EUR)", min_value=0.0, value=float(st.session_state.get(k("safety_quote"), 6.05)), step=1.0, key=k("safety_quote"))

st.sidebar.subheader("Exits")
tp_pct = st.sidebar.slider("Take-profit per cycle (%)", 0.1, 10.0, float(st.session_state.get(k("tp_pct"), 1.7)), step=0.1, key=k("tp_pct"))
enable_stop = st.sidebar.checkbox("Enable stop-loss goal", value=bool(st.session_state.get(k("enable_stop"), False)), key=k("enable_stop"))
stop_pct = st.sidebar.slider("Stop-loss goal (%)", 1.0, 60.0, float(st.session_state.get(k("stop_pct"), 20.0)), step=0.5, key=k("stop_pct"), disabled=not enable_stop)
sell_all_on_stop = st.sidebar.checkbox("Sell all on stop-loss", value=True, key=k("sell_all_on_stop"), disabled=not enable_stop)
sell_all_on_manual_stop = st.sidebar.checkbox("If STOP clicked: sell all (paper)", value=False, key=k("sell_all_manual_stop"),
                                              help="OKX often supports closing positions when stopping; keep OFF for safer testing.")

st.sidebar.subheader("Lifecycle")
auto_restart = st.sidebar.checkbox("Auto-restart cycles", value=bool(st.session_state.get(k("auto_restart"), True)), key=k("auto_restart"))
cash0 = st.sidebar.number_input("Paper starting cash (EUR)", min_value=10.0, value=float(st.session_state.get(k("cash0"), 200.0)), step=50.0, key=k("cash0"))

# ----------------------------
# Controls
# ----------------------------
c1, c2, c3 = st.columns(3)
if c1.button("▶ START / RUN", use_container_width=True):
    st.session_state[k("run")] = True
if c2.button("⏸ STOP", use_container_width=True):
    st.session_state[k("run")] = False
    # apply manual stop behavior if strategy exists
    if k("strat") in st.session_state:
        strat: DCAStrategy = st.session_state[k("strat")]
        # use last known price as mark
        try:
            df_tmp = fetch_ohlcv_bitvavo(symbol, timeframe=timeframe, limit=5)
            mark = float(df_tmp["close"].iloc[-1]) if df_tmp is not None and (not df_tmp.empty) else None
        except Exception:
            mark = None
        if mark is not None:
            strat.manual_stop(pd.Timestamp.utcnow(), mark)
if c3.button("⟲ RESET (strategy only)", use_container_width=True):
    for kk in list(st.session_state.keys()):
        if kk.startswith(NS + ":"):
            del st.session_state[kk]
    st.rerun()

running = st.session_state.get(k("run"), False)
st.caption(f"Status: {'RUNNING' if running else 'STOPPED'} | Mode: {mode} | Symbol: {symbol} | TF: {timeframe}")

# ----------------------------
# Data
# ----------------------------
df = fetch_ohlcv_bitvavo(symbol, timeframe=timeframe, limit=int(limit))
if df is None or df.empty:
    st.error("No data returned.")
    st.stop()

price = float(df["close"].iloc[-1])
now_ts = pd.Timestamp(df["timestamp"].iloc[-1])

# ----------------------------
# Build config + strategy (signature)
# ----------------------------
sizing_mode_norm = "budget" if sizing_mode == "Budget-based ladder" else ("fixed" if sizing_mode == "Fixed (OKX-like)" else "scaled")

sig = (
    symbol, timeframe, mode,
    float(step_pct), int(max_safety), float(base_quote), float(safety_quote), float(vol_scale),
    float(tp_pct), bool(enable_stop), float(stop_pct), bool(auto_restart),
    fee_mode, float(maker_fee), float(taker_fee),
    float(slip_buy), float(slip_tp), float(slip_stop),
    bool(reserve_funds), float(budget_quote),
    bool(max_one_safety_per_tick), bool(sell_all_on_manual_stop)
)

if st.session_state.get(k("sig")) != sig:
    st.session_state[k("sig")] = sig
    cfg = DCAConfig(
        step_pct=float(step_pct),
        max_safety_orders=int(max_safety),
        base_order_quote=float(base_quote),
        safety_order_quote=float(safety_quote),
        volume_scale=float(vol_scale),
        sizing_mode=str(sizing_mode_norm),
        budget_quote=float(budget_quote),
        reserve_funds=bool(reserve_funds),
        tp_pct=float(tp_pct),
        enable_stop=bool(enable_stop),
        stop_pct=float(stop_pct),
        sell_all_on_stop=bool(sell_all_on_stop),
        sell_all_on_manual_stop=bool(sell_all_on_manual_stop),
        fee_mode=str(fee_mode),
        maker_fee=float(maker_fee),
        taker_fee=float(taker_fee),
        slippage_buy=float(slip_buy),
        slippage_tp=float(slip_tp),
        slippage_stop=float(slip_stop),
        auto_restart=bool(auto_restart),
        max_one_safety_per_tick=bool(max_one_safety_per_tick),
    )
    acct = DCAPaperAccount(cash_quote=float(cash0), quote_ccy="EUR")
    strat = DCAStrategy(symbol, cfg, acct)
    st.session_state[k("strat")] = strat
    st.session_state[k("cursor")] = 0

strat: DCAStrategy = st.session_state[k("strat")]

# ----------------------------
# Run step
# ----------------------------
if running:
    if mode.startswith("Replay"):
        max_steps = 80
        steps = 0
        cursor = int(st.session_state.get(k("cursor"), 0))
        while cursor < len(df) and steps < max_steps:
            bar = {
                "timestamp": df["timestamp"].iloc[cursor],
                "open": df["open"].iloc[cursor],
                "high": df["high"].iloc[cursor],
                "low": df["low"].iloc[cursor],
                "close": df["close"].iloc[cursor],
            }
            strat.on_bar(bar, allow_new_cycle=True)
            cursor += 1
            steps += 1
        st.session_state[k("cursor")] = cursor
    else:
        # Live ticker paper: evaluate on current price only
        strat.on_tick(now_ts, price, allow_new_cycle=True)

# ----------------------------
# Snapshot + dashboard
# ----------------------------
snap = strat.status_snapshot(price)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Price", f"{price:.4f}")
m2.metric("Equity (EUR)", f"{snap['equity']:.2f}")
m3.metric("Realized PnL (EUR)", f"{snap['realized_pnl']:.2f}")
m4.metric("Unrealized PnL (EUR)", f"{snap['unrealized_pnl']:.2f}")
m5.metric("Invested (EUR)", f"{snap['invested']:.2f}")
m6.metric("Cycle / Completed", f"{snap['cycle']} / {snap['completed_cycles']}")

st.subheader("Cycle dashboard (interpretable)")
cA, cB, cC, cD, cE = st.columns(5)
avg = snap["avg_entry"]
tp = snap["tp_price"]
nxt = snap["next_safety"]
cA.metric("Avg entry", f"{avg:.4f}" if avg is not None else "—")
cB.metric("TP price", f"{tp:.4f}" if tp is not None else "—")
if tp is not None:
    dist_tp = (tp - price) / price * 100.0
    cC.metric("Distance to TP", f"{dist_tp:+.2f}%")
else:
    cC.metric("Distance to TP", "—")
cD.metric("Next safety", f"{nxt:.4f}" if nxt is not None else "—")
if nxt is not None:
    dist_n = (price - nxt) / price * 100.0
    cE.metric("Distance to next safety", f"{dist_n:+.2f}%")
else:
    cE.metric("Distance to next safety", "—")

if snap.get("budget_quote") == snap.get("budget_quote"):  # not NaN
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Allocated budget", f"{snap['budget_quote']:.2f} EUR")
    b2.metric("Spent (incl fees)", f"{snap['spent_total']:.2f} EUR")
    b3.metric("Budget left", f"{snap['budget_left']:.2f} EUR")
    b4.metric("Funds reserved (est.)", f"{snap['funds_reserved']:.2f} EUR")

st.subheader("Active setup (state)")
st.json({
    "symbol": snap["symbol"],
    "cycle": snap["cycle"],
    "active": snap["active"],
    "filled_safety": f"{snap['filled_safety']} / {snap['max_safety']}",
    "pos_base": snap["pos_base"],
    "avg_entry": snap["avg_entry"],
    "tp_price": snap["tp_price"],
    "stop_price": snap["stop_price"],
    "next_safety": snap["next_safety"],
    "cash": snap["cash"],
})

# ----------------------------
# Charts
# ----------------------------
levels = strat.ladder_levels()
tp = snap["tp_price"]
avg = snap["avg_entry"]

fig = go.Figure(go.Candlestick(
    x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
))
for lvl in levels:
    fig.add_hline(y=lvl, line_dash="dot", opacity=0.35)
if avg is not None:
    fig.add_hline(y=float(avg), line_dash="solid", opacity=0.6)
if tp is not None:
    fig.add_hline(y=float(tp), line_dash="dash", opacity=0.8)
if enable_stop and snap["stop_price"] is not None:
    fig.add_hline(y=float(snap["stop_price"]), line_dash="dashdot", opacity=0.8)

for f in strat.acct.fills[-400:]:
    symb = "triangle-up" if f.side == "BUY" else "triangle-down"
    fig.add_scatter(
        x=[f.time], y=[f.price],
        mode="markers",
        marker=dict(symbol=symb, size=9, color="green" if f.side=="BUY" else "red"),
        name=f.side
    )

fig.update_layout(height=520, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Ladder mini-visual (vertical)
st.subheader("Ladder view (OKX-like)")
lad = go.Figure()
ys = []
labels = []
for i, lvl in enumerate(levels, start=1):
    ys.append(lvl); labels.append(f"S{i}")
if avg is not None:
    ys.append(float(avg)); labels.append("AVG")
if tp is not None:
    ys.append(float(tp)); labels.append("TP")
if enable_stop and snap["stop_price"] is not None:
    ys.append(float(snap["stop_price"])); labels.append("SL")
ys.append(float(price)); labels.append("PX")

lad.add_scatter(x=[0]*len(ys), y=ys, mode="markers+text", text=labels, textposition="middle right")
lad.update_layout(height=380, xaxis=dict(visible=False), yaxis_title="Price")
st.plotly_chart(lad, use_container_width=True)

# ----------------------------
# Fills table
# ----------------------------
st.subheader("Fills / trades")
fills = strat.acct.fills
if fills:
    tdf = pd.DataFrame([{
        "time": f.time, "side": f.side, "price": f.price, "amount_base": f.amount_base,
        "fee_paid": f.fee_paid_quote, "fee_rate_pct": f.fee_rate * 100.0,
        "cash_delta": f.cash_delta_quote, "reason": f.reason
    } for f in fills]).sort_values("time", ascending=False)
    tdf["price"] = tdf["price"].astype(float).round(6)
    tdf["amount_base"] = tdf["amount_base"].astype(float).round(6)
    tdf["fee_paid"] = tdf["fee_paid"].astype(float).round(4)
    tdf["cash_delta"] = tdf["cash_delta"].astype(float).round(4)
    tdf["fee_rate_pct"] = tdf["fee_rate_pct"].astype(float).round(3)
    st.dataframe(tdf, use_container_width=True, height=360)
else:
    st.info("No fills yet. Click START/RUN and let it run.")

st.subheader("Notes")
st.write(
    "- OKX-like DCA/martingale ladder simulator on Bitvavo candles/ticker.\n"
    "- Replay mode uses intrabar wick logic (Stop → TP → Safety).\n"
    "- Live ticker mode evaluates on the last price sample (optional: only one safety fill per tick).\n"
    "- Reserved funds: shows an estimate and enforces budget cap (prevents over-buying)."
)
