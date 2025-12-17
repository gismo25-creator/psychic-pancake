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

from core.ml.volatility import atr, realized_vol, bollinger_bandwidth, adx, vol_cluster_acf1
from core.ml.regime import classify_regime


# ----------------------------
# Run control state (global)
# ----------------------------
if "trading_enabled" not in st.session_state:
    st.session_state.trading_enabled = False

if "start_pending" not in st.session_state:
    st.session_state.start_pending = False

if "start_pending_ts" not in st.session_state:
    st.session_state.start_pending_ts = 0.0

if "panic_flatten" not in st.session_state:
    st.session_state.panic_flatten = False

if "start_equity" not in st.session_state:
    st.session_state.start_equity = None


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
st.title("Grid Trading Bot – Bitvavo (Simulation + Panic Button + Auto-Pause)")

# --- Top controls: Start / Stop / Stop & Flatten / Reset ---
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if st.button("▶ START", width="stretch"):
        # Require confirmation to resume
        st.session_state.start_pending = True
        st.session_state.start_pending_ts = time.time()

with c2:
    if st.button("⏸ STOP", width="stretch"):
        st.session_state.trading_enabled = False
        st.session_state.start_pending = False

with c3:
    if st.button(
        "🛑 STOP & FLATTEN",
        width="stretch",
        help="Panic button: closes all positions at market (simulation) and pauses trading."
    ):
        # Defer execution until we have latest prices.
        st.session_state.panic_flatten = True
        st.session_state.trading_enabled = False
        st.session_state.start_pending = False
        # Latch portfolio stop so no new buys happen until reset.
        st.session_state.portfolio_stop_active = True

with c4:
    if st.button("🔓 UNLATCH STOP", width="stretch", help="Clear portfolio stop latch (allow new buys again). Trading stays paused; resume manually."):
        st.session_state.portfolio_stop_active = False
        st.session_state.trading_enabled = False
        st.session_state.start_pending = False
        # Reset peak to current equity to avoid immediate retrigger.
        # (Peak is set later after equity is computed.)
        st.session_state.portfolio_peak_eq = None

with c5:
    if st.button("⟲ RESET SESSION", width="stretch"):
        st.session_state.clear()
        st.rerun()


# --- Resume confirmation (anti-misclick) ---
# Window: 15 seconds to confirm, otherwise pending state expires.
if st.session_state.start_pending:
    if (time.time() - st.session_state.start_pending_ts) > 15:
        st.session_state.start_pending = False
    else:
        warn_col, btn_col = st.columns([3, 1])
        with warn_col:
            st.warning("Bevestig START (binnen 15s) om trading te hervatten.")
        with btn_col:
            if st.button("✅ CONFIRM RESUME", width="stretch"):
                # Only allow resume if portfolio stop not active.
                if st.session_state.get("portfolio_stop_active", False):
                    st.error("Portfolio stop is ACTIVE. Reset session om opnieuw te starten.")
                else:
                    st.session_state.trading_enabled = True
                st.session_state.start_pending = False

st.caption(f"Trading status: {'RUNNING' if st.session_state.trading_enabled else 'STOPPED'}")


# ----------------------------
# Market selection
# ----------------------------
st.sidebar.subheader("Market")
symbols_input = st.sidebar.text_input("Pairs (comma-separated)", "BTC/EUR, ETH/EUR")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
if not symbols:
    st.stop()

timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)

refresh = st.sidebar.slider("Refresh sec", 5, 60, 15)
st_autorefresh(interval=refresh * 1000, key="refresh")


# ----------------------------
# Fees & slippage
# ----------------------------
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


# ----------------------------
# Risk limits
# ----------------------------
st.sidebar.subheader("Risk limits")
default_cap = st.sidebar.number_input("Max exposure per asset (EUR)", min_value=0.0, value=300.0, step=50.0)
per_asset_caps = {}
for sym in symbols:
    base = sym.split("/")[0]
    if base in per_asset_caps:
        continue
    per_asset_caps[base] = st.sidebar.number_input(
        f"Cap {base} (EUR)", min_value=0.0, value=float(default_cap), step=50.0
    )


# ----------------------------
# Stop-loss testing (simulation)
# ----------------------------

# ----------------------------
# ----------------------------
# Equity-based position scaling
# ----------------------------
st.sidebar.subheader("Equity-based position scaling (simulation)")
enable_scaling = st.sidebar.checkbox("Enable equity-based scaling", value=False)
scaling_mode = st.sidebar.selectbox(
    "Scaling mode", ["Simple equity scaling", "ATR risk sizing"],
    index=0, disabled=not enable_scaling
)
min_order_size = st.sidebar.number_input(
    "Min order size (base)", min_value=0.0, value=0.0001, format="%.6f",
    disabled=not enable_scaling
)
max_order_size = st.sidebar.number_input(
    "Max order size (base)", min_value=0.0, value=0.01, format="%.6f",
    disabled=not enable_scaling
)
risk_per_trade_pct = st.sidebar.slider(
    "Risk per trade (% equity)", 0.01, 2.00, 0.25, step=0.01,
    disabled=(not enable_scaling or scaling_mode != "ATR risk sizing")
)
atr_risk_mult = st.sidebar.slider(
    "ATR risk multiplier", 0.5, 10.0, 3.0, step=0.5,
    disabled=(not enable_scaling or scaling_mode != "ATR risk sizing")
)
reset_baseline = st.sidebar.button("Reset scaling baseline (start equity)", disabled=not enable_scaling)
if reset_baseline:
    st.session_state.start_equity = None


# ----------------------------
# Portfolio risk: drawdown & correlation
# ----------------------------
st.sidebar.subheader("Portfolio risk: drawdown & correlation")

enable_dd_limit = st.sidebar.checkbox("Enable max assets-in-drawdown", value=True)
dd_asset_threshold_pct = st.sidebar.slider("Asset drawdown threshold (%)", 0.5, 50.0, 5.0, step=0.5, disabled=not enable_dd_limit)
max_assets_in_dd = st.sidebar.slider("Max assets in drawdown", 0, 10, 2, step=1, disabled=not enable_dd_limit)

enable_corr_filter = st.sidebar.checkbox("Enable correlation filter", value=True)
corr_window = st.sidebar.slider("Correlation window (candles)", 20, 300, 120, step=10, disabled=not enable_corr_filter)
corr_threshold = st.sidebar.slider("Correlation threshold", 0.0, 0.99, 0.85, step=0.01, disabled=not enable_corr_filter)
st.sidebar.subheader("Stop-loss testing (simulation)")
enable_portfolio_dd = st.sidebar.checkbox("Enable portfolio drawdown stop", value=True)
max_dd_pct = st.sidebar.slider(
    "Max drawdown (%)", 1.0, 50.0, 10.0, step=0.5, disabled=not enable_portfolio_dd
)
dd_action_flatten = st.sidebar.checkbox(
    "On portfolio stop: flatten all positions", value=True, disabled=not enable_portfolio_dd
)

enable_asset_stop = st.sidebar.checkbox("Enable per-asset stop", value=True)
asset_stop_pct = st.sidebar.slider(
    "Per-asset stop from avg entry (%)", 0.5, 50.0, 8.0, step=0.5, disabled=not enable_asset_stop
)

use_atr_stop = st.sidebar.checkbox("Also enable ATR-based stop", value=False, disabled=not enable_asset_stop)
atr_mult = st.sidebar.slider(
    "ATR multiple (stop = entry - m*ATR)", 0.5, 10.0, 3.0, step=0.5,
    disabled=(not enable_asset_stop or not use_atr_stop)
)


# ----------------------------
# Regime stability
# ----------------------------
st.sidebar.subheader("Regime stability")
cooldown_s = st.sidebar.slider("Cooldown (seconds)", 0, 3600, 300, step=30)
confirm_n = st.sidebar.slider("Confirmations required", 1, 10, 3)
# ----------------------------
# Volatility clustering (metrics)
# ----------------------------
st.sidebar.subheader("Volatility clustering (metrics)")
vc_window = st.sidebar.slider("VC window (candles)", 30, 300, 120, step=10)
vc_alert = st.sidebar.slider("VC alert threshold (ACF1)", 0.0, 0.99, 0.35, step=0.01)
# ----------------------------
# Range efficiency & streaks (metrics)
# ----------------------------
st.sidebar.subheader("Range efficiency & streaks (metrics)")
hit_window = st.sidebar.slider("Hit-rate window (candles)", 20, 500, 150, step=10)
streak_scope = st.sidebar.slider("Streak scope (closed cycles)", 20, 2000, 300, step=20)




# ----------------------------
# Per-pair grid settings
# ----------------------------
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
        "enable_cycle_tp": False,
        "cycle_tp_pct": 0.35,
    }

for sym in symbols:
    if sym not in st.session_state.pair_cfg:
        st.session_state.pair_cfg[sym] = default_cfg(sym)
    cfg = st.session_state.pair_cfg[sym]
    with st.sidebar.expander(sym, expanded=False):
        cfg["grid_type"] = st.selectbox(
            f"{sym} grid type", ["Linear", "Fibonacci"],
            index=0 if cfg["grid_type"] == "Linear" else 1,
            key=f"{sym}_grid_type"
        )
        cfg["base_range_pct"] = st.slider(
            f"{sym} base range ± (%)", 0.1, 20.0, float(cfg["base_range_pct"]), step=0.1, key=f"{sym}_range"
        )
        cfg["dynamic_spacing"] = st.checkbox(
            f"{sym} regime → dynamic spacing", value=bool(cfg["dynamic_spacing"]), key=f"{sym}_dyn"
        )
        cfg["k_range"] = st.slider(
            f"{sym} range multiplier strength", 0.5, 3.0, float(cfg["k_range"]), step=0.1, key=f"{sym}_krange"
        )
        cfg["k_levels"] = st.slider(
            f"{sym} levels reduction strength", 0.3, 1.0, float(cfg["k_levels"]), step=0.05, key=f"{sym}_klevels"
        )
        if cfg["grid_type"] == "Linear":
            cfg["base_levels"] = st.slider(
                f"{sym} base levels", 3, 30, int(cfg["base_levels"]), key=f"{sym}_levels"
            )
        cfg["order_size"] = st.number_input(
            f"{sym} order size (base)", value=float(cfg["order_size"]),
            min_value=0.0, format="%.6f", key=f"{sym}_osize"
        )
        cfg["enable_cycle_tp"] = st.checkbox(
            f"{sym} Cycle take-profit (per cycle)", value=bool(cfg.get("enable_cycle_tp", False)), key=f"{sym}_ctp_en"
        )
        cfg["cycle_tp_pct"] = st.slider(
            f"{sym} Cycle TP (%)", 0.05, 5.0, float(cfg.get("cycle_tp_pct", 0.35)), step=0.05,
            disabled=(not bool(cfg.get("enable_cycle_tp", False))), key=f"{sym}_ctp_pct"
        )


# ----------------------------
# Initialize portfolio trader
# ----------------------------
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


# ----------------------------
# Session state dicts
# ----------------------------
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
if "pair_paused" not in st.session_state:
    st.session_state.pair_paused = set()  # symbols paused manually (no trading)


# ----------------------------
# Fetch data per pair
# ----------------------------
dfs: Dict[str, pd.DataFrame] = {}
last_prices: Dict[str, float] = {}
last_ts_map: Dict[str, pd.Timestamp] = {}
atr_abs: Dict[str, float] = {}  # per symbol
vol_cluster_map: Dict[str, float] = {}  # per symbol

for sym in symbols:
    try:
        df = fetch_ohlcv_bitvavo(sym, timeframe=timeframe, limit=300)
    except Exception as e:
        st.error(f"Data error for {sym}: {e}")
        continue
    dfs[sym] = df
    last_prices[sym] = float(df["close"].iloc[-1])
    last_ts_map[sym] = df["timestamp"].iloc[-1]


def compute_returns(df: pd.DataFrame) -> pd.Series:
    # log returns on close
    s = pd.Series(df["close"]).astype(float)
    return (s.apply(lambda x: math.log(x)).diff()).dropna()

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
    now = time.time()
    if symbol not in st.session_state.regime_state:
        st.session_state.regime_state[symbol] = {
            "hist": deque(maxlen=confirm_n),
            "effective": raw_regime,
            "init_ts": now,
            "last_change": now
        }
    state = st.session_state.regime_state[symbol]
    if state["hist"].maxlen != confirm_n:
        state["hist"] = deque(list(state["hist"]), maxlen=confirm_n)

    state["hist"].append(raw_regime)
    hist = list(state["hist"])
    confirmed = (len(hist) == confirm_n) and all(r == hist[0] for r in hist)
    if confirmed:
        candidate = hist[0]
        if candidate != state["effective"] and (now - state["last_change"]) >= cooldown_s:
            state["effective"] = candidate
            state["last_change"] = now
    return state["effective"]


# ----------------------------
# Correlation prep (rolling)
# ----------------------------
corr_matrix = None
if 'enable_corr_filter' in globals() and enable_corr_filter and len(dfs) >= 2:
    rets = {}
    for s, d in dfs.items():
        r = compute_returns(d)
        if len(r) >= corr_window:
            rets[s] = r.tail(corr_window)
    if len(rets) >= 2:
        corr_matrix = pd.DataFrame(rets).corr()

def regime_duration_minutes(symbol: str) -> float:
    state = st.session_state.regime_state.get(symbol)
    if not state:
        return float("nan")
    lc = float(state.get("last_change", 0.0))
    if lc <= 0:
        lc = float(state.get("init_ts", 0.0))
    if lc <= 0:
        return float("nan")
    return (time.time() - lc) / 60.0

def compute_grid_hit_rate(df: pd.DataFrame, grid_levels, window_candles: int) -> float:
    """Range efficiency proxy: % of grid levels that were 'touched' by candle ranges in last window."""
    if df is None or df.empty or not grid_levels:
        return float("nan")
    w = int(max(1, window_candles))
    d = df.tail(w)
    if d.empty:
        return float("nan")
    levels = [float(x) for x in grid_levels]
    hits = set()
    # Candle 'touch' if level between low and high
    lows = d["low"].astype(float).to_numpy()
    highs = d["high"].astype(float).to_numpy()
    for lvl in levels:
        # vectorized-ish check
        for lo, hi in zip(lows, highs):
            if lo <= lvl <= hi:
                hits.add(lvl)
                break
    return float(len(hits) / max(1, len(levels)))

def compute_streaks(pnls):
    """Compute win/loss streak metrics from a list of realized PnL values (chronological)."""
    # classify: +1 win, -1 loss, 0 neutral
    cls = []
    for p in pnls:
        try:
            v = float(p)
        except Exception:
            continue
        if v > 1e-12:
            cls.append(1)
        elif v < -1e-12:
            cls.append(-1)
        else:
            cls.append(0)

    # win rate (exclude zeros)
    nz = [c for c in cls if c != 0]
    wins = sum(1 for c in nz if c == 1)
    losses = sum(1 for c in nz if c == -1)
    win_rate = float(wins / (wins + losses)) if (wins + losses) > 0 else float("nan")

    # streaks
    cur_type = 0
    cur_len = 0
    max_win = 0
    max_loss = 0

    def update_max(t, l):
        nonlocal max_win, max_loss
        if t == 1:
            max_win = max(max_win, l)
        elif t == -1:
            max_loss = max(max_loss, l)

    for c in cls:
        if c == 0:
            update_max(cur_type, cur_len)
            cur_type, cur_len = 0, 0
            continue
        if c == cur_type:
            cur_len += 1
        else:
            update_max(cur_type, cur_len)
            cur_type, cur_len = c, 1
    update_max(cur_type, cur_len)

    # current streak from the end
    end_type = 0
    end_len = 0
    for c in reversed(cls):
        if c == 0:
            break
        if end_type == 0:
            end_type = c
            end_len = 1
        elif c == end_type:
            end_len += 1
        else:
            break

    return {
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "cur_streak_type": "WIN" if end_type == 1 else ("LOSS" if end_type == -1 else "—"),
        "cur_streak_len": int(end_len),
        "max_win_streak": int(max_win),
        "max_loss_streak": int(max_loss),
    }


# ----------------------------
# STOP-LOSS CHECKS + PANIC FLATTEN
# ----------------------------
ts_any = next(iter(last_ts_map.values())) if last_ts_map else pd.Timestamp.utcnow()
eq = trader.equity(last_prices)
# --- Asset drawdown (unrealized vs avg entry) ---
asset_dd = {}  # base -> dd%
assets_in_dd = set()
for sym_, px_ in last_prices.items():
    base_ = sym_.split("/")[0]
    pos_ = trader.positions.get(base_, 0.0)
    if pos_ <= 1e-12:
        continue
    avg_ = trader.avg_entry_price(base_)
    if avg_ is None or avg_ <= 0:
        continue
    dd_pct_asset = max(0.0, (avg_ - float(px_)) / float(avg_) * 100.0)
    asset_dd[base_] = dd_pct_asset
    if 'enable_dd_limit' in globals() and enable_dd_limit and dd_pct_asset >= dd_asset_threshold_pct:
        assets_in_dd.add(base_)
dd_assets_count = len(assets_in_dd)

if st.session_state.start_equity is None:
    st.session_state.start_equity = float(eq) if eq > 0 else 1.0


# Execute panic flatten once prices are known (always)
if st.session_state.get("panic_flatten", False):
    trader.close_all(last_prices, ts_any, reason="PANIC_FLATTEN")
    for eng in st.session_state.engines.values():
        eng.reset_open_cycles()
    st.session_state.panic_flatten = False
    st.session_state.portfolio_stop_active = True
    st.session_state.trading_enabled = False  # auto-pause after panic

# Peak equity / drawdown
if st.session_state.portfolio_peak_eq is None:
    st.session_state.portfolio_peak_eq = eq
else:
    st.session_state.portfolio_peak_eq = max(st.session_state.portfolio_peak_eq, eq)

peak = st.session_state.portfolio_peak_eq or eq
dd = (peak - eq) / peak if peak > 0 else 0.0

# Portfolio drawdown stop (auto-pause)
portfolio_stop_triggered = False
if enable_portfolio_dd and (dd * 100.0) >= max_dd_pct:
    st.session_state.portfolio_stop_active = True
    st.session_state.trading_enabled = False  # auto-pause on portfolio stop
    portfolio_stop_triggered = True

# Per-asset stop checks
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
    for eng in st.session_state.engines.values():
        eng.reset_open_cycles()

if asset_stops_triggered:
    for sym, base, px, avg_entry, atr_val in asset_stops_triggered:
        amt = trader.positions.get(base, 0.0)
        if amt <= 1e-12:
            continue
        trader.sell(sym, px, amt, last_ts_map.get(sym, ts_any), reason="STOPLOSS_ASSET")
        st.session_state.asset_halt.add(base)
        if sym in st.session_state.engines:
            st.session_state.engines[sym].reset_open_cycles()

# If portfolio stop active: disallow new buys globally
global_allow_buys = not st.session_state.portfolio_stop_active


# --- BUY guard: portfolio-level pre-trade filters ---
def buy_guard(symbol: str, amount_base: float, limit_price: float, ts):
    # 1) Max assets-in-drawdown (hard block on new buys once limit reached)
    if 'enable_dd_limit' in globals() and enable_dd_limit and max_assets_in_dd > 0:
        if dd_assets_count >= max_assets_in_dd:
            return False, f"DRAWDOWN_LIMIT: {dd_assets_count} >= {max_assets_in_dd} assets in drawdown"

    # 2) Correlation filter vs currently held assets (base positions > 0)
    if 'enable_corr_filter' in globals() and enable_corr_filter and corr_matrix is not None:
        held_symbols = []
        for b, amt in trader.positions.items():
            if amt > 1e-12:
                held_symbols.append(f"{b}/EUR")
        for hs in held_symbols:
            if hs == symbol:
                continue
            if (symbol in corr_matrix.index) and (hs in corr_matrix.columns):
                c = float(corr_matrix.loc[symbol, hs])
                if (not math.isnan(c)) and c >= corr_threshold:
                    return False, f"CORRELATION_LIMIT: corr({symbol},{hs})={c:.2f} >= {corr_threshold:.2f}"

    return True, "OK"


# ----------------------------
# Run engines per pair
# ----------------------------
pair_summaries = {}

for sym, df in dfs.items():
    price = last_prices[sym]
    ts = last_ts_map[sym]
    cfg = st.session_state.pair_cfg[sym]

    # --- Effective order size (equity scaling) ---
    eff_order_size = float(cfg["order_size"])
    if "enable_scaling" in globals() and enable_scaling:
        if scaling_mode == "Simple equity scaling":
            start_eq = float(st.session_state.start_equity or 1.0)
            scale = (eq / start_eq) if start_eq > 0 else 1.0
            eff_order_size = float(cfg["order_size"]) * max(0.0, scale)
        else:
            # ATR risk sizing: size = (equity * risk%) / (ATR * multiplier)
            atr_tmp, *_ = compute_metrics(df, price)
            if atr_tmp is not None and (not math.isnan(float(atr_tmp))) and float(atr_tmp) > 0:
                risk_eur = float(eq) * (float(risk_per_trade_pct) / 100.0)
                eff_order_size = risk_eur / (float(atr_tmp) * float(atr_risk_mult))
        # clamps
        eff_order_size = max(float(min_order_size), min(float(max_order_size), float(eff_order_size)))

    atr_val, atr_pct, rv_val, bb_val, adx_val, raw_regime = compute_metrics(df, price)
    atr_abs[sym] = atr_val
    eff_regime = apply_hysteresis(sym, raw_regime)
    vc = vol_cluster_acf1(df, window=int(vc_window))
    vol_cluster_map[sym] = vc

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

    sig = (sym, timeframe, cfg["grid_type"], round(lower, 2), round(upper, 2), len(grid), float(cfg["order_size"]), bool(cfg.get("enable_cycle_tp", False)), float(cfg.get("cycle_tp_pct", 0.35)))
    if sym not in st.session_state.engines or getattr(st.session_state.engines[sym], "_signature", None) != sig:
        eng = GridEngine(sym, grid, cfg["order_size"])
        eng._signature = sig
        st.session_state.engines[sym] = eng

    eng: GridEngine = st.session_state.engines[sym]
    eng.order_size = float(eff_order_size)
    eng.enable_cycle_tp = bool(cfg.get("enable_cycle_tp", False))
    eng.cycle_tp_pct = float(cfg.get("cycle_tp_pct", 0.35))

    base = sym.split("/")[0]
    allow_buys = global_allow_buys and (base not in st.session_state.asset_halt)

    pair_is_paused = sym in st.session_state.pair_paused
    if st.session_state.trading_enabled and (not pair_is_paused):
        eng.check_price(price, trader, ts, allow_buys=allow_buys, buy_guard=buy_guard)

    # --- Range efficiency & streaks ---
    hr = compute_grid_hit_rate(df, grid, window_candles=int(hit_window)) if 'hit_window' in globals() else float('nan')
    pnls = [c.get('pnl', 0.0) for c in eng.closed_cycles]
    if 'streak_scope' in globals():
        pnls = pnls[-int(streak_scope):]
    streak = compute_streaks(pnls)

    pair_summaries[sym] = {
        "price": price,
        "raw_regime": raw_regime,
        "eff_regime": eff_regime,
        "regime_dur_min": regime_duration_minutes(sym),
        "vol_cluster_acf1": float(vol_cluster_map.get(sym, float("nan"))),
        "hit_rate": float(hr),
        "win_rate": float(streak["win_rate"]),
        "cur_streak": f"{streak["cur_streak_type"]} {streak["cur_streak_len"]}",
        "max_win_streak": int(streak["max_win_streak"]),
        "max_loss_streak": int(streak["max_loss_streak"]),
        "eff_range_pct": eff_range_pct,
        "levels": eff_levels,
        "order_size": float(eff_order_size),
        "cycle_tp_on": bool(cfg.get("enable_cycle_tp", False)),
        "cycle_tp_pct": float(cfg.get("cycle_tp_pct", 0.35)) if bool(cfg.get("enable_cycle_tp", False)) else 0.0,
        "pos_base": trader.positions.get(base, 0.0),
        "avg_entry": trader.avg_entry_price(base),
        "closed_pnl": sum(c["pnl"] for c in eng.closed_cycles),
        "trades": len(eng.trades),
        "halted": base in st.session_state.asset_halt,
        "paused": pair_is_paused,
        "asset_dd_pct": float(asset_dd.get(base, 0.0)),
        "in_drawdown": base in assets_in_dd,
    }

# ----------------------------
# Portfolio header
# ----------------------------
st.subheader("Portfolio")
colA, colB, colC, colD, colE = st.columns(5)
colA.metric("Cash (EUR)", f"{trader.cash:.2f}")
colB.metric("Equity (EUR)", f"{eq:.2f}")
colC.metric("Peak equity (EUR)", f"{peak:.2f}")
colD.metric("Drawdown", f"{dd*100:.2f}%")
colE.metric("Portfolio stop", "ACTIVE" if st.session_state.portfolio_stop_active else "—")

if st.session_state.asset_halt:
    st.warning(f"Asset halt active (no new buys): {', '.join(sorted(st.session_state.asset_halt))}")

summary_df = pd.DataFrame([{"symbol": k, **v} for k, v in pair_summaries.items()]).sort_values("symbol")
if not summary_df.empty:
    # Defensive: ensure optional ML columns exist (older session states / partial data)
    for col in ["regime_dur_min", "vol_cluster_acf1", "hit_rate", "win_rate", "cur_streak", "max_win_streak", "max_loss_streak"]:
        if col not in summary_df.columns:
            summary_df[col] = float("nan")

    cols = [
        "symbol", "price", "eff_regime", "regime_dur_min", "vol_cluster_acf1", "hit_rate", "win_rate", "cur_streak",
        "eff_range_pct", "levels", "order_size",
        "pos_base", "avg_entry", "asset_dd_pct", "in_drawdown", "halted", "paused", "closed_pnl", "trades"
    ]
    show = summary_df.reindex(columns=cols).copy()

    show["price"] = show["price"].round(2)
    show["eff_range_pct"] = show["eff_range_pct"].round(2)
    show["regime_dur_min"] = show["regime_dur_min"].astype(float).round(1)
    show["vol_cluster_acf1"] = show["vol_cluster_acf1"].astype(float).round(2)
    show["hit_rate"] = (show["hit_rate"].astype(float) * 100.0).round(1)
    show["win_rate"] = (show["win_rate"].astype(float) * 100.0).round(1)
    show["pos_base"] = show["pos_base"].astype(float).round(6)
    show["avg_entry"] = show["avg_entry"].astype(float).round(2)
    show["asset_dd_pct"] = show["asset_dd_pct"].astype(float).round(2)
    show["closed_pnl"] = show["closed_pnl"].round(2)

    st.dataframe(
        show.rename(columns={
            "eff_range_pct": "Eff range ± (%)",
            "closed_pnl": "Realized PnL (EUR)",
            "pos_base": "Pos (base)",
            "asset_dd_pct": "Asset DD (%)",
            "in_drawdown": "In DD?",
            "regime_dur_min": "Regime duration (min)",
            "vol_cluster_acf1": "Vol cluster ACF1"
        }),
        width="stretch",
        height=240
    )

with st.expander("Correlation matrix (rolling)", expanded=False):
    if corr_matrix is None:
        st.caption("Correlation filter disabled or insufficient data.")
    else:
        st.dataframe(corr_matrix.round(2), width="stretch", height=220)


# ----------------------------
# Tabs per pair
# ----------------------------
tabs = st.tabs(list(dfs.keys()))
for i, sym in enumerate(dfs.keys()):
    with tabs[i]:
        df = dfs[sym]
        price = last_prices[sym]
        eng: GridEngine = st.session_state.engines[sym]
        grid = eng.grid
        base = sym.split("/")[0]

        # --- Per-pair pause / resume ---
        p1, p2, p3 = st.columns([1, 1, 2])
        is_paused = sym in st.session_state.pair_paused
        with p1:
            if st.button("⏸ Pause pair", key=f"pause_{sym}", disabled=is_paused, width="stretch"):
                st.session_state.pair_paused.add(sym)
                st.rerun()
        with p2:
            if st.button("▶ Resume pair", key=f"resume_{sym}", disabled=(not is_paused), width="stretch"):
                st.session_state.pair_paused.discard(sym)
                st.rerun()
        with p3:
            pair_state = "PAUSED" if is_paused else "ACTIVE"
            global_state = "RUNNING" if st.session_state.trading_enabled else "STOPPED"
            st.caption(f"Pair status: {pair_state}  |  Global trading: {global_state}")
# --- Range efficiency & streaks quick view ---
hr_pct = float(pair_summaries.get(sym, {}).get("hit_rate", float("nan"))) * 100.0
wr_pct = float(pair_summaries.get(sym, {}).get("win_rate", float("nan"))) * 100.0
st1, st2, st3 = st.columns(3)
st1.metric("Hit-rate", f"{hr_pct:.1f}%" if not math.isnan(hr_pct) else "—")
st2.metric("Win-rate", f"{wr_pct:.1f}%" if not math.isnan(wr_pct) else "—")
st3.metric("Streak", str(pair_summaries.get(sym, {}).get("cur_streak", "—")))


        fig = go.Figure(go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
        ))
        for lvl in grid:
            fig.add_hline(y=lvl, line_dash="dot")

        for t in eng.trades[-400:]:
            marker_symbol = "triangle-up" if t["side"] == "BUY" else "triangle-down"
            fig.add_scatter(
                x=[t["time"]], y=[t["price"]],
                mode="markers",
                marker=dict(
                    color="green" if t["side"] == "BUY" else "red",
                    symbol=marker_symbol,
                    size=10
                ),
                name=f'{t["side"]}'
            )

        fig.update_layout(height=580, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Trades (executed, exact realized PnL on SELL)")
        if eng.trades:
            tdf = pd.DataFrame(eng.trades).sort_values("time", ascending=False)
            tdf["fee_rate_pct"] = (tdf["fee_rate"] * 100).round(3)
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

            show_cols = ["time", "side", "price", "amount", "fee_rate_pct", "fee_paid", "cash_delta", "pnl", "cum_realized_pnl", "reason"]
            st.dataframe(
                tdf[show_cols].rename(columns={
                    "fee_rate_pct": "fee (%)",
                    "fee_paid": "fee paid (EUR)",
                    "cash_delta": "cash Δ (EUR)",
                    "pnl": "realized PnL (EUR)",
                    "cum_realized_pnl": "cum PnL (EUR)",
                }),
                width="stretch",
                height=320
            )
        else:
            st.info("Nog geen grid trades.")

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
        if avg_entry:
            st.caption(f"Position: {trader.positions.get(base, 0.0):.6f} {base} | Avg entry: {avg_entry:.2f} EUR")
        else:
            st.caption(f"Position: {trader.positions.get(base, 0.0):.6f} {base}")

st.caption("Stop-loss in simulatie: portfolio drawdown stop (optioneel flatten) + per-asset stop (avg-entry % en optioneel ATR). Reset session om stops te clearen.")
