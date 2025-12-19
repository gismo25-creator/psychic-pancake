import json
import pandas as pd
import streamlit as st

from core.backtest.data_store import load_or_fetch
from core.training.regime_optimizer import SearchSpace, staged_optimize_regime_profiles

st.set_page_config(layout="wide")
st.title("Trainer – Offline tuning (interpretable profiles)")

st.info(
    "Deze trainer voert een staged grid-search uit (per regime één voor één) op historische data. "
    "Dit blijft interpreteerbaar: je krijgt per regime concrete parameters (range, levels, order-size multiplier, Cycle TP)."
)

st.sidebar.subheader("Data")
symbol = st.sidebar.selectbox("Symbol", ["BTC/EUR", "ETH/EUR", "ICNT/EUR"], index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)
lookback_days = st.sidebar.slider("Lookback (days)", 1, 90, 30)
force_refresh = st.sidebar.checkbox("Force refresh OHLCV cache", value=False)

st.sidebar.subheader("Fees / slippage (simulation)")
start_cash = st.sidebar.number_input("Start cash (EUR)", min_value=0.0, value=1000.0, step=100.0)
fee_mode = st.sidebar.selectbox("Assume fills as", ["taker", "maker"], index=0)
maker_fee = st.sidebar.number_input("Maker fee (%)", 0.0, 1.0, 0.10, step=0.01) / 100.0
taker_fee = st.sidebar.number_input("Taker fee (%)", 0.0, 1.0, 0.25, step=0.01) / 100.0
slippage = st.sidebar.number_input("Slippage (%)", 0.0, 1.0, 0.05, step=0.01) / 100.0

st.sidebar.subheader("Regime stability")
confirm_n = st.sidebar.slider("Regime confirmations", 1, 10, 3)
cooldown_candles = st.sidebar.slider("Cooldown (candles)", 0, 200, 0, step=5)

st.sidebar.subheader("Objective (interpretable)")
dd_penalty = st.sidebar.slider("DD penalty", 0.0, 10.0, 3.0, step=0.25)
trade_penalty = st.sidebar.slider("Low-trade penalty", 0.0, 2.0, 0.0, step=0.05)

st.sidebar.subheader("Base grid (baseline)")
grid_type = st.sidebar.selectbox("Grid type", ["Linear", "Fibonacci"], index=0)
base_range_pct = st.sidebar.slider("Base range ± (%)", 0.1, 20.0, 1.0, step=0.1)
base_levels = None
if grid_type == "Linear":
    base_levels = st.sidebar.slider("Base levels", 3, 30, 12)
order_size = st.sidebar.number_input("Base order size (base asset)", min_value=0.0, value=0.001, format="%.6f")

st.sidebar.subheader("Search space (fast)")
range_candidates = st.sidebar.multiselect("Range candidates (%)", [0.6,0.8,1.0,1.3,1.6,2.0,2.5,3.0], default=[0.8,1.0,1.3,1.6,2.0])
levels_candidates = st.sidebar.multiselect("Levels candidates", [6,8,10,12,14,16,18,20], default=[10,12,14,16])
os_mult_candidates = st.sidebar.multiselect("Order-size mult candidates", [0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.5,2.0], default=[0.6,0.8,1.0,1.2])
cycle_tp_enable = st.sidebar.multiselect("Cycle TP enable", [False, True], default=[False, True])
cycle_tp_pcts = st.sidebar.multiselect("Cycle TP (%) candidates", [0.15,0.20,0.35,0.50,0.80,1.00], default=[0.20,0.35,0.50,0.80])

run = st.sidebar.button("▶ Train profiles", width="stretch")

if "trained_profiles" not in st.session_state:
    st.session_state.trained_profiles = None

if run:
    # Load data window
    until = None
    since = pd.Timestamp.utcnow() - pd.Timedelta(days=int(lookback_days))

    with st.spinner("Loading OHLCV..."):
        df = load_or_fetch(symbol, timeframe=timeframe, since=since, until=until, force_refresh=force_refresh)
    if df is None or df.empty:
        st.error("No data returned.")
        st.stop()

    base_cfg = {
        "grid_type": grid_type,
        "base_range_pct": float(base_range_pct),
        "base_levels": int(base_levels) if base_levels is not None else 12,
        "order_size": float(order_size),
        # Cycle TP defaults can be overridden in profiles
        "cycle_tp_enable": False,
        "cycle_tp_pct": 0.35,
    }

    # Default starting profiles
    base_profiles = {
        "RANGE": {"range_pct": 1.0, "levels": 14, "order_size_mult": 1.0, "cycle_tp_enable": True, "cycle_tp_pct": 0.35},
        "TREND": {"range_pct": 2.0, "levels": 10, "order_size_mult": 0.8, "cycle_tp_enable": False, "cycle_tp_pct": 0.35},
        "CHAOS": {"range_pct": 3.0, "levels": 8,  "order_size_mult": 0.6, "cycle_tp_enable": True, "cycle_tp_pct": 0.50},
        "WARMUP": {"range_pct": 1.0, "levels": 12, "order_size_mult": 0.8, "cycle_tp_enable": False, "cycle_tp_pct": 0.35},
    }

    search = SearchSpace(
        range_pcts=[float(x) for x in range_candidates],
        levels=[int(x) for x in levels_candidates],
        order_size_mults=[float(x) for x in os_mult_candidates],
        cycle_tp_enable=[bool(x) for x in cycle_tp_enable],
        cycle_tp_pcts=[float(x) for x in cycle_tp_pcts],
    )

    with st.spinner("Optimizing (staged per regime)..."):
        profiles, best = staged_optimize_regime_profiles(
            sym=symbol,
            df=df,
            base_cfg=base_cfg,
            base_profiles=base_profiles,
            timeframe=timeframe,
            start_cash=float(start_cash),
            maker_fee=float(maker_fee),
            taker_fee=float(taker_fee),
            slippage=float(slippage),
            fee_mode=fee_mode,
            quote_ccy="EUR",
            caps={},  # keep trainer unconstrained unless you want caps later
            confirm_n=int(confirm_n),
            cooldown_candles=int(cooldown_candles),
            dd_penalty=float(dd_penalty),
            trade_penalty=float(trade_penalty),
            search=search,
        )

    st.session_state.trained_profiles = {
        symbol: {
            "use_regime_profiles": True,
            "regime_profile_rebuild": False,
            "regime_profiles": profiles,
        }
    }
    st.success("Training complete. Profiles stored in session (and downloadable below).")

if st.session_state.trained_profiles:
    st.subheader("Optimized profiles")
    st.json(st.session_state.trained_profiles)

    payload = json.dumps(st.session_state.trained_profiles, indent=2)
    st.download_button("Download profiles.json", data=payload, file_name="profiles.json")

    st.info("Ga terug naar de live pagina en gebruik: 'Apply optimized profiles from Trainer' (sidebar) of importeer profiles.json.")
else:
    st.caption("Nog geen training uitgevoerd.")
