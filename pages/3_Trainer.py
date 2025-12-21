import json
import pandas as pd
import streamlit as st

from core.backtest.data_store import load_or_fetch
from core.backtest.replay import run_backtest
from core.backtest.metrics import summarize_run
from core.training.regime_optimizer import SearchSpace, staged_optimize_regime_profiles

st.set_page_config(layout="wide")
st.title("Trainer – Offline tuning (interpretable profiles + multi-fold walk-forward)")

st.info(
    "Staged grid-search per regime (interpreteerbaar) + multi-fold walk-forward evaluatie. "
    "Je krijgt per fold train/test metrics en een risk-adjusted test score."
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.subheader("Data")
symbols = st.sidebar.multiselect(
    "Symbols",
    ["BTC/EUR", "ETH/EUR", "SOL/EUR", "XRP/EUR", "ADA/EUR"],
    default=["BTC/EUR"],
)
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)
lookback_days = st.sidebar.slider("Lookback (days)", 7, 365, 90)
force_refresh = st.sidebar.checkbox("Force refresh OHLCV cache", value=False)

st.sidebar.subheader("Walk-forward (rolling folds)")
folds = st.sidebar.slider("Folds", 2, 8, 4)
test_window_days = st.sidebar.slider("Test window (days)", 1, 60, 14)
step_days = st.sidebar.slider("Step (days)", 1, 60, 14, help="Hoeveel dagen het window opschuift per fold.")
min_train_days = st.sidebar.slider("Min train (days)", 7, 180, 30)

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
dd_penalty = st.sidebar.slider("DD penalty (trainer objective)", 0.0, 10.0, 3.0, step=0.25)
trade_penalty = st.sidebar.slider("Low-trade penalty", 0.0, 2.0, 0.0, step=0.05)

st.sidebar.subheader("Risk-adjusted test score")
score_mode = st.sidebar.selectbox("Score mode", ["PnL / MaxDD (Calmar-like)", "PnL - λ·MaxDD"], index=0)
lambda_dd = st.sidebar.slider("λ (only for PnL - λ·MaxDD)", 0.0, 50.0, 10.0, step=0.5)

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

run = st.sidebar.button("▶ Train (multi-fold WF)", width="stretch")

if "trained_profiles" not in st.session_state:
    st.session_state.trained_profiles = None
if "trainer_report" not in st.session_state:
    st.session_state.trainer_report = None
if "trainer_fold_details" not in st.session_state:
    st.session_state.trainer_fold_details = None


def _risk_score(total_pnl: float, max_dd_frac: float) -> float:
    dd = max(1e-9, float(max_dd_frac))
    pnl = float(total_pnl)
    if score_mode.startswith("PnL /"):
        return pnl / dd
    return pnl - float(lambda_dd) * dd


def _get_num_trades(summ: dict) -> int:
    if "num_trades" in summ:
        return int(summ.get("num_trades") or 0)
    return int(summ.get("n_trades") or 0)


def _rolling_folds(df: pd.DataFrame):
    df = df.dropna().copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return []

    end = df["timestamp"].max()
    start = max(df["timestamp"].min(), end - pd.Timedelta(days=int(lookback_days)))

    # Folds are built from the end backwards, then reversed to chronological
    fold_list = []
    for k in range(int(folds)):
        test_end = end - pd.Timedelta(days=int(step_days) * k)
        test_start = test_end - pd.Timedelta(days=int(test_window_days))
        train_end = test_start
        train_start = train_end - pd.Timedelta(days=int(min_train_days))

        # clamp train_start to available start
        train_start = max(train_start, start)

        train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)].copy()
        test = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)].copy()

        if train.empty or test.empty:
            continue

        fold_list.append((train_start, train_end, test_start, test_end, train, test))

    return list(reversed(fold_list))


if run:
    if not symbols:
        st.sidebar.error("Select at least one symbol.")
        st.stop()

    search = SearchSpace(
        range_pcts=[float(x) for x in range_candidates],
        levels=[int(x) for x in levels_candidates],
        order_size_mults=[float(x) for x in os_mult_candidates],
        cycle_tp_enable=[bool(x) for x in cycle_tp_enable],
        cycle_tp_pcts=[float(x) for x in cycle_tp_pcts],
    )

    base_cfg = {
        "grid_type": grid_type,
        "base_range_pct": float(base_range_pct),
        "base_levels": int(base_levels) if base_levels is not None else 12,
        "order_size": float(order_size),
        "cycle_tp_enable": False,
        "cycle_tp_pct": 0.35,
    }

    base_profiles = {
        "RANGE": {"range_pct": 1.0, "levels": 14, "order_size_mult": 1.0, "cycle_tp_enable": True, "cycle_tp_pct": 0.35},
        "TREND": {"range_pct": 2.0, "levels": 10, "order_size_mult": 0.8, "cycle_tp_enable": False, "cycle_tp_pct": 0.35},
        "CHAOS": {"range_pct": 3.0, "levels": 8,  "order_size_mult": 0.6, "cycle_tp_enable": True, "cycle_tp_pct": 0.50},
        "WARMUP": {"range_pct": 1.0, "levels": 12, "order_size_mult": 0.8, "cycle_tp_enable": False, "cycle_tp_pct": 0.35},
    }

    trained = {}
    report_rows = []
    fold_rows = []

    prog = st.progress(0, text="Training...")

    for i, sym in enumerate(symbols):
        prog.progress(i / max(1, len(symbols)), text=f"Loading data for {sym}...")
        since = pd.Timestamp.utcnow() - pd.Timedelta(days=int(lookback_days))
        df = load_or_fetch(sym, timeframe=timeframe, since=since, until=None, force_refresh=force_refresh)
        if df is None or df.empty:
            report_rows.append({"symbol": sym, "status": "NO_DATA"})
            continue

        folds_data = _rolling_folds(df)
        if not folds_data:
            report_rows.append({"symbol": sym, "status": "NO_FOLDS"})
            continue

        # Optimize on the most recent fold's train set (fast), evaluate across all folds
        train_start, train_end, test_start, test_end, train_df, _ = folds_data[-1]

        prog.progress((i + 0.2) / max(1, len(symbols)), text=f"Optimize train profiles for {sym} (latest fold)...")
        profiles, best_train = staged_optimize_regime_profiles(
            sym=sym,
            df=train_df,
            base_cfg=base_cfg,
            base_profiles=base_profiles,
            timeframe=timeframe,
            start_cash=float(start_cash),
            maker_fee=float(maker_fee),
            taker_fee=float(taker_fee),
            slippage=float(slippage),
            fee_mode=fee_mode,
            quote_ccy="EUR",
            caps={},
            confirm_n=int(confirm_n),
            cooldown_candles=int(cooldown_candles),
            dd_penalty=float(dd_penalty),
            trade_penalty=float(trade_penalty),
            search=search,
        )

        # Evaluate across folds
        test_scores = []
        test_pnls = []
        test_dds = []

        for fidx, (tr_s, tr_e, te_s, te_e, tr_df, te_df) in enumerate(folds_data):
            dfs = {sym: te_df}
            pair_cfg = {sym: dict(base_cfg)}
            trades_df, equity_curve, decision_log, trader = run_backtest(
                dfs=dfs,
                pair_cfg=pair_cfg,
                timeframe=timeframe,
                start_cash=float(start_cash),
                maker_fee=float(maker_fee),
                taker_fee=float(taker_fee),
                slippage=float(slippage),
                fee_mode=str(fee_mode),
                quote_ccy="EUR",
                max_exposure_quote={},
                regime_profiles=profiles,
                enable_regime_profiles=True,
                confirm_n=int(confirm_n),
                cooldown_candles=int(cooldown_candles),
                rebuild_on_regime_change=False,
            )
            test_summ = summarize_run(equity_curve, trades_df)

            tpnl = float(test_summ.get("total_pnl", 0.0))
            tdd = float(test_summ.get("max_drawdown", 0.0))
            tscore = _risk_score(tpnl, tdd)

            test_scores.append(tscore)
            test_pnls.append(tpnl)
            test_dds.append(tdd)

            fold_rows.append({
                "symbol": sym,
                "fold": fidx + 1,
                "train_start": str(tr_s),
                "train_end": str(tr_e),
                "test_start": str(te_s),
                "test_end": str(te_e),
                "test_total_pnl": tpnl,
                "test_max_dd_pct": tdd * 100.0,
                "test_score": tscore,
                "test_trades": _get_num_trades(test_summ),
                "test_win_rate_pct": float(test_summ.get("win_rate", 0.0)) * 100.0 if test_summ.get("win_rate") == test_summ.get("win_rate") else float("nan"),
            })

        trained[sym] = {
            "use_regime_profiles": True,
            "regime_profile_rebuild": False,
            "regime_profiles": profiles,
        }

        # Aggregate report
        report_rows.append({
            "symbol": sym,
            "status": "OK",
            "train_total_pnl": float(best_train.get("total_pnl", 0.0)),
            "train_max_dd_pct": float(best_train.get("max_drawdown", 0.0)) * 100.0,
            "train_win_rate_pct": float(best_train.get("win_rate", 0.0)) * 100.0 if best_train.get("win_rate") == best_train.get("win_rate") else float("nan"),
            "train_trades": _get_num_trades(best_train),
            "test_score_avg": float(pd.Series(test_scores).mean()),
            "test_score_med": float(pd.Series(test_scores).median()),
            "test_total_pnl_avg": float(pd.Series(test_pnls).mean()),
            "test_max_dd_pct_avg": float(pd.Series(test_dds).mean()) * 100.0,
            "folds_used": len(test_scores),
        })

    prog.progress(1.0, text="Done.")
    st.session_state.trained_profiles = trained
    st.session_state.trainer_report = pd.DataFrame(report_rows)
    st.session_state.trainer_fold_details = pd.DataFrame(fold_rows)
    st.success("Training complete. Profiles stored in session (and downloadable below).")

# ----------------------------
# Output
# ----------------------------
if st.session_state.trainer_report is not None:
    st.subheader("Multi-fold summary")
    st.dataframe(st.session_state.trainer_report, use_container_width=True, height=260)

if st.session_state.trainer_fold_details is not None:
    st.subheader("Fold details (test metrics per fold)")
    st.dataframe(st.session_state.trainer_fold_details, use_container_width=True, height=340)

if st.session_state.trained_profiles:
    st.subheader("Optimized profiles")
    st.json(st.session_state.trained_profiles)

    payload = json.dumps(st.session_state.trained_profiles, indent=2)
    st.download_button("Download profiles.json", data=payload, file_name="profiles.json")

    st.info("Ga terug naar de live pagina en gebruik: 'Apply optimized profiles from Trainer' of importeer profiles.json.")
else:
    st.caption("Nog geen training uitgevoerd.")
