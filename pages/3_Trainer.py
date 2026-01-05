from pathlib import Path
import json
import os
import pandas as pd
import streamlit as st

from core.profiles.registry import make_bundle, save_bundle, stable_hash_df, ensure_store_dir, list_bundles, load_bundle

from core.backtest.data_store import load_or_fetch
from core.backtest.replay import run_backtest
from core.backtest.metrics import summarize_run
from core.training.regime_optimizer import SearchSpace, staged_optimize_regime_profiles

def _git_commit() -> str:
    """Best-effort git commit hash; returns empty string if git is unavailable."""
    try:
        import subprocess
        return subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""

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

st.sidebar.subheader("Base strategy (used by trainer)")

grid_type = st.sidebar.selectbox("Grid type", ["Linear", "Fibonacci"], index=0)
base_range_pct = st.sidebar.slider("Base range ± (%)", 0.1, 20.0, 1.0, step=0.1)
base_levels = None
if grid_type == "Linear":
    base_levels = st.sidebar.slider("Base levels", 3, 30, 12)
order_size = st.sidebar.number_input("Base order size (base asset units)", min_value=0.0, value=0.001, format="%.6f")

st.sidebar.subheader("Search space (optimizer candidates)")

def _parse_csv_floats(s: str, default: list[float]) -> list[float]:
    try:
        vals = [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]
        return vals if vals else list(default)
    except Exception:
        return list(default)

def _parse_csv_ints(s: str, default: list[int]) -> list[int]:
    try:
        vals = [int(float(x.strip())) for x in str(s).split(",") if x.strip() != ""]
        return vals if vals else list(default)
    except Exception:
        return list(default)

range_candidates_str = st.sidebar.text_input("Range candidates (%)", value="0.5, 1.0, 1.5, 2.0")
levels_candidates_str = st.sidebar.text_input("Levels candidates", value="8, 10, 12, 14, 16")
os_mult_candidates_str = st.sidebar.text_input("Order size mult candidates", value="0.6, 0.8, 1.0, 1.2")

cycle_tp_mode = st.sidebar.selectbox("Cycle TP enable candidates", ["Both", "Enabled only", "Disabled only"], index=0)
if cycle_tp_mode == "Both":
    cycle_tp_enable = [False, True]
elif cycle_tp_mode == "Enabled only":
    cycle_tp_enable = [True]
else:
    cycle_tp_enable = [False]

cycle_tp_pcts_str = st.sidebar.text_input("Cycle TP % candidates", value="0.20, 0.35, 0.50")

# Parsed candidate lists used by SearchSpace
range_candidates = _parse_csv_floats(range_candidates_str, default=[1.0])
levels_candidates = _parse_csv_ints(levels_candidates_str, default=[12])
os_mult_candidates = _parse_csv_floats(os_mult_candidates_str, default=[1.0])
cycle_tp_pcts = _parse_csv_floats(cycle_tp_pcts_str, default=[0.35])

st.sidebar.subheader("BB mean-reversion buy-filter (interpretable)")
bb_mr_enable = st.sidebar.checkbox("Enable BB mean-reversion buy-filter", value=True,
    help="Blocks new BUYs unless price/limit is sufficiently below the Bollinger mid-band (z-score threshold).")
bb_mr_window = st.sidebar.slider("BB window", 10, 60, 20)
bb_mr_z = st.sidebar.slider("Z-threshold (buy only if z <= -thr)", 0.0, 3.0, 0.75, step=0.05)

st.sidebar.subheader("Inventory & trend guards (interpretable)")

# Trend-guard (downtrend): block new BUYs in TREND-down regimes
trend_guard_enable = st.sidebar.checkbox(
    "Enable TREND downtrend guard (block BUYs)",
    value=True,
    help="When the effective regime is TREND and price has fallen over the lookback window beyond the threshold, new BUYs are blocked (SELLs still allowed)."
, key="trainer_trend_guard_enable")
trend_lookback = st.sidebar.slider("Trend lookback (candles)", 8, 200, 48, step=4, disabled=(not trend_guard_enable), key="trainer_trend_lookback")
trend_down_thresh_pct = st.sidebar.slider("Downtrend threshold (%)", 0.1, 5.0, 1.0, step=0.1, disabled=(not trend_guard_enable), key="trainer_trend_down_thresh_pct")

# Time-stop per cycle: prevent long inventory hang by forcing more conservative exits after X hours
st.sidebar.subheader("Inventory & trend guards (interpretable)")

# Trend-guard (downtrend): block new BUYs in TREND-down regimes
trend_guard_enable = st.sidebar.checkbox(
    "Enable TREND downtrend guard (block BUYs)",
    value=True,
    help="When the effective regime is TREND and price has fallen over the lookback window beyond the threshold, new BUYs are blocked (SELLs still allowed)."
)
trend_lookback = st.sidebar.slider("Trend lookback (candles)", 8, 200, 48, step=4, disabled=(not trend_guard_enable))
trend_down_thresh_pct = st.sidebar.slider("Downtrend threshold (%)", 0.1, 5.0, 1.0, step=0.1, disabled=(not trend_guard_enable))

# Time-stop per cycle: prevent long inventory hang by forcing more conservative exits after X hours
enable_time_stop = st.sidebar.checkbox(
    "Enable time-stop per cycle",
    value=True,
    help="After a cycle has been open longer than X hours, apply an additional exit rule (e.g., net break-even) to reduce inventory hanging."
, key="trainer_enable_time_stop")
time_stop_hours = st.sidebar.slider("Time-stop age (hours)", 0.0, 72.0, 12.0, step=1.0, disabled=(not enable_time_stop), key="trainer_time_stop_hours")
time_stop_mode = st.sidebar.selectbox(
    "Time-stop mode",
    ["BREAK_EVEN_NET", "DECAY_TO_TP", "REDUCE_TO_TP"],
    index=0,
    disabled=(not enable_time_stop),
    help="BREAK_EVEN_NET: exit at net break-even (fees/slippage).\nDECAY_TO_TP: TP decays toward a floor as the cycle ages.\nREDUCE_TO_TP: after X hours use a lower fixed TP floor."
, key="trainer_time_stop_mode")
# Always define a floor value (used only for DECAY_TO_TP / REDUCE_TO_TP)
time_stop_tp_floor_pct = 0.25
if enable_time_stop and time_stop_mode != "BREAK_EVEN_NET":
    time_stop_tp_floor_pct = st.sidebar.slider("Time-stop TP floor (%)", 0.0, 2.0, 0.25, step=0.05, key="trainer_time_stop_tp_floor_pct")

st.sidebar.subheader("Regime hysteresis (stability)")
confirm_n = st.sidebar.slider(
    "Confirmations required",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    key="trainer_confirm_n",
    help="Effective regime changes only after N identical classifications (reduces churn).",
)
cooldown_candles = st.sidebar.slider(
    "Cooldown (candles)",
    min_value=0,
    max_value=500,
    value=35,
    step=1,
    key="trainer_cooldown_candles",
    help="Minimum candles between regime changes (additional hysteresis).",
)

rng_seed = st.sidebar.number_input("Random seed", min_value=0, value=1337, step=1)

st.sidebar.subheader("Best-overall selection")
global_best = st.sidebar.checkbox(
    "Global best across symbols",
    value=False,
    help="Als dit aanstaat, zoeken we één gedeelde regime-profielset die gemiddeld over alle geselecteerde symbols + folds het beste scoort. "
         "Als dit uitstaat, optimaliseren we per symbol apart (default).",
)
restarts = st.sidebar.slider(
    "Restarts (profile sets to try)",
    min_value=1,
    max_value=20,
    value=6,
    step=1,
    help="We trainen meerdere (gesamplede) profielsets met verschillende seeds en kiezen de beste op gemiddelde test-score over folds.",
)
min_test_trades_avg = st.sidebar.slider(
    "Min avg test trades (eligibility)",
    min_value=0,
    max_value=200,
    value=10,
    step=1,
    help="Voorkomt dat een 'best' profiel wint door nauwelijks te traden.",
)
max_test_dd_cap_pct = st.sidebar.slider(
    "Max test drawdown cap (%)",
    min_value=0.5,
    max_value=50.0,
    value=15.0,
    step=0.5,
    help="Alleen profielsets waarvan de worst-case test drawdown onder deze cap blijft, zijn eligible.",
)

# Additional promotion-quality gates (recommended)
require_positive_test_pnl = st.sidebar.checkbox(
    "Gate: require test PnL avg > 0",
    value=True,
    help="Reject profile sets that are net negative out-of-sample (after fees/slippage)."
)
min_worst_fold_test_pnl = st.sidebar.number_input(
    "Gate: worst fold test PnL (>=)",
    value=0.0,
    step=0.1,
    help="Reject if any fold has test PnL below this threshold."
)
min_test_trades_per_fold = st.sidebar.slider(
    "Gate: min test trades per fold",
    min_value=0,
    max_value=500,
    value=20,
    step=5,
    help="Reject if any test fold has fewer trades than this threshold."
)
use_median_tiebreak = st.sidebar.checkbox(
    "Use median test score as tiebreak",
    value=True,
)



st.sidebar.subheader("Objective weights (scoring)")
dd_penalty = st.sidebar.slider(
    "Drawdown penalty weight",
    min_value=0.0,
    max_value=500.0,
    value=100.0,
    step=5.0,
    key="trainer_dd_penalty",
    help="Higher penalizes drawdowns more strongly in the optimizer score."
)
trade_penalty = st.sidebar.slider(
    "Low-trade penalty weight",
    min_value=0.0,
    max_value=500.0,
    value=50.0,
    step=5.0,
    key="trainer_trade_penalty",
    help="Higher penalizes strategies with too few trades (helps avoid overfitting via inactivity)."
)

st.sidebar.subheader("Optimizer budget")
max_evals_per_regime = st.sidebar.number_input(
    "Max evals per regime",
    min_value=10,
    max_value=5000,
    value=250,
    step=10,
    key="trainer_max_evals_per_regime",
    help="Limits the number of candidate evaluations per regime (keeps training bounded)."
)

run = st.sidebar.button("▶ Train (multi-fold WF)", width="stretch")

if "trained_profiles" not in st.session_state:
    st.session_state.trained_profiles = None
if "trainer_report" not in st.session_state:
    st.session_state.trainer_report = None
if "trained_profiles_best" not in st.session_state:
    st.session_state.trained_profiles_best = None
if "trainer_fold_details" not in st.session_state:
    st.session_state.trainer_fold_details = None

# --- Persistence: reload last bundle if session resets (Streamlit reruns/reconnects)
store_dir = ensure_store_dir()
if "last_bundle_path" not in st.session_state:
    st.session_state.last_bundle_path = None

st.sidebar.subheader("Results persistence")
auto_reload_last = st.sidebar.checkbox(
    "Auto-reload last saved bundle",
    value=True,
    help="If the session reruns/reconnects, reload the most recent saved bundle so results remain visible."
)

bundles = list_bundles(store_dir)
selected_bundle = None
if bundles:
    default_idx = 0
    if st.session_state.last_bundle_path in bundles:
        default_idx = bundles.index(st.session_state.last_bundle_path)
    selected_bundle = st.sidebar.selectbox(
        "Load bundle from disk",
        options=bundles,
    index=default_idx,
        format_func=lambda p: os.path.basename(p),
    help="Select a previously saved bundle to view/inspect."
    )
    if st.sidebar.button("Load selected bundle", use_container_width=True):
        st.session_state._force_load_bundle = True

def _apply_loaded_bundle(bundle: dict) -> None:
    # Keep display compatible with existing UI
    st.session_state.trained_profiles = bundle.get("profiles")
    st.session_state.trained_profiles_best = bundle.get("profiles")
    meta = bundle.get("meta", {}) or {}
    st.session_state.last_bundle_path = bundle.get("path") or st.session_state.last_bundle_path

    # Optional: restore report/fold tables if present
    rep = meta.get("trainer_report_rows")
    if rep:
        try:
            st.session_state.trainer_report = pd.DataFrame(rep)
        except Exception:
            pass
    folds = meta.get("fold_rows") or meta.get("trainer_fold_rows")
    if folds:
        try:
            st.session_state.trainer_fold_details = pd.DataFrame(folds)
        except Exception:
            pass

# Auto reload on cold start / session reset
if auto_reload_last and (st.session_state.trained_profiles is None) and bundles:
    try:
        bpath = selected_bundle or bundles[0]
        b = load_bundle(bpath)
        b["path"] = bpath
        _apply_loaded_bundle(b)
    except Exception:
        pass

# Manual forced reload
if st.session_state.get("_force_load_bundle", False) and selected_bundle:
    try:
        b = load_bundle(selected_bundle)
        b["path"] = selected_bundle
        _apply_loaded_bundle(b)
    finally:
        st.session_state._force_load_bundle = False



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
    """Build rolling walk-forward folds.

    Common NO_FOLDS causes:
      - not enough history for min_train_days + test_window_days
      - step/test windows too big vs lookback
      - timezone / missing timestamps (handled upstream)
    """
    df = df.dropna().copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return [], {"reason": "EMPTY_DF"}

    end_ts = df["timestamp"].max()
    start_ts = df["timestamp"].min()

    need_days = int(min_train_days) + int(test_window_days)
    have_days = max(0.0, (end_ts - start_ts) / pd.Timedelta(days=1))

    # If user asks for more lookback than exists, just use what we have.
    if have_days < need_days:
        return [], {
            "reason": "INSUFFICIENT_HISTORY",
            "have_days": float(have_days),
            "need_days": float(need_days),
            "start": str(start_ts),
            "end": str(end_ts),
        }

    fold_list = []
    # Build folds from the end backwards
    for k in range(int(folds)):
        test_end = end_ts - pd.Timedelta(days=int(step_days) * k)
        test_start = test_end - pd.Timedelta(days=int(test_window_days))
        train_end = test_start
        train_start = train_end - pd.Timedelta(days=int(min_train_days))

        # Ensure bounds are within available data
        if train_start < start_ts:
            train_start = start_ts

        train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)].copy()
        test = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)].copy()

        # Require at least some rows in both sets
        if len(train) < 50 or len(test) < 20:
            continue

        fold_list.append((train_start, train_end, test_start, test_end, train, test))

    if not fold_list:
        return [], {
            "reason": "WINDOWS_TOO_STRICT",
            "have_days": float(have_days),
            "need_days": float(need_days),
            "start": str(start_ts),
            "end": str(end_ts),
            "hint": "Reduce min_train_days / test_window_days / folds, or increase lookback_days / use larger timeframe.",
        }

    return list(reversed(fold_list)), {"reason": "OK", "folds": len(fold_list), "start": str(start_ts), "end": str(end_ts)}



if run:
    prog = st.progress(0, text='Training...')
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
        "bb_mr_enable": bool(bb_mr_enable),
        "bb_mr_window": int(bb_mr_window),
        "bb_mr_z": float(bb_mr_z),
        "trend_guard_enable": bool(trend_guard_enable),
        "trend_lookback": int(trend_lookback),
        "trend_down_thresh_pct": float(trend_down_thresh_pct),
        "enable_time_stop": bool(enable_time_stop),
        "time_stop_hours": float(time_stop_hours),
        "time_stop_mode": str(time_stop_mode),
        "time_stop_tp_floor_pct": float(time_stop_tp_floor_pct),
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

    # If global_best: build folds per symbol first, then select ONE shared profile set.
    if global_best:
        sym_data = {}
        sym_dbg = {}
        for i, sym in enumerate(symbols):
            prog.progress(i / max(1, len(symbols)), text=f"Loading data for {sym}...")
            since = pd.Timestamp.utcnow() - pd.Timedelta(days=int(lookback_days))
            df = load_or_fetch(sym, timeframe=timeframe, since=since, until=None, force_refresh=force_refresh)
            if df is None or df.empty:
                sym_dbg[sym] = {"status": "NO_DATA"}
                continue

            folds_data, folds_dbg = _rolling_folds(df)
            if not folds_data:
                # fallback percent split 70/30
                n = len(df)
                if n >= 100:
                    n_test = max(20, int(round(n * 0.30)))
                    train_df = df.iloc[: max(50, n - n_test)].copy()
                    test_df = df.iloc[max(50, n - n_test):].copy()
                    folds_data = [(
                        train_df["timestamp"].min(), train_df["timestamp"].max(),
                        test_df["timestamp"].min(), test_df["timestamp"].max(),
                        train_df, test_df
                    )]
                    folds_dbg = {"reason": "FALLBACK_PERCENT_SPLIT", **(folds_dbg or {})}
                else:
                    sym_dbg[sym] = {"status": "NO_FOLDS", **(folds_dbg or {})}
                    continue

            sym_data[sym] = {"folds": folds_data}
            sym_dbg[sym] = {"status": "OK", **(folds_dbg or {})}

        if not sym_data:
            report_rows.append({"symbol": "GLOBAL", "status": "NO_DATA_OR_FOLDS"})
        else:
            primary = next(iter(sym_data.keys()))
            primary_folds = sym_data[primary]["folds"]
            train_start, train_end, test_start, test_end, train_df, _ = primary_folds[-1]

            def _eval_profiles_global(profiles_i):
                test_scores, test_pnls, test_dds, test_trades = [], [], [], []
                for sym2, blob in sym_data.items():
                    folds2 = blob["folds"]
                    for fidx, (tr_s, tr_e, te_s, te_e, tr_df2, te_df) in enumerate(folds2):
                        dfs = {sym2: te_df}
                        pair_cfg = {sym2: dict(base_cfg)}
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
                            regime_profiles=profiles_i,
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
                        test_trades.append(_get_num_trades(test_summ))

                score_avg = float(pd.Series(test_scores).mean()) if test_scores else float("nan")
                score_med = float(pd.Series(test_scores).median()) if test_scores else float("nan")
                pnl_avg = float(pd.Series(test_pnls).mean()) if test_pnls else 0.0
                dd_worst = float(pd.Series(test_dds).max()) if test_dds else 1.0
                dd_avg = float(pd.Series(test_dds).mean()) if test_dds else 1.0
                trades_avg = float(pd.Series(test_trades).mean()) if test_trades else 0.0
                return score_avg, score_med, pnl_avg, dd_worst, dd_avg, trades_avg

            cand_rows = []
            best_profiles = None
            best_test_score = None
            best_test_score_med = None
            best_dbg = None

            for r in range(int(restarts)):
                seed_i = int(rng_seed) + 1000 * r
                prog.progress(0.25, text=f"Optimize GLOBAL (restart {r+1}/{restarts}) on {primary}...")

                profiles_i, best_train_i = staged_optimize_regime_profiles(
                    sym=primary,
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
                    max_evals_per_regime=int(max_evals_per_regime),
                    seed=int(seed_i),
                    progress_cb=None,
                )

                score_avg, score_med, pnl_avg, dd_worst, dd_avg, trades_avg = _eval_profiles_global(profiles_i)
                eligible = (trades_avg >= float(min_test_trades_avg)) and ((dd_worst * 100.0) <= float(max_test_dd_cap_pct))

                cand_rows.append({
                    "restart": r + 1,
                    "seed": seed_i,
                    "eligible": bool(eligible),
                    "global_test_score_avg": score_avg,
                    "global_test_score_med": score_med,
                    "global_test_pnl_avg": pnl_avg,
                    "global_test_dd_worst_pct": dd_worst * 100.0,
                    "global_test_dd_avg_pct": dd_avg * 100.0,
                    "global_test_trades_avg": trades_avg,
                    "primary_symbol": primary,
                })

                if eligible:
                    better = False
                    if best_test_score is None or score_avg > best_test_score:
                        better = True
                    elif best_test_score is not None and score_avg == best_test_score and use_median_tiebreak:
                        if best_test_score_med is None or score_med > best_test_score_med:
                            better = True

                    if better:
                        best_profiles = profiles_i
                        best_test_score = score_avg
                        best_test_score_med = score_med
                        best_dbg = {"seed": seed_i, "restart": r + 1, "score_avg": score_avg, "score_med": score_med, "dd_worst_pct": dd_worst*100.0, "trades_avg": trades_avg}

            cand_df = pd.DataFrame(cand_rows).sort_values(["eligible", "global_test_score_avg", "global_test_score_med"], ascending=[False, False, False])
            with st.expander("GLOBAL best-overall candidates", expanded=True):
                st.dataframe(cand_df, use_container_width=True, height=280)
                st.caption(f"Symbols used: {', '.join(sym_data.keys())}")
                if best_dbg:
                    st.caption(
                        f"Selected GLOBAL: restart {best_dbg['restart']} (seed {best_dbg['seed']}) | "
                        f"score avg {best_dbg['score_avg']:.3f} | median {best_dbg['score_med']:.3f} | "
                        f"worst DD {best_dbg['dd_worst_pct']:.2f}% | avg trades {best_dbg['trades_avg']:.1f}"
                    )
                else:
                    st.warning("No eligible GLOBAL candidate found under current constraints; falling back to best by score.")

            if best_profiles is None and not cand_df.empty:
                best_profiles = None
                best_profiles_seed = int(cand_df.iloc[0]["seed"])
                best_profiles, _ = staged_optimize_regime_profiles(
                    sym=primary,
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
                    max_evals_per_regime=int(max_evals_per_regime),
                    seed=int(best_profiles_seed),
                    progress_cb=None,
                )

            # Apply same profiles to all symbols
            for sym2 in sym_data.keys():
                trained[sym2] = {
                    "use_regime_profiles": True,
                    "regime_profile_rebuild": False,
                    "regime_profiles": best_profiles,
                }

            score_avg, score_med, pnl_avg, dd_worst, dd_avg, trades_avg = _eval_profiles_global(best_profiles)
            report_rows.append({
                "symbol": "GLOBAL",
                "status": "OK",
                "primary_symbol": primary,
                "symbols_used": ", ".join(sym_data.keys()),
                "test_score_avg": score_avg,
                "test_score_med": score_med,
                "test_total_pnl_avg": pnl_avg,
                "test_max_dd_worst_pct": dd_worst * 100.0,
                "test_max_dd_avg_pct": dd_avg * 100.0,
                "test_trades_avg": trades_avg,
                "folds_used_total": sum(len(v["folds"]) for v in sym_data.values()),
            })

            with st.expander("GLOBAL fold debug per symbol", expanded=False):
                st.json(sym_dbg)

    else:
        for i, sym in enumerate(symbols):
                prog.progress(i / max(1, len(symbols)), text=f"Loading data for {sym}...")
                since = pd.Timestamp.utcnow() - pd.Timedelta(days=int(lookback_days))
                df = load_or_fetch(sym, timeframe=timeframe, since=since, until=None, force_refresh=force_refresh)
                if df is None or df.empty:
                    report_rows.append({"symbol": sym, "status": "NO_DATA"})
                    continue

                folds_data, folds_dbg = _rolling_folds(df)
                if not folds_data:
                    # Fallback: percent split 70/30 so we still produce a result if possible
                    n = len(df)
                    if n >= 100:
                        n_test = max(20, int(round(n * 0.30)))
                        train_df = df.iloc[: max(50, n - n_test)].copy()
                        test_df = df.iloc[max(50, n - n_test):].copy()
                        folds_data = [(
                            train_df["timestamp"].min(), train_df["timestamp"].max(),
                            test_df["timestamp"].min(), test_df["timestamp"].max(),
                            train_df, test_df
                        )]
                        folds_dbg = {"reason": "FALLBACK_PERCENT_SPLIT", **(folds_dbg or {})}
                    else:
                        report_rows.append({"symbol": sym, "status": "NO_FOLDS", **(folds_dbg or {})})
                        continue

                # Optional debug panel
                with st.expander(f"{sym} fold debug", expanded=False):
                    st.json(folds_dbg)

                # Optimize on the most recent fold's train set (fast), evaluate across all folds
                train_start, train_end, test_start, test_end, train_df, _ = folds_data[-1]

                prog.progress((i + 0.2) / max(1, len(symbols)), text=f"Optimize train profiles for {sym} (latest fold)...")
                status_ph = st.empty()

                def progress_cb(regime: str, done: int, total: int):
                    status_ph.info(f"{sym} | optimizing {regime}: {done}/{total} evals")
                # --- Best-overall selection: try multiple restarts (different seeds),
                # pick the profile-set that maximizes average test score over folds, with stability filters.
                cand_rows = []
                best_profiles = None
                best_train = None
                best_test_score = None
                best_test_score_med = None
                best_dbg = None

                def _eval_profiles_on_folds(profiles_i):
                    test_scores = []
                    test_pnls = []
                    test_dds = []
                    test_trades = []
                    for fidx, (tr_s, tr_e, te_s, te_e, tr_df2, te_df) in enumerate(folds_data):
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
                            regime_profiles=profiles_i,
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
                        test_trades.append(_get_num_trades(test_summ))
                    score_avg = float(pd.Series(test_scores).mean()) if test_scores else float("nan")
                    score_med = float(pd.Series(test_scores).median()) if test_scores else float("nan")
                    pnl_avg = float(pd.Series(test_pnls).mean()) if test_pnls else 0.0
                    dd_worst = float(pd.Series(test_dds).max()) if test_dds else 1.0
                    dd_avg = float(pd.Series(test_dds).mean()) if test_dds else 1.0
                    trades_avg = float(pd.Series(test_trades).mean()) if test_trades else 0.0
                    return score_avg, score_med, pnl_avg, dd_worst, dd_avg, trades_avg

                for r in range(int(restarts)):
                    seed_i = int(rng_seed) + 1000 * r

                    # Train on latest fold's train set (fast), sampled within each regime
                    profiles_i, best_train_i = staged_optimize_regime_profiles(
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
                        max_evals_per_regime=int(max_evals_per_regime),
                        seed=int(seed_i),
                        progress_cb=None,
                    )

                    score_avg, score_med, pnl_avg, dd_worst, dd_avg, trades_avg = _eval_profiles_on_folds(profiles_i)
                    eligible = (trades_avg >= float(min_test_trades_avg)) and ((dd_worst * 100.0) <= float(max_test_dd_cap_pct))

                    cand_rows.append({
                        "restart": r + 1,
                        "seed": seed_i,
                        "eligible": bool(eligible),
                        "test_score_avg": score_avg,
                        "test_score_med": score_med,
                        "test_pnl_avg": pnl_avg,
                        "test_dd_worst_pct": dd_worst * 100.0,
                        "test_dd_avg_pct": dd_avg * 100.0,
                        "test_trades_avg": trades_avg,
                        "train_total_pnl": float(best_train_i.get("total_pnl", 0.0)),
                        "train_max_dd_pct": float(best_train_i.get("max_drawdown", 0.0)) * 100.0,
                        "train_trades": _get_num_trades(best_train_i),
                    })

                    # Select best eligible
                    if eligible:
                        better = False
                        if best_test_score is None or score_avg > best_test_score:
                            better = True
                        elif best_test_score is not None and score_avg == best_test_score and use_median_tiebreak:
                            if best_test_score_med is None or score_med > best_test_score_med:
                                better = True

                        if better:
                            best_profiles = profiles_i
                            best_train = best_train_i
                            best_test_score = score_avg
                            best_test_score_med = score_med
                            best_dbg = {"seed": seed_i, "restart": r + 1, "score_avg": score_avg, "score_med": score_med, "dd_worst_pct": dd_worst*100.0, "trades_avg": trades_avg}

                cand_df = pd.DataFrame(cand_rows).sort_values(["eligible", "test_score_avg", "test_score_med"], ascending=[False, False, False])
                with st.expander(f"{sym} best-overall candidates", expanded=False):
                    st.dataframe(cand_df, use_container_width=True, height=260)
                    if best_dbg:
                        st.caption(
                            f"Selected: restart {best_dbg['restart']} (seed {best_dbg['seed']}) | "
                            f"test score avg {best_dbg['score_avg']:.3f} | median {best_dbg['score_med']:.3f} | "
                            f"worst DD {best_dbg['dd_worst_pct']:.2f}% | avg trades {best_dbg['trades_avg']:.1f}"
                        )
                    else:
                        st.warning(
                            "No eligible candidates found under the current constraints. "
                            "Consider lowering 'Min avg test trades' or increasing 'Max test DD cap'."
                        )

                # If none eligible: fall back to the best by score_avg (even if ineligible) so you still get profiles.json
                if best_profiles is None and not cand_df.empty:
                    best_row = cand_df.iloc[0]
                    fallback_seed = int(best_row["seed"])
                    best_profiles, best_train = staged_optimize_regime_profiles(
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
                        max_evals_per_regime=int(max_evals_per_regime),
                        seed=int(fallback_seed),
                        progress_cb=None,
                    )

                profiles = best_profiles
                best_train = best_train if best_train is not None else {}
                # Evaluate across folds

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
# Guard / filter settings (from base_cfg)
"bb_mr_enable": bool(base_cfg.get("bb_mr_enable", False)),
"bb_mr_window": int(base_cfg.get("bb_mr_window", 20)),
"bb_mr_z": float(base_cfg.get("bb_mr_z", 0.0)),
"trend_guard_enable": bool(base_cfg.get("trend_guard_enable", False)),
"trend_lookback": int(base_cfg.get("trend_lookback", 0)),
"trend_down_thresh_pct": float(base_cfg.get("trend_down_thresh_pct", 0.0)),
"enable_time_stop": bool(base_cfg.get("enable_time_stop", False)),
"time_stop_hours": float(base_cfg.get("time_stop_hours", 0.0)),
"time_stop_mode": str(base_cfg.get("time_stop_mode", "")),
"time_stop_tp_floor_pct": float(base_cfg.get("time_stop_tp_floor_pct", 0.0)),
                    "test_score_avg": float(pd.Series(test_scores).mean()),
                    "test_score_med": float(pd.Series(test_scores).median()),
                    "test_total_pnl_avg": float(pd.Series(test_pnls).mean()),
                    "test_max_dd_pct_avg": float(pd.Series(test_dds).mean()) * 100.0,
                    "folds_used": len(test_scores),
                })

    prog.progress(1.0, text="Done.")
    st.session_state.trained_profiles = trained
    st.session_state.trained_profiles_best = trained
    st.session_state.trainer_report = pd.DataFrame(report_rows)
    # --- Governance export: bundle with metadata + data hashes (saved to disk)
    store_dir = ensure_store_dir()
    data_hashes = {}
    try:
        if "df_cache" in locals():
            for sym_k, df_ in df_cache.items():
                data_hashes[str(sym_k).upper()] = stable_hash_df(df_)
    except Exception:
        pass

    meta = {
        "mode": "GLOBAL" if global_best else "PER_SYMBOL",
        "symbols": [str(s).upper() for s in symbols],
        "timeframe": timeframe,
        "lookback_days": int(lookback_days),
        "folds": int(folds),
        "test_window_days": int(test_window_days),
        "step_days": int(step_days),
        "min_train_days": int(min_train_days),
        "max_evals_per_regime": int(max_evals_per_regime),
        "restarts": int(restarts),
        "rng_seed": int(rng_seed),
        "fees": {"maker": float(maker_fee), "taker": float(taker_fee), "slippage": float(slippage), "mode": str(fee_mode)},
        "git_commit": ((_git_commit() if "_git_commit" in globals() else "") if "_git_commit" in globals() else ""),
        "data_hashes": data_hashes,
    }

    # --- Evaluate gates for this bundle (used by Profile Manager promotion)
    gates = {
        "require_positive_test_pnl": bool(require_positive_test_pnl),
        "min_worst_fold_test_pnl": float(min_worst_fold_test_pnl),
        "min_test_trades_per_fold": int(min_test_trades_per_fold),
        "min_test_trades_avg": int(min_test_trades_avg),
        "max_test_dd_cap_pct": float(max_test_dd_cap_pct),
    }
    gate_fail = []

    # High-level gates from trainer_report
    try:
        rep = st.session_state.trainer_report
        if rep is not None and not rep.empty:
            if "symbol" in rep.columns and (rep["symbol"] == "GLOBAL").any():
                row = rep[rep["symbol"] == "GLOBAL"].iloc[0]
                test_pnl_avg = float(row.get("test_total_pnl_avg", row.get("global_test_pnl_avg", 0.0)))
                test_dd_worst = float(row.get("test_max_dd_worst_pct", row.get("global_test_dd_worst_pct", 0.0)))
                test_trades_avg_val = float(row.get("test_trades_avg", row.get("global_test_trades_avg", 0.0)))
            else:
                test_pnl_avg = float(rep["test_total_pnl_avg"].sum()) if "test_total_pnl_avg" in rep.columns else float(rep.get("test_pnl_avg", 0.0).sum()) if "test_pnl_avg" in rep.columns else 0.0
                test_dd_worst = float(rep["test_max_dd_worst_pct"].max()) if "test_max_dd_worst_pct" in rep.columns else float(rep.get("test_dd_worst_pct", 0.0).max()) if "test_dd_worst_pct" in rep.columns else 0.0
                test_trades_avg_val = float(rep["test_trades_avg"].mean()) if "test_trades_avg" in rep.columns else 0.0

            if gates["require_positive_test_pnl"] and (test_pnl_avg <= 0.0):
                gate_fail.append("NEG_TEST_PNL_AVG")
            if test_trades_avg_val < gates["min_test_trades_avg"]:
                gate_fail.append("MIN_TRADES_AVG")

            dd_pct = (test_dd_worst * 100.0) if (0.0 <= test_dd_worst <= 1.0) else test_dd_worst
            if dd_pct > gates["max_test_dd_cap_pct"]:
                gate_fail.append("DD_CAP")
    except Exception:
        pass

    # Fold-level gates from fold_rows
    try:
        worst_fold_pnl = None
        min_fold_trades = None
        for r in fold_rows:
            pnl = r.get("test_total_pnl", r.get("test_pnl", r.get("pnl_test", None)))
            trades = r.get("test_trades", r.get("trades", r.get("n_test_trades", None)))
            if pnl is not None:
                worst_fold_pnl = float(pnl) if worst_fold_pnl is None else min(worst_fold_pnl, float(pnl))
            if trades is not None:
                min_fold_trades = int(trades) if min_fold_trades is None else min(min_fold_trades, int(trades))

        if worst_fold_pnl is not None and worst_fold_pnl < gates["min_worst_fold_test_pnl"]:
            gate_fail.append("WORST_FOLD_PNL")
        if min_fold_trades is not None and min_fold_trades < gates["min_test_trades_per_fold"]:
            gate_fail.append("MIN_TRADES_PER_FOLD")
    except Exception:
        pass

    meta["gates"] = gates
    meta["gates_passed"] = (len(gate_fail) == 0)
    meta["gates_failed"] = gate_fail

    bundle = make_bundle(trained, meta)
    default_name = f"bundle_{meta['mode'].lower()}_{timeframe}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
    saved_path = save_bundle(bundle, store_dir=store_dir, name=default_name)

    st.success(f"Saved profile bundle: {saved_path}")
    st.download_button(
        "Download bundle JSON",
        data=json.dumps(bundle, indent=2).encode("utf-8"),
        file_name=Path(saved_path).name,
        mime="application/json",
        use_container_width=True,
    )
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
