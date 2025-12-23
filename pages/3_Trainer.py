from pathlib import Path
import json
import pandas as pd
import streamlit as st

from core.profiles.registry import make_bundle, save_bundle, stable_hash_df, ensure_store_dir

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

st.sidebar.subheader("Training speed")
max_evals_per_regime = st.sidebar.slider(
    "Max evals per regime (sampled)",
    min_value=20,
    max_value=800,
    value=150,
    step=10,
    help="Begrenst het aantal backtests per regime door random sampling. Lager = sneller.",
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
use_median_tiebreak = st.sidebar.checkbox(
    "Use median test score as tiebreak",
    value=True,
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
