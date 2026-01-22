from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import itertools
import random
import time

import pandas as pd

from core.backtest.replay import run_backtest
from core.backtest.metrics import summarize_run


@dataclass
class SearchSpace:
    range_pcts: List[float]
    levels: List[int]
    order_size_mults: List[float]
    cycle_tp_enable: List[bool]
    cycle_tp_pcts: List[float]


def _num_trades(summ: Dict) -> float:
    if "num_trades" in summ:
        return float(summ.get("num_trades") or 0.0)
    return float(summ.get("n_trades") or 0.0)


def objective_from_summary(
    summ: Dict,
    dd_penalty: float = 3.0,
    trade_penalty: float = 0.0,
) -> float:
    """Interpretable objective: maximize total pnl, penalize max drawdown (and optionally low activity)."""
    total_pnl = float(summ.get("total_pnl", 0.0))
    max_dd = float(summ.get("max_drawdown", 0.0))  # fraction 0..1
    trades = _num_trades(summ)
    # penalize DD relative to pnl magnitude (avoid optimizing by just not trading)
    dd_term = dd_penalty * (max_dd * max(1.0, abs(total_pnl) + 1.0))
    trade_term = trade_penalty * (1.0 / max(1.0, trades))
    return total_pnl - dd_term - trade_term


def _candidate_tuples(
    base_cfg: Dict,
    best_reg_profile: Dict,
    search: SearchSpace,
) -> Tuple[List[Tuple[float, Optional[int], float, bool, float]], int]:
    levels_iter = search.levels if base_cfg.get("grid_type", "Linear") == "Linear" else [None]
    tuples: List[Tuple[float, Optional[int], float, bool, float]] = []
    for r_pct in search.range_pcts:
        for lv in levels_iter:
            for osm in search.order_size_mults:
                for ctp_en in search.cycle_tp_enable:
                    if ctp_en:
                        for ctp in search.cycle_tp_pcts:
                            tuples.append((float(r_pct), None if lv is None else int(lv), float(osm), True, float(ctp)))
                    else:
                        tuples.append((float(r_pct), None if lv is None else int(lv), float(osm), False, float(best_reg_profile.get("cycle_tp_pct", 0.35))))
    return tuples, len(tuples)


def staged_optimize_regime_profiles(
    sym: str,
    df: pd.DataFrame,
    base_cfg: Dict,
    base_profiles: Dict[str, Dict],
    *,
    timeframe: str,
    start_cash: float,
    maker_fee: float,
    taker_fee: float,
    slippage: float,
    fee_mode: str,
    quote_ccy: str,
    caps: Optional[Dict[str, float]] = None,
    confirm_n: int = 3,
    cooldown_candles: int = 0,
    dd_penalty: float = 3.0,
    trade_penalty: float = 0.0,
    search: Optional[SearchSpace] = None,
    max_evals_per_regime: Optional[int] = 150,
    seed: int = 1337,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[Dict[str, Dict], Dict]:
    """Optimize regime profiles with a bounded (optionally sampled) grid-search per regime.

    - If max_evals_per_regime is set, we randomly sample that many candidates per regime.
    - progress_cb(regime, done, total) is called periodically.
    """
    if search is None:
        search = SearchSpace(
            range_pcts=[0.6, 0.8, 1.0, 1.3, 1.6, 2.0],
            levels=[6, 8, 10, 12, 14, 16],
            order_size_mults=[0.5, 0.7, 0.8, 1.0, 1.2],
            cycle_tp_enable=[False, True],
            cycle_tp_pcts=[0.20, 0.35, 0.50, 0.80],
        )

    dfs = {sym: df}
    pair_cfg = {sym: dict(base_cfg)}

    # Start from current profiles
    current = {k: dict(v) for k, v in base_profiles.items()}

    # Baseline run
    trades_df, equity_curve, decision_log, trader = run_backtest(
        dfs=dfs,
        pair_cfg=pair_cfg,
        timeframe=timeframe,
        start_cash=float(start_cash),
        maker_fee=float(maker_fee),
        taker_fee=float(taker_fee),
        slippage=float(slippage),
        fee_mode=str(fee_mode),
        quote_ccy=str(quote_ccy),
        max_exposure_quote=caps or {},
        regime_profiles=current,
        enable_regime_profiles=True,
        confirm_n=int(confirm_n),
        cooldown_candles=int(cooldown_candles),
        rebuild_on_regime_change=False,
    )
    baseline = summarize_run(equity_curve, trades_df)

    best_overall = dict(baseline)
    best_overall["objective"] = objective_from_summary(baseline, dd_penalty=dd_penalty, trade_penalty=trade_penalty)

    regimes = ["RANGE", "TREND", "CHAOS", "WARMUP"]

    for reg in regimes:
        best_reg_profile = dict(current.get(reg, {}))
        best_reg_score = float(best_overall["objective"])
        best_reg_summary = dict(best_overall)

        cand_list, total = _candidate_tuples(base_cfg, best_reg_profile, search)

        # Sample if needed
        order = list(range(total))
        rng = random.Random(seed ^ hash((sym, reg)) & 0xFFFFFFFF)
        rng.shuffle(order)
        if max_evals_per_regime is not None:
            take = max(1, min(int(max_evals_per_regime), total))
            order = order[:take]

        for idx, oi in enumerate(order, start=1):
            r_pct, lv, osm, ctp_en, ctp = cand_list[oi]
            cand = dict(best_reg_profile)
            cand["range_pct"] = float(r_pct)
            if lv is not None:
                cand["levels"] = int(lv)
            cand["order_size_mult"] = float(osm)
            cand["cycle_tp_enable"] = bool(ctp_en)
            cand["cycle_tp_pct"] = float(ctp)

            tmp_profiles = {k: dict(v) for k, v in current.items()}
            tmp_profiles[reg] = cand

            trades_df, equity_curve, decision_log, trader = run_backtest(
                dfs=dfs,
                pair_cfg=pair_cfg,
                timeframe=timeframe,
                start_cash=float(start_cash),
                maker_fee=float(maker_fee),
                taker_fee=float(taker_fee),
                slippage=float(slippage),
                fee_mode=str(fee_mode),
                quote_ccy=str(quote_ccy),
                max_exposure_quote=caps or {},
                regime_profiles=tmp_profiles,
                enable_regime_profiles=True,
                confirm_n=int(confirm_n),
                cooldown_candles=int(cooldown_candles),
                rebuild_on_regime_change=False,
            )
            summ = summarize_run(equity_curve, trades_df)
            score = objective_from_summary(summ, dd_penalty=dd_penalty, trade_penalty=trade_penalty)

            if score > best_reg_score:
                best_reg_score = score
                best_reg_profile = cand
                best_reg_summary = dict(summ)
                best_reg_summary["objective"] = score

            if progress_cb is not None and (idx == 1 or idx % 20 == 0 or idx == len(order)):
                progress_cb(reg, idx, len(order))

        current[reg] = best_reg_profile
        best_overall = best_reg_summary

    return current, best_overall
