from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def objective_from_summary(
    summ: Dict,
    dd_penalty: float = 3.0,
    trade_penalty: float = 0.0,
) -> float:
    """Interpretable objective: maximize total pnl, penalize max drawdown (and optionally low activity)."""
    total_pnl = float(summ.get("total_pnl", 0.0))
    max_dd = float(summ.get("max_drawdown", 0.0))  # fraction 0..1
    trades = float(summ.get("n_trades", 0.0))
    return total_pnl - dd_penalty * (max_dd * max(1.0, abs(total_pnl) + 1.0)) - trade_penalty * (1.0 / max(1.0, trades))


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
) -> Tuple[Dict[str, Dict], Dict]:
    """Optimize regime profiles in a staged manner (one regime at a time), keeping it interpretable.

    This avoids a huge combinatorial grid-search across all regimes.
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

    # Baseline run (with current)
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
        best_reg_score = best_overall["objective"]
        best_reg_summary = dict(best_overall)

        for r_pct in search.range_pcts:
            for lv in (search.levels if base_cfg.get("grid_type", "Linear") == "Linear" else [None]):
                for osm in search.order_size_mults:
                    for ctp_en in search.cycle_tp_enable:
                        for ctp in (search.cycle_tp_pcts if ctp_en else [best_reg_profile.get("cycle_tp_pct", 0.35)]):
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

        current[reg] = best_reg_profile
        best_overall = best_reg_summary

    return current, best_overall
