from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.exchange.simulator import PortfolioSimulatorTrader
from core.grid.engine import GridEngine
from core.grid.linear import generate_linear_grid
from core.grid.fibonacci import generate_fibonacci_grid

from core.ml.volatility import atr, realized_vol, bollinger_bandwidth, adx
from core.ml.regime import classify_regime



def _grid_for_cfg(price: float, cfg: Dict) -> List[float]:
    range_pct = float(cfg.get("base_range_pct", 1.0))
    lower = float(price) * (1.0 - range_pct / 100.0)
    upper = float(price) * (1.0 + range_pct / 100.0)
    gtype = str(cfg.get("grid_type", "Linear"))
    if gtype == "Linear":
        levels = int(cfg.get("base_levels", 10))
        return generate_linear_grid(lower, upper, levels)
    return generate_fibonacci_grid(lower, upper)


def run_backtest(
    dfs: Dict[str, pd.DataFrame],
    pair_cfg: Dict[str, Dict],
    timeframe: str,
    start_cash: float,
    maker_fee: float,
    taker_fee: float,
    slippage: float,
    fee_mode: str,
    quote_ccy: str,
    max_exposure_quote: Optional[Dict[str, float]] = None,
    regime_profiles: Optional[Dict[str, Dict]] = None,
    enable_regime_profiles: bool = False,
    confirm_n: int = 3,
    cooldown_candles: int = 0,
    rebuild_on_regime_change: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PortfolioSimulatorTrader]:
    """Multi-pair replay backtest (deterministic) on candle CLOSE prices.

    Step 2 additions:
    - Computes regime per candle from ATR%, RV, BB bandwidth, ADX.
    - Optional regime-conditional parameter sets (interpretable, rule-based).
    - Optional rebuild on regime change (closes position for the asset, rebuilds grid, resets cycles).

    Returns: (trades_df, equity_curve, decision_log_df, trader)
    """
    trader = PortfolioSimulatorTrader(
        cash_quote=float(start_cash),
        maker_fee=float(maker_fee),
        taker_fee=float(taker_fee),
        slippage=float(slippage),
        fee_mode=str(fee_mode),
        quote_ccy=str(quote_ccy),
        max_exposure_quote=max_exposure_quote or {},
    )

    engines: Dict[str, GridEngine] = {}
    mark_prices: Dict[str, float] = {}

    # Build event list: (timestamp, symbol, close)
    events: List[Tuple[pd.Timestamp, str, float]] = []
    for sym, df in dfs.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            px = float(row["close"])
            events.append((ts, sym, px))
    events.sort(key=lambda x: x[0])

    # Pre-compute indicators per symbol
    ind: Dict[str, pd.DataFrame] = {}
    for sym, df in dfs.items():
        if df is None or df.empty:
            continue
        d = df.copy().reset_index(drop=True)
        d["atr"] = atr(d, 14)
        d["rv"] = realized_vol(d, 30)
        d["bb"] = bollinger_bandwidth(d, 20, 2.0)
        d["adx"] = adx(d, 14)
        d["atr_pct"] = d["atr"] / d["close"].astype(float)
        d["raw_regime"] = [
            classify_regime(d, float(a) if a == a else float("nan"),
                            float(rv) if rv == rv else float("nan"),
                            float(bb) if bb == bb else float("nan"),
                            float(ad) if ad == ad else float("nan"))
            for a, rv, bb, ad in zip(d["atr_pct"], d["rv"], d["bb"], d["adx"])
        ]
        ind[sym] = d

    def _grid_for_profile(price: float, cfg: Dict, profile: Optional[Dict]) -> List[float]:
        range_pct = float((profile or {}).get("range_pct", cfg.get("base_range_pct", 1.0)))
        lower = float(price) * (1.0 - range_pct / 100.0)
        upper = float(price) * (1.0 + range_pct / 100.0)
        gtype = str(cfg.get("grid_type", "Linear"))
        if gtype == "Linear":
            levels = int((profile or {}).get("levels", cfg.get("base_levels", 10)))
            return generate_linear_grid(lower, upper, levels)
        return generate_fibonacci_grid(lower, upper)

    # Initialize engines using first seen price for each pair
    first_price: Dict[str, float] = {}
    for ts, sym, px in events:
        if sym not in first_price:
            first_price[sym] = px

    for sym, px in first_price.items():
        cfg = pair_cfg.get(sym, {})
        grid = _grid_for_profile(px, cfg, None)
        osize = float(cfg.get("order_size", 0.001))
        eng = GridEngine(sym, grid, osize)
        engines[sym] = eng

    # Regime hysteresis state per symbol
    regime_state: Dict[str, Dict] = {}
    last_row_idx: Dict[str, int] = {sym: -1 for sym in dfs.keys()}

    def _effective_regime(sym: str, raw: str, candle_idx: int) -> str:
        state = regime_state.setdefault(sym, {"hist": [], "eff": raw, "last_change_idx": candle_idx})
        # maintain hist length confirm_n
        state["hist"].append(raw)
        if len(state["hist"]) > int(confirm_n):
            state["hist"] = state["hist"][-int(confirm_n):]

        confirmed = (len(state["hist"]) == int(confirm_n)) and all(r == state["hist"][0] for r in state["hist"])
        if confirmed:
            cand = state["hist"][0]
            if cand != state["eff"]:
                if (candle_idx - int(state["last_change_idx"])) >= int(cooldown_candles):
                    state["eff"] = cand
                    state["last_change_idx"] = candle_idx
        return state["eff"]

    def _profile_for(reg: str) -> Optional[Dict]:
        if not enable_regime_profiles or not regime_profiles:
            return None
        return regime_profiles.get(reg)

    decision_rows = []
    equity_rows = []

    # Replay
    for ts, sym, px in events:
        mark_prices[sym] = px

        # Determine candle index for this symbol
        d = ind.get(sym)
        if d is None or d.empty:
            continue

        # Find matching index by timestamp progression (fast-ish sequential)
        i0 = last_row_idx.get(sym, -1) + 1
        idx = None
        for j in range(i0, min(len(d), i0 + 5_000)):  # safety search window
            if pd.Timestamp(d.loc[j, "timestamp"]) == ts:
                idx = j
                break
        if idx is None:
            # fallback: search exact match
            m = d.index[d["timestamp"].astype("datetime64[ns]") == ts.to_datetime64()]
            if len(m) == 0:
                continue
            idx = int(m[0])

        last_row_idx[sym] = idx
        raw_reg = str(d.loc[idx, "raw_regime"])
        eff_reg = _effective_regime(sym, raw_reg, idx)

        cfg = pair_cfg.get(sym, {})
        profile = _profile_for(eff_reg)

        # Apply profile to engine (interpretable)
        eng = engines.get(sym)
        if eng is None:
            continue

        base_order_size = float(cfg.get("order_size", 0.001))
        os_mult = float((profile or {}).get("order_size_mult", 1.0))
        eng.order_size = base_order_size * os_mult

        eng.enable_cycle_tp = bool((profile or {}).get("cycle_tp_enable", cfg.get("cycle_tp_enable", False)))
        eng.cycle_tp_pct = float((profile or {}).get("cycle_tp_pct", cfg.get("cycle_tp_pct", 0.35)))

        # Optional rebuild on regime change: close position and reset grid/cycles
        if rebuild_on_regime_change:
            state = regime_state.get(sym, {})
            # If the effective regime just changed on this idx, state["last_change_idx"] == idx
            if state.get("last_change_idx") == idx and idx != 0:
                base = sym.split("/")[0]
                amt = float(trader.positions.get(base, 0.0))
                if amt > 1e-12:
                    trader.sell(sym, float(px), amt, ts, reason="REGIME_REBUILD_FLATTEN")
                # rebuild grid and reset cycles
                eng.grid = sorted(_grid_for_profile(px, cfg, profile))
                eng.reset_open_cycles()

        # Execute
        eng.check_price(px, trader, ts, allow_buys=True)

        # Log decision (interpretable)
        used_range_pct = float((profile or {}).get("range_pct", cfg.get("base_range_pct", 1.0)))
        used_levels = (int((profile or {}).get("levels", cfg.get("base_levels", 10)))
                       if str(cfg.get("grid_type", "Linear")) == "Linear" else None)

        decision_rows.append({
            "timestamp": ts,
            "symbol": sym,
            "price": float(px),
            "raw_regime": raw_reg,
            "eff_regime": eff_reg,
            "profile_enabled": bool(enable_regime_profiles),
            "range_pct": used_range_pct,
            "levels": used_levels,
            "order_size_base": base_order_size,
            "order_size_mult": os_mult,
            "order_size_eff": float(eng.order_size),
            "cycle_tp_enable": bool(getattr(eng, "enable_cycle_tp", False)),
            "cycle_tp_pct": float(getattr(eng, "cycle_tp_pct", 0.35)),
        })

        eq = trader.equity(mark_prices)
        equity_rows.append({
            "timestamp": ts,
            "equity": eq,
            "cash": float(trader.cash),
        })

    equity_curve = pd.DataFrame(equity_rows)
    equity_curve = equity_curve.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Aggregate trades from engines
    all_trades = []
    for sym, eng in engines.items():
        for t in eng.trades:
            all_trades.append(t)

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("time").reset_index(drop=True)

    decision_log = pd.DataFrame(decision_rows)
    if not decision_log.empty:
        decision_log = decision_log.sort_values("timestamp").reset_index(drop=True)

    return trades_df, equity_curve, decision_log, trader
