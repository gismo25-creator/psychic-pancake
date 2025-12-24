# ROADMAP

This roadmap tracks the evolution of the Bitvavo grid trading research/simulation app, including regime-aware parameter profiles and governance.

## Current status (implemented)

### Core trading & simulation
- Bitvavo (spot) simulation framework with fees, slippage, and per-asset exposure caps.
- Multi-pair grid trading with per-pair settings.
- Linear and Fibonacci grids.
- Regime metrics (ATR, realized vol, Bollinger bandwidth, ADX) + regime classification with hysteresis/cooldown.
- Dynamic spacing (range/levels adjustments by regime).
- Trade table under chart with realized PnL per trade and markers on chart.
- Risk controls:
  - Max open exposure per asset
  - Portfolio drawdown stop + per-asset stop (avg-entry %, optional ATR)
  - STOP & FLATTEN (panic) + auto-pause + resume confirmation
  - Per-pair pause/resume
  - Max concurrent assets in drawdown + correlation filter
- Cycle take-profit improvements:
  - TP sells at TP price as limit (less optimistic simulation).

### Backtesting & training
- Backtest page with decision log and results persistence across UI changes.
- Offline trainer:
  - Walk-forward training
  - Multi-fold training (bounded execution to avoid infinite runs)
  - Per-symbol regime profiles and optional **Global best across symbols** mode.

### Governance (profiles)
- Bundle export from Trainer with metadata (mode, timeframe, folds, seeds, fees, data hashes).
- Profile Manager page:
  - Validate bundles + diff view
  - Sanity backtest gate before apply/promotion
  - Apply bundle to session
  - Promote to ACTIVE (`data/profiles/active.json`) with versioned history
  - Rollback ACTIVE from history
  - Audit log (`data/profiles/audit_log.jsonl`)
  - Compare Candidate vs ACTIVE (meta + diff + optional sanity side-by-side)
  - Load ACTIVE into session

## Next steps (recommended)

### 1) Governance hardening
- Add explicit **promotion gates** in UI:
  - Min trades per symbol (already present in compare; integrate consistently into main sanity gate)
  - Worst-symbol DD cap
  - No-trade guard (already present) + "min realized PnL" optional
- Add "ACTIVE provenance" display in Live/Simulation:
  - Show active bundle name + created_at + mode (read-only)
- Add "export report" from Trainer/Profile Manager:
  - Save summary as CSV/JSON for reproducibility.

### 2) Live readiness (non-black-box, controlled)
- Add a **dry-run execution mode** for live (no orders) that uses live prices but logs intended orders.
- Add Bitvavo live order executor behind an explicit enable switch:
  - API key handling via Streamlit secrets
  - Explicit risk acknowledgment and safe defaults
  - Rate-limit handling and retry strategy
- Add robust state persistence for live:
  - Restore open cycles and avg-entry from broker state on restart.

### 3) Research improvements (interpretable)
- Regime-conditional parameter sets improvements:
  - Better regime duration weighting
  - Volatility clustering-aware adjustments
- Parameter optimization:
  - Optimize range/levels/order-size under constraints
  - Stability scoring (variance across folds) as a primary objective.

### 4) Portfolio analytics
- Add portfolio attribution:
  - Per-symbol contribution to PnL and DD
  - Exposure timeline and heatmaps
- Add transaction cost breakdown:
  - Fees vs slippage vs adverse selection proxy.

### 5) Quality / engineering
- Add unit tests for:
  - GridEngine cycle accounting
  - PortfolioSimulatorTrader equity/pnl consistency
  - Governance validation + audit log writing
- Add CI checks:
  - Lint + formatting guard
  - Basic import tests for Streamlit pages.

## Notes
- This app is designed to stay interpretable: regimes and parameter changes must remain explainable.
- Live execution should remain opt-in with layered safety gates.
