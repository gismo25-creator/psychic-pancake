Added: Per-pair / per-cycle take-profit (Cycle TP)

- New per-pair settings:
  - Cycle take-profit (per cycle) toggle
  - Cycle TP (%) threshold
- Logic:
  - For each open cycle, if price >= buy_price*(1+tp%), sell the cycle amount at current price.
  - Clears the corresponding grid sell level and reactivates the buy level.
  - Trades are logged with reason=CYCLE_TP and exact realized PnL.
