Fix: Rewrote core/grid/engine.py from scratch to eliminate indentation corruption.

- Consistent 4-space indentation
- No tabs
- Provides GridEngine.check_price, _next, _prev, reset_open_cycles
- Supports buy_guard + record_blocked logging
- Supports per-cycle take-profit (CYCLE_TP)
