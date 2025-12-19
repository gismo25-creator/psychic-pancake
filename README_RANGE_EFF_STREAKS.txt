Added metrics:
1) Range efficiency (grid hit-rate):
   - % of grid levels touched by candle ranges (low<=level<=high) over a rolling window.
2) Win/loss streaks:
   - Based on realized PnL per closed cycle (eng.closed_cycles), with configurable scope.
   - Exposes win-rate, current streak, max win streak, max loss streak.

Controls in sidebar:
- Hit-rate window (candles)
- Streak scope (closed cycles)
