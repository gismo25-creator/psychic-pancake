Portfolio Risk Layer v1 adds:
1) Max concurrent assets-in-drawdown:
   - computes per-asset drawdown vs avg entry
   - blocks new BUYs when dd_assets_count >= max_assets_in_dd

2) Correlation filter (rolling):
   - computes rolling correlation of log returns per selected pair
   - blocks BUY when corr(candidate, held_asset) >= threshold

3) Equity-based position scaling:
   - Simple equity scaling (proportional to equity/start_equity)
   - ATR risk sizing (size = equity*risk% / (ATR * multiplier))
   - clamps min/max order size

Blocked BUY intents are recorded in the trader ledger with reason:
- DRAWDOWN_LIMIT
- CORRELATION_LIMIT
and shown under 'Order attempts blocked' per pair.
