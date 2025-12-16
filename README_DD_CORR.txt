This build adds portfolio-level controls:
1) Max concurrent assets-in-drawdown:
   - asset drawdown vs avg entry (%)
   - blocks BUYs when dd_assets_count >= max_assets_in_dd
2) Correlation filter (rolling):
   - blocks BUYs if corr(candidate, held_asset) >= threshold
Blocked BUY attempts are logged via trader.record_blocked() and visible in the 'Order attempts blocked' table.
