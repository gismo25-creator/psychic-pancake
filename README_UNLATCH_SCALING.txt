This build extends the stable stop-loss + panic version with:

1) 🔓 UNLATCH STOP (Portfolio stop latch clear)
- Clears portfolio_stop_active without resetting the whole session
- Trading remains paused; resume manually via START + CONFIRM
- Peak equity is reset to current equity (avoids instant retrigger)

2) Equity-based position scaling (simulation)
- Toggle in sidebar
- Modes:
  - Simple equity scaling: order_size * (equity / start_equity)
  - ATR risk sizing: size = (equity * risk%) / (ATR * multiplier)
- Min/Max clamps
- Baseline reset button
