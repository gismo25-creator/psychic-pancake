This build adds per-pair PAUSE / RESUME buttons inside each pair tab.
When a pair is paused:
- No new BUY/SELL grid executions for that pair (eng.check_price is skipped)
- Charts and metrics continue updating
- Other pairs can keep trading (if global trading is RUNNING)

Use case:
- Trade only one pair while keeping others visible.
