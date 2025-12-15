This build is based on the stable stop-loss simulation version and adds:
- 🛑 STOP & FLATTEN (panic): closes all positions and auto-pauses.
- Auto-pause when portfolio drawdown stop triggers.
- Resume requires confirmation (15s). Resume is blocked while portfolio_stop_active is ACTIVE.
