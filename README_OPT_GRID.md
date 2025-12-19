# Added: Optimal grid range/levels + order size scaling

## Optimal grid range / levels (heuristic)
Per pair, enable **Auto-optimize grid range/levels**.
- Computes suggested range (±%) and (Linear) levels to target a hit-rate.
- Uses ATR to avoid overly tight/wide spacing.
- Suggestions are applied only when you click **Apply suggested params**.

## Order size scaling (dynamic multiplier)
Per pair, enable **Dynamic order-size multiplier**.
- Adjusts order size based on effective regime, volatility clustering and hit-rate.
- Clamped between min/max multipliers.
- Applied on top of existing equity scaling.
