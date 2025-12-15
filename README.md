# Grid Trading Bot – Bitvavo (Simulation)

## Overzicht
Dit project is een **multi-pair grid trading simulator** voor **Bitvavo spot trading (EUR-paren)**, gebouwd met **Python + Streamlit**. Het systeem is ontworpen om grid-strategieën **realistisch te simuleren**, inclusief **exacte PnL-berekening**, **fees**, **slippage**, **regime-detectie** en **geavanceerde risk-managementlogica**.

De codebase is expliciet opgezet als **voortraject richting live trading**, waarbij strategie, risk controls en parameters uitgebreid getest kunnen worden vóórdat echte orders worden geplaatst.

---

## Kernfunctionaliteit

### Exchange & Marktdata
- Exchange: **Bitvavo** (via CCXT)
- Spot trading (EUR quote currency)
- Multi-pair ondersteuning (bijv. `BTC/EUR, ETH/EUR`)
- OHLCV-data per pair en timeframe

---

## Grid Trading

### Grid types (per pair)
- **Linear grid**
- **Fibonacci grid**

### Grid parameters (per pair instelbaar)
- Grid range ± (%)
- Aantal levels (linear)
- Order size (base asset)
- Regime-based dynamic spacing
  - Range multiplier
  - Levels reduction

### Grid engine
- BUY bij neerwaartse grid-crossing
- SELL bij eerstvolgende grid omhoog
- Exacte grid-cycles (BUY → SELL)
- Open cycles worden gereset bij geforceerde exits (stop-loss)

---

## Market Regime & Volatility (ML – Stap 1)

### Volatility metrics
- **ATR (14)**
- **Realized volatility**
- **Bollinger Bandwidth**
- **ADX (trend strength)**

### Regime classificatie
- `WARMUP`
- `RANGE`
- `TREND`
- `CHAOS`

### Regime stabilisatie
- Confirmations (N opeenvolgende signals)
- Cooldown (minimale tijd tussen regime-wissels)
- Per pair eigen regime-state

### Regime → Grid behavior
- Dynamische aanpassing van grid range en levels
- ATR-floor voorkomt te smalle grids bij hoge volatiliteit

---

## Portfolio Simulator (Exact Accounting)

### Cash & posities
- Eén **EUR cash ledger**
- Posities per base asset (BTC, ETH, …)
- Average cost basis per asset

### Fees & slippage
- Bitvavo **fee tiers (Category A)**
- Maker / Taker keuze
- Custom fee override
- Slippage (% van prijs)

### Exact realized PnL
- BUY: exacte cash-out (incl. fees)
- SELL: exacte cash-in (na fees)
- **Realized PnL = cash_in − cash_out**
- Equity-verandering matcht 1-op-1 met trade-ledger

---

## Risk Management

### Max open exposure per asset
- Cap per base asset in **EUR**
- Blokkeert BUY-orders die exposure overschrijden
- SELL-orders altijd toegestaan

### Portfolio drawdown stop (simulatie)
- Houdt **peak equity** bij
- Trigger bij maximale drawdown (%)
- Acties:
  - Optioneel: flatten alle posities
  - Globale BUY-stop (latched tot reset)

### Per-asset stop-loss (simulatie)
- Trigger bij:
  - Prijs X% onder **average entry**
  - Optioneel: **ATR-multiple stop**
- Acties:
  - Flatten volledige asset-positie
  - Asset halt: geen nieuwe BUY’s voor die asset
  - Grid open cycles reset

---

## Multi-Pair Trading

- Meerdere paren tegelijk
- Eén portfolio / cash ledger
- Eén grid engine per pair
- Per pair:
  - Eigen grid parameters
  - Eigen regime state
  - Eigen trade history
- UI met tabs per pair

---

## Visualisatie & Dashboard

### Charts
- Candlestick chart per pair
- Grid levels (horizontale lijnen)
- Trade markers:
  - BUY (groen ▲)
  - SELL (rood ▼)
- Markers op **echte trade-timestamps**

### Tabellen
Per pair:
- Trades:
  - Time
  - Side
  - Price
  - Amount
  - Fee (%)
  - Cash Δ (EUR)
  - Exact realized PnL
  - Cumulatieve PnL
  - Reason (GRID / STOPLOSS / RISK_LIMIT)

Extra:
- Geblokkeerde orders (risk-limit / insufficient cash)

### Portfolio overzicht
- Cash (EUR)
- Equity (EUR)
- Peak equity
- Drawdown (%)
- Portfolio stop status
- Posities + average entry

---

## Simulator Controls

- Session reset
- Refresh interval
- Timeframe selectie
- Pair selectie (comma-separated)
- Stop-loss toggles (simulatie)

---

## Wat zit er bewust nog NIET in

- Live trading (Bitvavo REST orders)
- Stop-loss orders op exchange-niveau
- Partial fills / orderbook simulation
- Leverage / funding (spot only)
- ML-parameter optimalisatie

---

## Aanbevolen vervolgstappen

1. **Cool-off timers** voor asset halts
2. **Soft stops** (gedeeltelijke de-risking)
3. **ML stap 2**: grid-parameter learning per regime
4. **Paper → live switch** met identieke strategy engine

---

## Starten

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

**Doel van dit project:**
Een robuuste, transparante en uitbreidbare grid trading basis die veilig getest kan worden vóór live deployment.
