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

Quickstart (eerste keer gebruiken)
1) Start de app

Start Streamlit zoals je gewend bent (lokaal of via Streamlit Cloud).

Controleer dat de app laadt en dat je Live/Simulation en de Trainer en Profile Manager pagina’s ziet.

2) Kies je markt/pairs en timeframe

Ga naar de Live/Simulation pagina.

Zet je gewenste pairs (bijv. BTC/EUR, ETH/EUR) en timeframe (bijv. 15m).

Laat de overige instellingen voorlopig op default.

3) Draai de Trainer om profielen te maken

Ga naar Trainer.

Kies:

Timeframe (bijv. 15m)

Lookback (genoeg dagen om folds te kunnen maken)

Folds/test window/step (defaults zijn prima om te starten)

Kies Per symbol (default) of zet Global best across symbols aan als je één profielset voor alle symbols wilt.

Klik Run Trainer.

Resultaat:

De Trainer maakt profielen en toont je resultaten.

Aan het einde wordt automatisch een bundle JSON opgeslagen in:

data/profiles/…json

Je krijgt ook een knop Download bundle JSON.

4) Valideer en test de bundle (Profile Manager)

Ga naar Profile Manager (Governance).

Selecteer je nieuwste bundle uit Stored bundles (of upload je JSON).

Check:

Validation = YES

Bekijk de Diff vs current session (wat verandert er).

Run de Sanity Backtest:

Doel: snelle smoke-test op recente data.

Dit geeft per symbool PASS/FAIL + trades, PnL, drawdown.

5) Promote naar ACTIVE (alleen na Sanity PASS)

Als Sanity PASS is: klik Promote this bundle to ACTIVE.

Dit zet de bundle als:

data/profiles/active.json

En bewaart de vorige active in:

data/profiles/active_history/…

6) Apply in je sessie (optioneel)

Wil je direct met die settings door in de huidige Streamlit sessie:

In Profile Manager: vink confirm aan → Apply bundle

Dit overschrijft alleen instellingen in session_state.pair_cfg (het plaatst geen echte orders).

7) Rollback als iets niet bevalt

In Profile Manager → Rollback ACTIVE:

Kies een bestand uit active_history

Klik Rollback ACTIVE

Alles wordt gelogd in Audit log (data/profiles/audit_log.jsonl).

Praktische tips

Krijg je “INSUFFICIENT_HISTORY”: verhoog Lookback of verlaag folds/windows.

Voor een snelle iteratie: gebruik minder restarts en lagere max evals/regime.

Als je meerdere pairs test: Global best is handig, maar per-symbol geeft vaak betere fit per markt.

Als je wilt, kan ik dit ook als docs/GETTING_STARTED.md toevoegen in de volgende zip, zodat het netjes in je repo staat.


