# Getting Started

This project is a Streamlit grid-trading research + simulation environment with a governance workflow for regime profiles.

## Quickstart (first use)

### 1) Start the app
- Start Streamlit locally or on Streamlit Cloud.
- Confirm you can open:
  - Live/Simulation
  - Trainer
  - Profile Manager (Governance)

### 2) Choose market/pairs and timeframe
- Open **Live/Simulation**
- Set your **pairs** (e.g. `BTC/EUR, ETH/EUR`) and a **timeframe** (e.g. `15m`)
- Keep the rest on defaults for a first run.

### 3) Run the Trainer (create profiles)
- Open **Trainer**
- Configure:
  - **Timeframe** (e.g. `15m`)
  - **Lookback** (enough days to build folds)
  - **Folds / test window / step** (defaults are fine)
- Choose:
  - **Per symbol** (default), or
  - Enable **Global best across symbols** for one shared profile-set across all selected symbols.
- Click **Run Trainer**.

Result:
- A **bundle JSON** is created and saved automatically to:
  - `data/profiles/<bundle_...>.json`
- You can also download the bundle via **Download bundle JSON**.

### 4) Validate + test the bundle (Profile Manager)
- Open **Profile Manager (Governance)**
- Select the newest bundle under **Stored bundles** (or upload your JSON).
- Check:
  - **Validation = YES**
  - Review **Diff vs current session** (what will change)
- Run the **Sanity Backtest**:
  - A quick smoke test on recent data
  - Produces PASS/FAIL per symbol with trades, PnL, drawdown

### 5) Promote to ACTIVE (only after PASS)
- When sanity is **PASS**, click **Promote this bundle to ACTIVE**
- This writes:
  - `data/profiles/active.json`
- And stores previous actives automatically in:
  - `data/profiles/active_history/active_<timestamp>.json`

### 6) Apply to your current session (optional)
If you want to use the bundle immediately in the current Streamlit session:
- In **Profile Manager**, confirm the checkbox and click **Apply bundle**  
This overwrites only `st.session_state.pair_cfg` (no real orders are placed).

### 7) Rollback when needed
- In **Profile Manager → Rollback ACTIVE**
  - Select a file from `active_history`
  - Click **Rollback ACTIVE**

### 8) Audit trail
All key events are logged to:
- `data/profiles/audit_log.jsonl`

## Practical tips
- If you get `INSUFFICIENT_HISTORY`: increase **Lookback**, or reduce folds/windows.
- For faster iteration: reduce **Restarts** and **Max evals/regime**.
- Global-best is convenient for multi-pair, but per-symbol often fits each market better.
