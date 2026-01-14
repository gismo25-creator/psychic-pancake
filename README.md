# Gridbot

## Live Bitvavo execution (experimental)

This project defaults to **Simulation** and **Dry-run Live** (paper).  
To enable **Live (Bitvavo, real orders)** you must:

1. Add credentials to Streamlit secrets:
   - `.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets
   - Keys: `BITVAVO_API_KEY`, `BITVAVO_API_SECRET`
2. In the Live page sidebar:
   - Select **Mode: Live (Bitvavo, real orders)**
   - Check the risk acknowledgment checkbox
   - Toggle **Enable Live executor (REAL orders)**

Safety notes:
- Live executor v1 uses **MARKET** orders only (immediate fills).
- Keep refresh interval >= 15s to reduce rate-limit pressure.


### Extra Live safeties
- **Arm STOP & FLATTEN**: required to allow market sells from the panic button in Live mode.
- **Live max order notional (EUR)**: hard cap per order (price*amount) for Live orders.
