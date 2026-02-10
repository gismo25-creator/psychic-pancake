# Fix: Live gate blocks the page (non-blocking)

If selecting **Live** shows only the gate error and the rest of the page seems to disappear,
your code is likely calling `st.stop()` when no ACTIVE bundle is present.

This patch removes that hard stop so you can still access the Live page UI.

## Apply
Unzip into your project root and run:

    python tools/apply_fix_live_gate_nonblocking.py

Then restart Streamlit:

    streamlit run streamlit_app.py