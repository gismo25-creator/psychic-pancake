import json
from pathlib import Path

import streamlit as st
import pandas as pd
import copy

from core.profiles.registry import list_bundles, load_bundle, validate_bundle, diff_profiles, save_bundle, ensure_store_dir
from core.backtest.replay import run_backtest
from core.backtest.metrics import summarize_run
from core.backtest.data_store import load_or_fetch

st.set_page_config(layout="wide")
st.title("Profile Manager (Governance)")

store_dir = ensure_store_dir()

st.sidebar.subheader("Stored bundles")
files = list_bundles(store_dir)
selected = None
if files:
    selected = st.sidebar.selectbox("Select bundle", files, format_func=lambda p: Path(p).name)

st.sidebar.subheader("Upload bundle")
uploaded = st.sidebar.file_uploader("Upload bundle (.json)", type=["json"])

bundle = None
bundle_path = None

if uploaded is not None:
    try:
        bundle = json.loads(uploaded.getvalue().decode("utf-8"))
        bundle_path = f"UPLOAD::{uploaded.name}"
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
elif selected:
    bundle = load_bundle(selected)
    bundle_path = selected

if bundle is None:
    st.info("Run Trainer to create bundles, or upload a bundle.")
    st.stop()

ok, errors, warnings = validate_bundle(bundle)

c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    st.subheader("Validation")
    st.metric("Valid", "YES" if ok else "NO")
with c2:
    st.subheader("Bundle")
    st.caption(bundle_path)
    st.caption(f"schema_version: {bundle.get('schema_version')}")
    st.caption(f"created_at: {bundle.get('created_at')}")
with c3:
    st.subheader("Metadata")
    st.json(bundle.get("meta", {}))

if warnings:
    st.warning("\n".join(warnings))
if errors:
    st.error("\n".join(errors))

profiles = bundle.get("profiles", {})
st.subheader("Symbols in bundle")
st.write(sorted(list(profiles.keys())))

st.subheader("Diff vs current session")
cur_cfg = st.session_state.get("pair_cfg", {}) or {}
st.json(diff_profiles(cur_cfg, profiles))

st.subheader("Sanity Backtest (smoke test before apply)")

# Keep results stable across dropdown changes
bundle_id = str(bundle.get("created_at", "")) + "::" + str(bundle_path)
if "sanity_cache" not in st.session_state:
    st.session_state.sanity_cache = {}

# Build a candidate cfg by overlaying bundle profiles on current session cfg
cur_cfg = st.session_state.get("pair_cfg", {}) or {}
profiles = bundle.get("profiles", {}) or {}
cand_cfg = copy.deepcopy(cur_cfg)
for sym, cfg in profiles.items():
    sym_u = str(sym).upper()
    cand_cfg.setdefault(sym_u, {}).update(cfg)

# If a symbol has no base cfg yet, provide a safe default so the smoke test can run.
def _default_base_cfg() -> dict:
    return {
        "grid_type": "Linear",
        "base_range_pct": 1.0,
        "dynamic_spacing": True,
        "k_range": 1.5,
        "k_levels": 0.7,
        "base_levels": 10,
        "order_size": 0.001,
        "use_regime_profiles": False,
        "regime_profile_rebuild": False,
    }

for sym in profiles.keys():
    sym_u = str(sym).upper()
    cand_cfg.setdefault(sym_u, _default_base_cfg())

meta = bundle.get("meta", {}) or {}

colS1, colS2, colS3 = st.columns([2, 2, 2])
with colS1:
    sanity_days = st.number_input("Recent history (days)", min_value=3, max_value=365, value=30, step=1)
with colS2:
    sanity_start_cash = st.number_input("Start cash (EUR)", min_value=50.0, value=float(meta.get("start_cash", 1000.0) or 1000.0), step=50.0)
with colS3:
    sanity_force_refresh = st.checkbox("Force refresh market data", value=False)

fees_meta = (meta.get("fees") or {}) if isinstance(meta.get("fees"), dict) else {}
cF1, cF2, cF3, cF4 = st.columns(4)
with cF1:
    maker_fee = st.number_input("Maker fee", min_value=0.0, max_value=0.02, value=float(fees_meta.get("maker", 0.0015)), step=0.0001, format="%.5f")
with cF2:
    taker_fee = st.number_input("Taker fee", min_value=0.0, max_value=0.02, value=float(fees_meta.get("taker", 0.0025)), step=0.0001, format="%.5f")
with cF3:
    slippage = st.number_input("Slippage", min_value=0.0, max_value=0.02, value=float(fees_meta.get("slippage", 0.0005)), step=0.0001, format="%.5f")
with cF4:
    fee_mode = st.selectbox("Fee mode", ["taker", "maker"], index=0 if str(fees_meta.get("mode", "taker")) == "taker" else 1)

cP1, cP2, cP3 = st.columns(3)
with cP1:
    min_trades = st.number_input("Min trades (pass)", min_value=0, max_value=500, value=5, step=1)
with cP2:
    max_dd_pct = st.number_input("Max drawdown % (pass)", min_value=0.1, max_value=80.0, value=20.0, step=0.5)
with cP3:
    require_no_errors = st.checkbox("Require no errors", value=True)

run_sanity = st.button("Run sanity backtest", use_container_width=True)

if run_sanity:
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=float(sanity_days))
    results = []
    errors = []
    prog = st.progress(0.0, text="Running sanity backtest...")

    syms = sorted([str(s).upper() for s in profiles.keys()])
    for i, sym in enumerate(syms):
        prog.progress((i) / max(1, len(syms)), text=f"Fetching data for {sym}...")
        try:
            df = load_or_fetch(sym, timeframe=str(meta.get("timeframe", "15m")), since=since, until=None, force_refresh=bool(sanity_force_refresh))
            if df is None or df.empty or len(df) < 50:
                results.append({"symbol": sym, "status": "FAIL", "reason": "INSUFFICIENT_DATA", "trades": 0, "total_pnl": 0.0, "max_dd_pct": 0.0})
                continue

            dfs = {sym: df}
            pcfg = {sym: cand_cfg.get(sym, _default_base_cfg())}

            trades_df, equity_curve, decision_log, trader = run_backtest(
                dfs=dfs,
                pair_cfg=pcfg,
                timeframe=str(meta.get("timeframe", "15m")),
                start_cash=float(sanity_start_cash),
                maker_fee=float(maker_fee),
                taker_fee=float(taker_fee),
                slippage=float(slippage),
                fee_mode=str(fee_mode),
                quote_ccy="EUR",
                max_exposure_quote={},
                regime_profiles=pcfg[sym].get("regime_profiles"),
                enable_regime_profiles=bool(pcfg[sym].get("use_regime_profiles", False)),
                confirm_n=3,
                cooldown_candles=0,
                rebuild_on_regime_change=False,
            )
            summ = summarize_run(equity_curve, trades_df)
            ntr = int(summ.get("num_trades", 0) or 0)
            pnl = float(summ.get("total_pnl", 0.0) or 0.0)
            mdd = float(summ.get("max_drawdown", 0.0) or 0.0) * 100.0

            pass_trades = ntr >= int(min_trades)
            pass_dd = mdd <= float(max_dd_pct)

            status = "PASS" if (pass_trades and pass_dd) else "FAIL"
            results.append({"symbol": sym, "status": status, "trades": ntr, "total_pnl": pnl, "max_dd_pct": mdd})

        except Exception as e:
            err = f"{sym}: {type(e).__name__}: {e}"
            errors.append(err)
            results.append({"symbol": sym, "status": "ERROR", "trades": 0, "total_pnl": 0.0, "max_dd_pct": 0.0, "reason": str(e)})

    prog.progress(1.0, text="Done.")
    df_res = pd.DataFrame(results).sort_values(["status", "symbol"])
    st.session_state.sanity_cache[bundle_id] = {"results": df_res.to_dict(orient="records"), "errors": errors}

# Show cached results if present
cache = st.session_state.sanity_cache.get(bundle_id)
sanity_pass = False
if cache:
    df_res = pd.DataFrame(cache.get("results", []))
    if not df_res.empty:
        st.dataframe(df_res, use_container_width=True, height=220)
        # Determine overall pass
        has_error = any(r.get("status") == "ERROR" for r in cache.get("results", []))
        any_fail = any(r.get("status") == "FAIL" for r in cache.get("results", []))
        sanity_pass = (not any_fail) and (not (has_error and require_no_errors))
        st.caption(f"Sanity result: {'PASS' if sanity_pass else 'FAIL'}")
    if cache.get("errors"):
        st.error("\n".join(cache["errors"]))

# Gate apply: require sanity PASS by default
st.session_state["sanity_pass_for_bundle"] = sanity_pass

st.subheader("Apply to session")
confirm = st.checkbox("I understand this overwrites per-symbol settings in this session.")
require_sanity = st.checkbox("Require sanity PASS to apply", value=True)
sanity_ok = (st.session_state.get("sanity_pass_for_bundle", False) or (not require_sanity))
if st.button("Apply bundle", disabled=(not ok or not confirm or not sanity_ok)):
    st.session_state.setdefault("pair_cfg", {})
    for sym, cfg in profiles.items():
        sym_u = str(sym).upper()
        st.session_state.pair_cfg.setdefault(sym_u, {}).update(cfg)
    st.success("Applied to session_state.pair_cfg.")

st.subheader("Save a copy to disk")
name = st.text_input("Save as", value=Path(bundle_path).name.replace(".json","") if bundle_path else "profile_copy")
if st.button("Save bundle"):
    p = save_bundle(bundle, store_dir=store_dir, name=name)
    st.success(f"Saved to {p}")
