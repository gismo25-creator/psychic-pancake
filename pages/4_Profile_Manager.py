import json
from pathlib import Path

import streamlit as st
import pandas as pd
import copy

from core.profiles.registry import (
    list_bundles, load_bundle, validate_bundle, diff_profiles, save_bundle, ensure_store_dir,
    active_path, load_bundle as _lb, promote_to_active, list_active_history, rollback_active, append_audit
)
from core.backtest.replay import run_backtest
from core.backtest.metrics import summarize_run
from core.backtest.data_store import load_or_fetch

st.set_page_config(layout="wide")
st.title("Profile Manager (Governance)")

store_dir = ensure_store_dir()
st.sidebar.subheader("ACTIVE")
ap = active_path(store_dir)
if Path(ap).is_file():
    st.sidebar.caption(f"Active: {Path(ap).name}")
else:
    st.sidebar.caption("Active: (none)")


# Load ACTIVE bundle (if present) and apply to current session
ap_file = active_path(store_dir)
if Path(ap_file).is_file():
    if st.sidebar.button("Load ACTIVE into session", use_container_width=True):
        active_bundle = load_bundle(ap_file)
        ok_a, errs_a, warns_a = validate_bundle(active_bundle)
        if warns_a:
            st.sidebar.warning("\n".join(warns_a))
        if not ok_a:
            st.sidebar.error("ACTIVE bundle is invalid:\n" + "\n".join(errs_a))
        else:
            act_profiles = active_bundle.get("profiles", {})
            st.session_state.setdefault("pair_cfg", {})
            for sym, cfg in act_profiles.items():
                sym_u = str(sym).upper()
                st.session_state.pair_cfg.setdefault(sym_u, {}).update(cfg)
            append_audit("applied_active_session", {"source": "active.json", "symbols": sorted(list(act_profiles.keys()))}, store_dir=store_dir)
            st.sidebar.success("Loaded ACTIVE into session.")
else:
    st.sidebar.caption("No ACTIVE bundle yet.")


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

st.subheader("Compare Candidate vs ACTIVE")

ap = active_path(store_dir)
active_bundle = None
if Path(ap).is_file():
    try:
        active_bundle = load_bundle(ap)
    except Exception as e:
        st.error(f"Could not load ACTIVE bundle: {e}")

if active_bundle is None:
    st.info("No ACTIVE bundle found yet (promote one after a sanity PASS).")
else:
    # High-level meta compare
    cmeta = (bundle.get("meta", {}) or {})
    ameta = (active_bundle.get("meta", {}) or {})
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Candidate meta")
        st.json(cmeta)
    with c2:
        st.caption("ACTIVE meta")
        st.json(ameta)

    # Profile diff (candidate vs active)
    cand_prof = bundle.get("profiles", {}) or {}
    act_prof = active_bundle.get("profiles", {}) or {}
    # Reuse diff_profiles by treating act_prof as "current"
    diff_ca = diff_profiles(act_prof, cand_prof)
    with st.expander("Profile diff (Candidate vs ACTIVE)", expanded=False):
        st.json(diff_ca)

    st.caption("Run sanity on both to compare quickly before promotion.")
    min_trades_per_symbol = st.slider(
        "Min trades per symbol (sanity gate)",
        min_value=0, max_value=200, value=5, step=1,
        help="No-trade guard + minimum activity gate for sanity checks.",
    )
    max_dd_gate_pct = st.slider(
        "Max drawdown gate (%)",
        min_value=0.5, max_value=50.0, value=15.0, step=0.5,
        help="If a symbol exceeds this DD in the sanity run, it fails.",
    )
    require_all_symbols = st.checkbox(
        "Require sanity PASS for all symbols",
        value=True,
        help="If enabled, any FAIL makes overall FAIL.",
    )

    def _summarize_sanity(rows):
        # rows: list of dicts with keys: symbol, trades, max_dd_pct, status
        if not rows:
            return {"pass": False, "reason": "NO_RESULTS"}
        failed = [r for r in rows if r.get("status") != "PASS"]
        overall = (len(failed) == 0) if require_all_symbols else (len(failed) < len(rows))
        return {"pass": bool(overall), "n": len(rows), "fails": len(failed)}

    # Use the existing sanity runner on page if present; else do minimal evaluation based on last stored results.
    if "run_sanity_backtest" in globals():
        colA, colB = st.columns(2)
        with colA:
            if st.button("Run sanity: Candidate", use_container_width=True):
                cand_rows = run_sanity_backtest(bundle, min_trades_per_symbol=min_trades_per_symbol, max_dd_gate_pct=max_dd_gate_pct)
                st.session_state["sanity_rows_candidate"] = cand_rows
                summ = _summarize_sanity(cand_rows)
                st.session_state["sanity_pass_candidate"] = summ["pass"]
                st.session_state["sanity_summary_candidate"] = summ
        with colB:
            if st.button("Run sanity: ACTIVE", use_container_width=True):
                act_rows = run_sanity_backtest(active_bundle, min_trades_per_symbol=min_trades_per_symbol, max_dd_gate_pct=max_dd_gate_pct)
                st.session_state["sanity_rows_active"] = act_rows
                summ = _summarize_sanity(act_rows)
                st.session_state["sanity_pass_active"] = summ["pass"]
                st.session_state["sanity_summary_active"] = summ

        cand_rows = st.session_state.get("sanity_rows_candidate", [])
        act_rows = st.session_state.get("sanity_rows_active", [])
        if cand_rows or act_rows:
            st.markdown("**Sanity results (latest runs)**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption(f"Candidate: {'PASS' if st.session_state.get('sanity_pass_candidate') else 'FAIL'}")
                st.dataframe(cand_rows, use_container_width=True, height=240) if cand_rows else st.caption("Not run yet.")
            with c2:
                st.caption(f"ACTIVE: {'PASS' if st.session_state.get('sanity_pass_active') else 'FAIL'}")
                st.dataframe(act_rows, use_container_width=True, height=240) if act_rows else st.caption("Not run yet.")
    else:
        st.caption("Sanity runner function not found; use the Sanity Backtest section above.")


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

# Store canonical sanity artifacts for Promotion/Dry-run gates
if cache and isinstance(cache.get("results"), list):
    st.session_state["sanity_rows"] = cache.get("results", [])
else:
    st.session_state["sanity_rows"] = []
st.session_state["sanity_summary"] = {
    "bundle_id": bundle_id,
    "require_no_errors": bool(require_no_errors),
    "n_symbols": len(st.session_state["sanity_rows"]),
    "fails": sum(1 for r in st.session_state["sanity_rows"] if r.get("status") == "FAIL"),
    "errors": sum(1 for r in st.session_state["sanity_rows"] if r.get("status") == "ERROR"),
}
st.session_state["sanity_pass"] = bool(sanity_pass)


# --- Normalize sanity results (legacy keys -> candidate keys)
if "sanity_pass_candidate" not in st.session_state and "sanity_pass" in st.session_state:
    st.session_state["sanity_pass_candidate"] = bool(st.session_state.get("sanity_pass"))
if "sanity_rows_candidate" not in st.session_state and "sanity_rows" in st.session_state:
    st.session_state["sanity_rows_candidate"] = st.session_state.get("sanity_rows", [])
if "sanity_summary_candidate" not in st.session_state and "sanity_summary" in st.session_state:
    st.session_state["sanity_summary_candidate"] = st.session_state.get("sanity_summary", {})

st.subheader("Promotion")
st.caption("Workflow: Sanity PASS → Promote to ACTIVE → (optioneel) Rollback via history.")

# Expect sanity results in session_state if the sanity test was run on this page.
sanity_ok = bool(st.session_state.get("sanity_pass_for_bundle", st.session_state.get("sanity_pass_candidate", st.session_state.get("sanity_pass", False))))
sanity_summary = st.session_state.get("sanity_summary", st.session_state.get("sanity_summary_candidate", st.session_state.get("sanity_summary", {})))
if sanity_summary or sanity_ok:
    st.caption(f"Last sanity: {'PASS' if sanity_ok else 'FAIL'} | {sanity_summary}")
else:
    st.info("Run the Sanity Backtest above to enable promotion.")

note = st.text_input("Promotion note (optional)", value="")
cand_rows = st.session_state.get('sanity_rows', st.session_state.get('sanity_rows_candidate', []))
no_trade = any((r.get('trades', 0) == 0) for r in cand_rows) if cand_rows else False
if no_trade:
    st.warning('Sanity gate: at least one symbol had 0 trades (no-trade guard). Promotion disabled.')
if st.button("Promote this bundle to ACTIVE", disabled=(not sanity_ok or no_trade)):
        # Stamp meta so Live/Dry-run can enforce safety gates
    bundle.setdefault('meta', {})
    bundle['meta']['sanity_passed'] = True
    bundle['meta']['sanity_summary'] = st.session_state.get('sanity_summary_candidate', st.session_state.get('sanity_summary', {}))
    bundle['meta']['sanity_rows'] = st.session_state.get('sanity_rows_candidate', st.session_state.get('sanity_rows', []))
    apath, _ = promote_to_active(bundle, store_dir=store_dir, note=note)
    st.success(f"Promoted to {apath}")

st.subheader("Rollback ACTIVE")
hist = list_active_history(store_dir)
if not hist:
    st.caption("No active history available yet.")
else:
    pick = st.selectbox("Select history to roll back to", hist, format_func=lambda p: Path(p).name)
    if st.button("Rollback ACTIVE to selected"):
        apath = rollback_active(store_dir=store_dir, history_path=pick)
        st.success(f"Rolled back ACTIVE to {apath}")

st.subheader("Audit log")
audit_path = Path(store_dir) / "audit_log.jsonl"
if audit_path.is_file():
    tail_n = st.slider("Show last N events", 10, 200, 50, step=10)
    lines = audit_path.read_text(encoding="utf-8").splitlines()[-tail_n:]
    events = []
    for ln in lines:
        try:
            events.append(json.loads(ln))
        except Exception:
            continue
    st.json(events)
else:
    st.caption("No audit log yet.")


st.subheader("Apply to session")
confirm = st.checkbox("I understand this overwrites per-symbol settings in this session.")
require_sanity = st.checkbox("Require sanity PASS to apply", value=True)
sanity_ok = (st.session_state.get("sanity_pass_for_bundle", False) or (not require_sanity))
if st.button("Apply bundle", disabled=(not ok or not confirm or not sanity_ok)):
    st.session_state.setdefault("pair_cfg", {})
    for sym, cfg in profiles.items():
        sym_u = str(sym).upper()
        st.session_state.pair_cfg.setdefault(sym_u, {}).update(cfg)
    append_audit("applied_bundle_session", {"bundle": bundle_path, "symbols": sorted(list(profiles.keys()))}, store_dir=store_dir)
    st.success("Applied to session_state.pair_cfg.")

st.subheader("Save a copy to disk")
name = st.text_input("Save as", value=Path(bundle_path).name.replace(".json","") if bundle_path else "profile_copy")
if st.button("Save bundle"):
    p = save_bundle(bundle, store_dir=store_dir, name=name)
    append_audit("saved_bundle_copy", {"saved_path": p, "source": bundle_path}, store_dir=store_dir)
    st.success(f"Saved to {p}")
