from pathlib import Path
import math
import time
import json
import hashlib
import re
import io
from collections import deque
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

# ----------------------------
# Health / diagnostics (persisted in session_state)
# ----------------------------
HEALTH_KEY = "health"

def _health() -> dict:
    if HEALTH_KEY not in st.session_state or not isinstance(st.session_state.get(HEALTH_KEY), dict):
        st.session_state[HEALTH_KEY] = {}
    return st.session_state[HEALTH_KEY]

def health_set(k: str, v) -> None:
    h = _health()
    h[k] = v
    st.session_state[HEALTH_KEY] = h

def health_note(event: str, detail: str = "") -> None:
    now = time.time()
    health_set("last_event", event)
    health_set("last_event_ts", now)
    if detail:
        health_set("last_event_detail", str(detail)[:800])

PAGE_NS_LIVE = "live"

def k_live(s: str) -> str:
    return f"{PAGE_NS_LIVE}:{s}"

# --- Scanner export fallback (pairs) ---
try:
    _qp_pairs = None
    try:
        _qp_pairs = st.query_params.get("pairs")
    except Exception:
        _qp = st.experimental_get_query_params()
        _qp_pairs = (_qp.get("pairs", [None])[0] if isinstance(_qp.get("pairs"), list) else _qp.get("pairs"))
    if _qp_pairs:
        st.session_state[k_live("manual_symbols_input")] = str(_qp_pairs)
except Exception:
    pass

if "scanner:export_pairs_fallback" in st.session_state:
    st.session_state[k_live("manual_symbols_input")] = st.session_state.get("scanner:export_pairs_fallback", st.session_state.get(k_live("manual_symbols_input"), ""))

STATE_NS_LIVE = "live"

def s_live(key: str) -> str:
    return f"{STATE_NS_LIVE}:{key}"
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from core.market_data import (fetch_ohlcv_bitvavo_cached, fetch_ticker_bitvavo_cached, BitvavoRateLimitBan)

# ----------------------------
# Bitvavo rate-limit ban guard
# ----------------------------
def _bitvavo_is_banned() -> bool:
    until_ms = int(st.session_state.get('bitvavo_banned_until_ms', 0) or 0)
    now_ms = int(time.time() * 1000)
    if until_ms > now_ms:
        # show once per rerun
        remaining_s = max(0, (until_ms - now_ms) // 1000)
        st.sidebar.error(f"Bitvavo rate-limit ban active (~{remaining_s}s remaining). Data fetches are paused.")
        return True
    return False

from core.grid.linear import generate_linear_grid
from core.grid.fibonacci import generate_fibonacci_grid
from core.grid.engine import GridEngine
from core.exchange.simulator import PortfolioSimulatorTrader
from core.bots.store import load_bots
from core.exchange.bitvavo_live import BitvavoLiveTrader, BitvavoRateLimitBanned

from core.ml.volatility import atr, realized_vol, bollinger_bandwidth, adx, vol_cluster_acf1
from core.ml.regime import classify_regime

from core.profiles.registry import active_path, load_bundle, validate_bundle
from core.profiles.library import list_entries as list_library_entries, load_profile as load_library_profile

# ----------------------------
# Run control state (global)
# ----------------------------
if "trading_enabled" not in st.session_state:
    st.session_state.trading_enabled = False

if "start_pending" not in st.session_state:
    st.session_state.start_pending = False

if "start_pending_ts" not in st.session_state:
    st.session_state.start_pending_ts = 0.0

if "panic_flatten" not in st.session_state:
    st.session_state.panic_flatten = False

if "start_equity" not in st.session_state:
    st.session_state.start_equity = None

FEE_TIERS_CAT_A = [
    ("€0+",        0.0015, 0.0025),
    ("€100k+",     0.0010, 0.0020),
    ("€250k+",     0.0008, 0.0016),
    ("€500k+",     0.0006, 0.0012),
    ("€1M+",       0.0005, 0.0010),
    ("€2.5M+",     0.0004, 0.0008),
    ("€5M+",       0.0004, 0.0006),
    ("€10M+",      0.0000, 0.0005),
    ("€25M+",      0.0000, 0.0002),
    ("€100M+",     0.0000, 0.0001),
    ("€500M+",     0.0000, 0.0001),
]

# --- Execution mode defaults (defined early so top buttons can reference them) ---
exec_mode = st.session_state.get('exec_mode', 'Simulation (paper, candle close)')
dryrun_allowed = bool(st.session_state.get('dryrun_allowed', True))
active_bundle = st.session_state.get('active_bundle', None)

st.set_page_config(layout="wide")

# Update per-rerun heartbeat (used by Health panel)
health_set("last_rerun_ts", time.time())

# --- Stable Live page keys (avoid resets when switching pages)
def lk(name: str) -> str:
    return "live:" + name

# --- Persist Live UI prefs to disk (survive app reload / session reset)
import json as _json
from pathlib import Path as _Path

def _ui_prefs_path() -> _Path:
    return _Path(__file__).parent / 'data' / 'ui_prefs.json'

def _load_ui_prefs() -> dict:
    p = _ui_prefs_path()
    try:
        if p.exists():
            return _json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}

def _save_ui_prefs(prefs: dict) -> None:
    p = _ui_prefs_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(prefs, indent=2, sort_keys=True), encoding='utf-8')
    except Exception:
        # keep UI running even if disk is read-only
        pass

def _sync_prefs_from_state() -> None:
    prefs = _load_ui_prefs()
    prefs.update({
        'use_bot_manager': bool(st.session_state.get(lk('use_bot_manager'), False)),
        'manual_symbols_input': str(st.session_state.get(lk('manual_symbols_input'), 'BTC/EUR, ETH/EUR')),
        'timeframe': str(st.session_state.get(lk('timeframe'), '5m')),
        'refresh_sec': int(st.session_state.get(lk('refresh_sec'), 15)),
        'exec_mode': str(st.session_state.get(lk('exec_mode'), 'Simulation (paper)')),
    })
    _save_ui_prefs(prefs)

# one-time load of prefs into session_state (if missing)
_prefs = _load_ui_prefs()
if lk('use_bot_manager') not in st.session_state and 'use_bot_manager' in _prefs:
    st.session_state[lk('use_bot_manager')] = bool(_prefs['use_bot_manager'])
if lk('manual_symbols_input') not in st.session_state and 'manual_symbols_input' in _prefs:
    st.session_state[lk('manual_symbols_input')] = str(_prefs['manual_symbols_input'])
if lk('timeframe') not in st.session_state and 'timeframe' in _prefs:
    st.session_state[lk('timeframe')] = str(_prefs['timeframe'])
if lk('refresh_sec') not in st.session_state and 'refresh_sec' in _prefs:
    st.session_state[lk('refresh_sec')] = int(_prefs['refresh_sec'])
if lk('exec_mode') not in st.session_state and 'exec_mode' in _prefs:
    st.session_state[lk('exec_mode')] = str(_prefs['exec_mode'])

st.title("Grid Trading Bot – Bitvavo (Simulation + Panic Button + Auto-Pause)")

# --- BB state (for mean-reversion buy-filter) ---
if "bb_state" not in st.session_state:
    st.session_state["bb_state"] = {}

# --- Top controls: Start / Stop / Stop & Flatten / Reset ---
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if st.button("▶ START", width="stretch", disabled=((exec_mode.startswith("Dry-run") and (not dryrun_allowed)) or (exec_mode.startswith("Live") and (not st.session_state.get("live_enabled", False))))):
        # Require confirmation to resume
        st.session_state.start_pending = True
        st.session_state.start_pending_ts = time.time()

with c2:
    if st.button("⏸ STOP", width="stretch"):
        st.session_state.trading_enabled = False
        st.session_state.start_pending = False

with c3:
    if st.button(
        "🛑 STOP & FLATTEN",
        width="stretch",
        help="Panic button: closes all positions at market (simulation) and pauses trading."
    ):
        # Defer execution until we have latest prices.
        # In LIVE mode, require arming before submitting market sells.
        if exec_mode.startswith("Live") and st.session_state.get("live_enabled", False) and (not st.session_state.get("live_arm_panic", False)):
            st.session_state.panic_flatten = False
            st.warning("STOP & FLATTEN is not armed for LIVE. Trading paused only. Arm it in the Live sidebar to allow market sells.")
        else:
            st.session_state.panic_flatten = True
        st.session_state.trading_enabled = False
        st.session_state.start_pending = False
        # Latch portfolio stop so no new buys happen until reset.
        st.session_state[s_live('portfolio_stop_active')] = True

with c4:
    if st.button("🔓 UNLATCH STOP", width="stretch", help="Clear portfolio stop latch (allow new buys again). Trading stays paused; resume manually."):
        st.session_state[s_live('portfolio_stop_active')] = False
        st.session_state.trading_enabled = False
        st.session_state.start_pending = False
        # Reset peak to current equity to avoid immediate retrigger.
        # (Peak is set later after equity is computed.)
        st.session_state[s_live('portfolio_peak_eq')] = None

with c5:
    if st.button("⟲ RESET SESSION", width="stretch"):
        st.session_state.clear()
        st.rerun()

# --- Resume confirmation (anti-misclick) ---
# Window: 15 seconds to confirm, otherwise pending state expires.
if st.session_state.start_pending:
    if (time.time() - st.session_state.start_pending_ts) > 15:
        st.session_state.start_pending = False
    else:
        warn_col, btn_col = st.columns([3, 1])
        with warn_col:
            st.warning("Bevestig START (binnen 15s) om trading te hervatten.")
        with btn_col:
            if st.button("✅ CONFIRM RESUME", width="stretch", disabled=((exec_mode.startswith("Dry-run") and (not dryrun_allowed)) or (exec_mode.startswith("Live") and (not st.session_state.get("live_enabled", False))))):
                # Only allow resume if portfolio stop not active.
                if st.session_state.get("portfolio_stop_active", False):
                    st.error("Portfolio stop is ACTIVE. Reset session om opnieuw te starten.")
                else:
                    st.session_state.trading_enabled = True
                st.session_state.start_pending = False

if exec_mode.startswith("Live") and st.session_state.get("live_enabled", False):
    st.warning("LIVE EXECUTOR ENABLED: this session can place REAL orders on Bitvavo. Use small sizing and confirm your settings.")

st.caption(f"Trading status: {'RUNNING' if st.session_state.trading_enabled else 'STOPPED'} | Mode: {exec_mode}")

# ----------------------------

# Safety gates for LIVE (real orders)
live_allowed = True
live_enabled = False
live_missing = []
live_ack = False

# Developer/escape hatch: allow Live without an ACTIVE bundle (NOT recommended).
# This used to exist in earlier builds but was missing in this file, causing a NameError.
allow_dryrun_without_active = False

if exec_mode.startswith("Live"):
    # Optional override for advanced testing only. Keep OFF for normal operation.
    allow_dryrun_without_active = st.sidebar.checkbox(
        "Allow Live without ACTIVE bundle (dangerous)",
        value=False,
    )

    # Also allow LIVE if you have PASS Library profiles applied (simple workflow).
    allow_live_with_library = st.sidebar.checkbox(
        "Allow Live with PASS Library profiles (no ACTIVE bundle)",
        value=False,
        help="If ON, Live can run when you have applied a PASS profile from Profiles (simple) for each selected symbol/timeframe.",
    )

    # Ensure `symbols` exists before library gating (pairs input may be parsed later in the file)
    def _parse_symbols_csv(pairs_csv: str) -> list[str]:
        if not pairs_csv:
            return []
        out = []
        for _s in str(pairs_csv).split(','):
            _s = _s.strip().upper()
            if _s:
                out.append(_s)
        # unique, keep order
        seen, uniq = set(), []
        for _s in out:
            if _s not in seen:
                seen.add(_s)
                uniq.append(_s)
        return uniq

    _ui_prefs = st.session_state.get('ui_prefs', {}) or {}
    pairs_csv_for_symbols = (
        st.session_state.get('pairs_csv')
        or st.session_state.get('manual_pairs_csv')
        or st.session_state.get('manual_pairs')
        or _ui_prefs.get('manual_pairs_csv')
        or _ui_prefs.get('manual_pairs')
        or ''
    )
    symbols = _parse_symbols_csv(pairs_csv_for_symbols)

    def _library_live_ready(_symbols, _tf):
        src = st.session_state.get("live_profile_source", {}) or {}
        miss = []
        for s in _symbols:
            meta = src.get(s) or {}
            if (not meta.get("id")) or (meta.get("timeframe") != _tf) or (meta.get("gates_passed") is not True):
                miss.append(s)
        return (len(miss) == 0), miss

    if symbols:
        library_ready, library_missing = _library_live_ready(symbols, timeframe)
    else:
        library_ready, library_missing = (False, [])

    if active_bundle is None and (not allow_dryrun_without_active):
        if allow_live_with_library and library_ready:
            st.sidebar.warning("Live gate bypass: using PASS Library profiles (no ACTIVE bundle). Test carefully.")
        else:
            live_allowed = False
            msg = "Live gate: no ACTIVE bundle found. Promote a bundle to ACTIVE (or enable PASS Library bypass)."
            if allow_live_with_library and (not library_ready):
                msg += "\nLibrary gate failed for: " + ", ".join(library_missing) + ". Apply latest PASS profile in Profiles (simple)."
            st.sidebar.error(msg)
    else:
        try:
            if active_bundle is not None:
                ok, errs, warns = validate_bundle(active_bundle)
                if warns:
                    st.sidebar.warning("\n".join(warns))
                if (not ok) and (not allow_dryrun_without_active):
                    live_allowed = False
                else:
                    meta = active_bundle.get("meta", {}) or {}
                    if (not bool(meta.get("sanity_passed", False))) and (not allow_dryrun_without_active):
                        live_allowed = False
                        st.sidebar.error("Live gate: ACTIVE bundle is not marked sanity_passed. Re-promote after a PASS sanity test.")
        except Exception as e:
            if not allow_dryrun_without_active:
                live_allowed = False
                st.sidebar.error(f"Live gate: could not load/validate ACTIVE bundle: {e}")

    st.sidebar.subheader("Live executor (Bitvavo)")
    st.sidebar.caption("Live mode places REAL orders only if you explicitly enable it and acknowledge risk. Default is OFF.")

    api_key = st.secrets.get("BITVAVO_API_KEY", "")
    api_secret = st.secrets.get("BITVAVO_API_SECRET", "")

    if not api_key:
        live_missing.append("BITVAVO_API_KEY")
    if not api_secret:
        live_missing.append("BITVAVO_API_SECRET")

    if live_missing:
        live_allowed = False
        st.sidebar.error("Missing Streamlit secrets: " + ", ".join(live_missing))
        st.sidebar.code('BITVAVO_API_KEY = "..."\\nBITVAVO_API_SECRET = "..."', language="toml")

    live_ack = st.sidebar.checkbox(
        "I understand Live mode can place REAL orders on Bitvavo",
        value=bool(st.session_state.get("live_ack", False)),
        key="live_ack",
    )

    # Streamlit compatibility: older versions may not have _sidebar_toggle()
    def _sidebar_toggle(label: str, value: bool, key: str, disabled: bool = False, help: str | None = None) -> bool:
        try:
            return _sidebar_toggle(label, value=value, key=key, disabled=disabled, help=help)
        except Exception:
            return st.sidebar.checkbox(label, value=value, key=key, disabled=disabled, help=help)

    live_enabled = _sidebar_toggle(
        "Enable Live executor (REAL orders)",
        value=False,
        disabled=(not live_allowed) or (not live_ack),
        help="When enabled, the bot will send market orders via authenticated Bitvavo API.",
        key="live_enabled_toggle",
    )

    

    # Extra safety: panic / STOP & FLATTEN must be explicitly armed in LIVE
    live_arm_panic = _sidebar_toggle(
        "Arm STOP & FLATTEN (market sells)",
        value=bool(st.session_state.get("live_arm_panic", False)),
        disabled=(not live_enabled),
        help="When armed, STOP & FLATTEN will submit market SELL orders for all held assets. When NOT armed, the button only pauses trading.",
        key="live_arm_panic",
    )

    # Extra safety: hard cap per order in quote currency
    live_max_order_quote = st.sidebar.number_input(
        "Live: max order notional (EUR) per order",
        min_value=0.0,
        max_value=50000.0,
        value=float(st.session_state.get("live_max_order_quote", 50.0)),
        step=10.0,
        help="Hard cap on order notional (price*amount) for LIVE orders. Set 0 to disable (not recommended).",
        key="live_max_order_quote",
    )
    refresh_sec_live = int(st.session_state.get(lk('refresh_sec'), 15))
    if refresh_sec_live < 15:
        st.sidebar.warning("For Live mode, keep refresh >= 15s to reduce rate-limit pressure.")

    st.sidebar.info("Live executor v1 uses MARKET orders (immediate fills). Limit-order lifecycle management is not enabled yet.")

    st.session_state["live_allowed"] = bool(live_allowed)
    st.session_state["live_enabled"] = bool(live_enabled)
else:
    st.session_state["live_allowed"] = False
    st.session_state["live_enabled"] = False

# Market selection
# ----------------------------
st.sidebar.subheader("Market")
# Apply imported pairs before widget instantiation (prevents Streamlit state error)
PENDING_PAIRS_KEY = k_live("symbols_pending")
if PENDING_PAIRS_KEY in st.session_state and st.session_state[PENDING_PAIRS_KEY]:
    st.session_state[k_live("manual_symbols_input")] = st.session_state[PENDING_PAIRS_KEY]
    st.session_state[PENDING_PAIRS_KEY] = ""
# Import pairs from file (optional) — sets a pending value and triggers rerun BEFORE the Pairs widget is created
st.sidebar.caption("Importeer pairs via bestand (CSV of TXT). Handig na scanner export.")
pairs_file = st.sidebar.file_uploader(
    "Import pairs file",
    type=["txt", "csv"],
    key=k_live("pairs_file"),
    help="TXT: één symbol per regel of comma-separated. CSV: kolom 'symbol' of 'pairs'."
)
if pairs_file is not None:
    try:
        name = (pairs_file.name or "").lower()
        content = pairs_file.getvalue()

        # Avoid infinite rerun loop: file_uploader keeps the file on rerun.
        # Process each exact file only once per session.
        file_token = f"{pairs_file.name}:{len(content)}:{hashlib.md5(content).hexdigest()}"
        last_token_key = k_live("pairs_file_last_token")
        already = (st.session_state.get(last_token_key) == file_token)
        if already:
            st.sidebar.info("Pairs-bestand is al verwerkt in deze sessie (skip). Verwijder het bestand of upload een nieuwe versie om opnieuw te importeren.")
        else:
            st.session_state[last_token_key] = file_token

            imported = []
            if name.endswith(".txt"):
                s = content.decode("utf-8", errors="ignore")
                imported = [x.strip().upper() for x in re.split(r"[,;	 ]+", s) if x.strip()]
            elif name.endswith(".csv"):
                df_imp = pd.read_csv(io.BytesIO(content))
                col = None
                for c in ["symbol", "symbols", "pair", "pairs"]:
                    if c in df_imp.columns:
                        col = c
                        break
                if col is None:
                    raise ValueError("CSV mist kolom 'symbol' (of 'pairs').")
                vals = df_imp[col].astype(str).tolist()
                tmp = []
                for v in vals:
                    tmp.extend([x.strip().upper() for x in str(v).split(",") if x.strip()])
                imported = tmp

            imported = [x.replace("-", "/").upper() for x in imported]
            seen = set()
            imported_u = []
            for x in imported:
                if "/" not in x:
                    continue
                if x not in seen:
                    imported_u.append(x)
                    seen.add(x)

            if imported_u:
                st.session_state[k_live("symbols_pending")] = ", ".join(imported_u)
                st.sidebar.success(
                    f"Pairs klaar om te zetten: {', '.join(imported_u[:10])}" + (" ..." if len(imported_u) > 10 else "")
                )
                st.rerun()
            else:
                st.sidebar.warning("Geen geldige pairs gevonden in het bestand.")
    except Exception as e:
        st.sidebar.error(f"Import error: {e}")
# --- Bot Manager integration (virtual budgets per bot; Simulation/Dry-run only for now)
if lk("use_bot_manager") not in st.session_state:
    st.session_state[lk("use_bot_manager")] = False

use_bot_manager = st.sidebar.checkbox(
    "Use Bot Manager bots (recommended for multi-bot budgets)",
    key=lk("use_bot_manager"),
    help="Als dit aan staat worden bots geladen uit data/bots.json. Elke bot draait met een eigen virtueel budget (paper/dry-run)."
)

# Default pairs (only used when not using Bot Manager)
if lk("manual_symbols_input") not in st.session_state:
    st.session_state[lk("manual_symbols_input")] = "BTC/EUR, ETH/EUR"

bots = []
symbols_from_bots = []
if use_bot_manager:
    bots_all = load_bots()
    bots = [b for b in bots_all if bool(b.get('enabled', True))]
    if not bots:
        st.sidebar.warning("Geen enabled bots gevonden in Bot Manager. Ga naar pagina 'Bot Manager' en maak/enable bots.")
        st.stop()
    symbols_from_bots = [str(b.get('symbol','')).upper().replace('-', '/').strip() for b in bots]
    symbols_from_bots = [s for s in symbols_from_bots if s and '/' in s]
    symbols_from_bots = list(dict.fromkeys(symbols_from_bots))
    if not symbols_from_bots:
        st.sidebar.warning("Bots hebben geen geldige symbols.")
        st.stop()

    # Show bot pairs (read-only) — does not modify the manual pairs widget state
    st.sidebar.text_area("Bot pairs (from Bot Manager)", ", ".join(symbols_from_bots), height=90, disabled=True)
manual_symbols_input = st.sidebar.text_input("Pairs (comma-separated)", key=lk("manual_symbols_input"), disabled=use_bot_manager, on_change=_sync_prefs_from_state)
symbols = [s.strip().upper().replace('-', '/') for s in str(manual_symbols_input).split(',') if s.strip()]
# If Bot Manager is enabled, use symbols from bots (UI shows them in the input too)
if use_bot_manager and symbols_from_bots:
    symbols = symbols_from_bots
symbols = list(dict.fromkeys(symbols))  # de-dupe to avoid duplicate widget keys
if not symbols:
    st.sidebar.warning("No pairs selected. Fill 'Pairs (comma-separated)' with at least one symbol (e.g. BTC/EUR).")
    symbols = []
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)

# ----------------------------
# Profiles (simple): load a per-pair profile from the library
# ----------------------------
with st.sidebar.expander("Profiles (simple)", expanded=False):
    st.caption("Selecteer een opgeslagen Trainer-profiel en apply direct op de Live-config.")
    if symbols:
        sym_sel = st.selectbox("Symbol", options=symbols, key=k_live("profile_lib_symbol"))

        auto_apply = st.checkbox(
            "Auto-apply bij selectie",
            value=False,
            help="Als dit aan staat, wordt het gekozen library-profiel direct toegepast zodra je het selecteert.",
            key=k_live("profile_lib_auto_apply"),
        )

        def _apply_entry(_entry):
            cfg = load_library_profile(_entry)
            st.session_state.setdefault("pair_cfg", {})
            st.session_state["pair_cfg"].setdefault(sym_sel, {}).update(cfg)
            # Track which library profile is applied (for Live gating / diagnostics)
            st.session_state.setdefault("library_applied", {})
            st.session_state["library_applied"][sym_sel] = {
                "id": getattr(_entry, "id", None),
                "gates_passed": bool(getattr(_entry, "gates_passed", False)),
                "timeframe": timeframe,
                "created_at": getattr(_entry, "created_at", None),
            }
            health_note("Applied library profile", f"{sym_sel} {timeframe} {_entry.id}")
            st.toast(f"Applied profile for {sym_sel}", icon="✅")

        # Filter entries by symbol + timeframe (sorted newest first)
        entries = list_library_entries(symbol=sym_sel, timeframe=timeframe)

        if not entries:
            st.info("Geen library-profielen gevonden voor dit symbol/timeframe. Run Trainer met 'Auto-save PASS profiles to Library' aan.")
        else:
            def _fmt(e):
                tag = "PASS" if bool(e.gates_passed) else "FAIL"
                hint = "" if e.score_hint is None else f" | hint {float(e.score_hint):.2f}"
                return f"{e.created_at} | {tag}{hint} | id {e.id}"

            # --- Quick apply: latest PASS
            latest_pass = next((e for e in entries if bool(getattr(e, "gates_passed", False))), None)
            cols = st.columns([1, 1])
            with cols[0]:
                if st.button("Use latest PASS profile", width='stretch', disabled=(latest_pass is None)):
                    try:
                        _apply_entry(latest_pass)
                        st.success(f"Applied latest PASS profile for {sym_sel}.")
                    except Exception as e:
                        st.error(f"Could not apply latest PASS profile: {e}")
            with cols[1]:
                if st.button("Open Profile Library folder", width='stretch'):
                    st.info("Library staat in: data/profile_library/ (index.json + profiles/).")

            if latest_pass is None:
                st.warning("Geen PASS-profiel gevonden voor dit symbol/timeframe (wel entries aanwezig).")

            entry = st.selectbox("Library profile", options=entries, format_func=_fmt, key=k_live("profile_lib_entry"))

            # Manual apply button
            if st.button("Apply selected profile to session", width='stretch'):
                try:
                    _apply_entry(entry)
                    st.success(f"Applied library profile for {sym_sel}.")
                except Exception as e:
                    st.error(f"Could not apply library profile: {e}")

            # Auto-apply when selection changes
            sel_key = k_live("profile_lib_entry_last_id")
            last_id = st.session_state.get(sel_key)
            current_id = getattr(entry, "id", None)
            if auto_apply and current_id and current_id != last_id:
                try:
                    _apply_entry(entry)
                except Exception as e:
                    st.error(f"Auto-apply failed: {e}")
            st.session_state[sel_key] = current_id

refresh = st.sidebar.slider("Refresh sec", 5, 60, 30, help="Page auto-refresh interval. If the app feels like it keeps rerunning, disable auto-refresh below.")
enable_autorefresh = st.sidebar.checkbox("Auto-refresh page", value=True, help="If OFF, the page won't auto-rerun on a timer (useful while debugging or when runs take long).")
if enable_autorefresh:
    st_autorefresh(interval=refresh * 1000, key=k_live("refresh"))

# Persist Live UI prefs (survive reload/session reset)
_sync_prefs_from_state()

# --- Execution mode (no real orders are ever sent from this app unless live executor is added)
st.sidebar.subheader("Execution mode")
if len(symbols) > 1:
    st.sidebar.info("Meerdere bots tegelijk = meer API-calls. Gebruik caching/backoff en houd refresh ≥ 30–60s.")

exec_mode = st.sidebar.selectbox(
    "Mode",
    ["Simulation (paper, candle close)", "Dry-run Live (paper, ticker mid)", "Live (Bitvavo, real orders)"],
    index=0, key="exec_mode",
    help="Simulation and Dry-run Live never place real orders. Live (Bitvavo) can place REAL orders when explicitly enabled and acknowledged."
)
# Guard: Bot Manager virtual budgets currently supported only for Simulation/Dry-run
if use_bot_manager and exec_mode.startswith("Live"):
    st.sidebar.error("Live executor met virtuele bot-budget-reservering is nog niet ingeschakeld. Kies Simulation of Dry-run.")
    st.stop()

# Safety gates for Dry-run Live
allow_dryrun_without_active = st.sidebar.checkbox(
    "Allow Dry-run without ACTIVE bundle (not recommended)",
    value=False,
)

dryrun_allowed = True
active_bundle = None
if exec_mode.startswith("Dry-run"):
    ap = active_path()
    if (not allow_dryrun_without_active) and (not Path(ap).is_file()):
        dryrun_allowed = False
    elif Path(ap).is_file():
        try:
            active_bundle = load_bundle(ap)
            ok, errs, warns = validate_bundle(active_bundle)
            if warns:
                st.sidebar.warning("\n".join(warns))
            if (not ok) and (not allow_dryrun_without_active):
                dryrun_allowed = False
            else:
                meta = active_bundle.get("meta", {}) or {}
                if (not bool(meta.get("sanity_passed", False))) and (not allow_dryrun_without_active):
                    dryrun_allowed = False
                    st.sidebar.error("Dry-run Live gate: ACTIVE bundle not marked sanity_passed. Re-promote after a PASS sanity test.")
        except Exception as e:
            if not allow_dryrun_without_active:
                dryrun_allowed = False
                st.sidebar.error(f"Dry-run Live gate: could not load/validate ACTIVE bundle: {e}")

st.session_state['dryrun_allowed'] = bool(dryrun_allowed)
st.session_state['active_bundle'] = active_bundle

# ----------------------------
# Fees & slippage
# ----------------------------
st.sidebar.subheader("Fees & slippage")
fee_mode = st.sidebar.selectbox("Assume fills as", ["taker", "maker"], index=0)
tier_labels = [t[0] for t in FEE_TIERS_CAT_A]
tier_map = {t[0]: (t[1], t[2]) for t in FEE_TIERS_CAT_A}
tier_label = st.sidebar.selectbox("30d volume tier (Category A)", tier_labels, index=0)
maker_fee, taker_fee = tier_map[tier_label]

custom_fees = st.sidebar.checkbox("Override fees (custom)", value=False)
if custom_fees:
    maker_fee = st.sidebar.number_input("Maker fee (%)", 0.0, 1.0, float(maker_fee * 100), step=0.01) / 100.0
    taker_fee = st.sidebar.number_input("Taker fee (%)", 0.0, 1.0, float(taker_fee * 100), step=0.01) / 100.0

slippage_pct = st.sidebar.number_input("Slippage (%)", 0.0, 1.0, 0.05, step=0.01) / 100.0

# ----------------------------
# Risk (collapsible)
# ----------------------------
with st.sidebar.expander("Risk", expanded=False):

    # ----------------------------
    # Risk limits
    # ----------------------------
    st.subheader("Risk limits")
    default_cap = st.number_input("Max exposure per asset (EUR)", min_value=0.0, value=300.0, step=50.0)
    per_asset_caps = {}
    for sym in symbols:
        base = sym.split("/")[0]
        if base in per_asset_caps:
            continue
        per_asset_caps[base] = st.number_input(
            f"Cap {base} (EUR)", min_value=0.0, value=float(default_cap), step=50.0
        )

    # ----------------------------
    # Equity-based position scaling
    # ----------------------------
    st.subheader("Equity-based position scaling (simulation)")
    enable_scaling = st.checkbox("Enable equity-based scaling", value=False)
    scaling_mode = st.selectbox(
        "Scaling mode", ["Simple equity scaling", "ATR risk sizing"],
        index=0, disabled=not enable_scaling
    )
    min_order_size = st.number_input(
        "Min order size (base)", min_value=0.0, value=0.0001, format="%.6f",
        disabled=not enable_scaling
    )
    max_order_size = st.number_input(
        "Max order size (base)", min_value=0.0, value=0.01, format="%.6f",
        disabled=not enable_scaling
    )
    risk_per_trade_pct = st.slider(
        "Risk per trade (% equity)", 0.01, 2.00, 0.25, step=0.01,
        disabled=(not enable_scaling or scaling_mode != "ATR risk sizing")
    )
    atr_risk_mult = st.slider(
        "ATR risk multiplier", 0.5, 10.0, 3.0, step=0.5,
        disabled=(not enable_scaling or scaling_mode != "ATR risk sizing")
    )
    reset_baseline = st.button("Reset scaling baseline (start equity)", disabled=not enable_scaling)
    if reset_baseline:
        st.session_state.start_equity = None

    # ----------------------------
    # Portfolio risk: drawdown & correlation
    # ----------------------------
    st.subheader("Portfolio risk: drawdown & correlation")
    enable_dd_limit = st.checkbox("Enable max assets-in-drawdown", value=True)
    dd_asset_threshold_pct = st.slider(
        "Asset drawdown threshold (%)", 0.5, 50.0, 5.0, step=0.5, disabled=not enable_dd_limit
    )
    max_assets_in_dd = st.slider(
        "Max assets in drawdown", 0, 10, 2, step=1, disabled=not enable_dd_limit
    )

    enable_corr_filter = st.checkbox("Enable correlation filter", value=True)
    corr_window = st.slider(
        "Correlation window (candles)", 20, 300, 120, step=10, disabled=not enable_corr_filter
    )
    corr_threshold = st.slider(
        "Correlation threshold", 0.0, 0.99, 0.85, step=0.01, disabled=not enable_corr_filter
    )

    # ----------------------------
    # Stop-loss testing (simulation)
    # ----------------------------
    st.subheader("Stop-loss testing (simulation)")
    enable_portfolio_dd = st.checkbox("Enable portfolio drawdown stop", value=True)
    max_dd_pct = st.slider(
        "Max drawdown (%)", 1.0, 50.0, 10.0, step=0.5, disabled=not enable_portfolio_dd
    )
    dd_action_flatten = st.checkbox(
        "On portfolio stop: flatten all positions", value=True, disabled=not enable_portfolio_dd
    )

    enable_asset_stop = st.checkbox("Enable per-asset stop", value=True)
    asset_stop_pct = st.slider(
        "Per-asset stop from avg entry (%)", 0.5, 50.0, 8.0, step=0.5, disabled=not enable_asset_stop
    )

    use_atr_stop = st.checkbox("Also enable ATR-based stop", value=False, disabled=not enable_asset_stop)
    atr_mult = st.slider(
        "ATR multiple (stop = entry - m*ATR)", 0.5, 10.0, 3.0, step=0.5,
        disabled=(not enable_asset_stop or not use_atr_stop)
    )

# ----------------------------
# Interpretable execution (UI)
# ----------------------------
st.sidebar.subheader("Interpretable execution")
show_decision_log = st.sidebar.checkbox("Show decision log tables", value=True)
decision_log_rows = st.sidebar.slider("Decision log rows (per pair)", 50, 2000, 300, step=50)

with st.sidebar.expander("Metrics", expanded=False):

    # ----------------------------
    # Regime stability
    # ----------------------------
    st.subheader("Regime stability")
    cooldown_s = st.slider("Cooldown (seconds)", 0, 3600, 300, step=30)
    confirm_n = st.slider("Confirmations required", 1, 10, 3)

    # ----------------------------
    # Volatility clustering (metrics)
    # ----------------------------
    st.subheader("Volatility clustering (metrics)")
    vc_window = st.slider("VC window (candles)", 30, 300, 120, step=10)
    vc_alert = st.slider("VC alert threshold (ACF1)", 0.0, 0.99, 0.35, step=0.01)

    # ----------------------------
    # Range efficiency & streaks (metrics)
    # ----------------------------
    st.subheader("Range efficiency & streaks (metrics)")
    hit_window = st.slider("Hit-rate window (candles)", 20, 500, 150, step=10)
    streak_scope = st.slider("Streak scope (closed cycles)", 20, 2000, 300, step=20)

    st.subheader("Profiles import/export")

    # --- Import profiles.json safely (avoid rerun loops)
    if "profiles_import_hash" not in st.session_state:
        st.session_state.profiles_import_hash = None

    uploaded = st.file_uploader("Import profiles.json", type=["json"], key="profiles_uploader")
    import_clicked = False
    file_bytes = None
    file_hash = None
    if uploaded is not None:
        st.caption("Selecteer het bestand en klik daarna op **Import**.")
        import_clicked = st.button("Import profiles.json", width="stretch")
        try:
            file_bytes = uploaded.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
        except Exception:
            file_bytes = None
            file_hash = None

    if import_clicked and uploaded is not None:
        if file_hash is not None and file_hash == st.session_state.profiles_import_hash:
            st.info("Dit profielbestand is al geïmporteerd in deze sessie.")
        else:
            try:
                cfg_blob = json.loads(file_bytes.decode("utf-8")) if file_bytes is not None else json.load(uploaded)
                if not isinstance(cfg_blob, dict):
                    raise ValueError("profiles.json moet een dict zijn, bijv. {'BTC/EUR': {...}, 'ETH/EUR': {...}}")

                st.session_state.setdefault("pair_cfg", {})
                for sym, blob in cfg_blob.items():
                    sym_u = str(sym).upper()
                    st.session_state[s_live('pair_cfg')].setdefault(sym_u, {})
                    if isinstance(blob, dict):
                        st.session_state[s_live('pair_cfg')][sym_u]["use_regime_profiles"] = bool(blob.get("use_regime_profiles", True))
                        st.session_state[s_live('pair_cfg')][sym_u]["regime_profile_rebuild"] = bool(blob.get("regime_profile_rebuild", False))
                        if "regime_profiles" in blob and isinstance(blob["regime_profiles"], dict):
                            st.session_state[s_live('pair_cfg')][sym_u]["regime_profiles"] = blob["regime_profiles"]

                st.session_state.profiles_import_hash = file_hash
                st.success("Profiles geïmporteerd. Je kunt nu per pair de settings bekijken/aanpassen.")
            except Exception as e:
                st.error(f"Import failed: {e}")

    # --- Apply trained profiles from Trainer page (same session)
    if st.button("Apply BEST profiles from Trainer", width="stretch"):
        trained_best = st.session_state.get("trained_profiles_best")
        trained = trained_best if isinstance(trained_best, dict) and trained_best else st.session_state.get("trained_profiles")
        if isinstance(trained, dict) and trained:
            st.session_state.setdefault("pair_cfg", {})
            for sym, blob in trained.items():
                sym_u = str(sym).upper()
                st.session_state[s_live('pair_cfg')].setdefault(sym_u, {}).update(blob)
            st.success("Applied BEST trained profiles.")
        else:
            st.info("No trained profiles found in session. Run Trainer page first.")

    if st.button("Apply optimized profiles from Trainer", width="stretch"):
        trained = st.session_state.get("trained_profiles")
        if isinstance(trained, dict) and trained:
            st.session_state.setdefault("pair_cfg", {})
            for sym, blob in trained.items():
                sym_u = str(sym).upper()
                st.session_state[s_live('pair_cfg')].setdefault(sym_u, {}).update(blob)
            st.success("Applied trained profiles.")
        else:
            st.info("No trained profiles found in session. Run Trainer page first.")

    # --- Export current profiles (only the regime-related fields)
    def _export_profiles_sidebar():
        out = {}
        pair_cfg = st.session_state.get("pair_cfg", {})
        if isinstance(pair_cfg, dict):
            for sym, cfg in pair_cfg.items():
                if not isinstance(cfg, dict):
                    continue
                out[str(sym).upper()] = {
                    "use_regime_profiles": bool(cfg.get("use_regime_profiles", False)),
                    "regime_profile_rebuild": bool(cfg.get("regime_profile_rebuild", False)),
                    "regime_profiles": cfg.get("regime_profiles", {}),
                }
        return out

    export_payload = json.dumps(_export_profiles_sidebar(), indent=2)
    st.download_button(
        "Download current profiles.json",
        data=export_payload,
        file_name="profiles.json",
        width="stretch",
    )

# ----------------------------
# Per-pair grid settings
# ----------------------------
st.sidebar.subheader("Per-pair grid settings")

if "opt_suggestions" not in st.session_state:
    st.session_state.opt_suggestions = {}
if "last_hit_rate" not in st.session_state:
    st.session_state.last_hit_rate = {}

if s_live('pair_cfg') not in st.session_state:
    st.session_state[s_live('pair_cfg')] = {}

def default_cfg(sym: str):
    return {
        "grid_type": "Linear",
        "base_range_pct": 1.0,
        "dynamic_spacing": True,
        "k_range": 1.5,
        "k_levels": 0.7,
        "base_levels": 10,
        "order_size": 0.001,
        "price_decimals": 2,
        "intrabar_replay": True,
        "intrabar_path": "Standard (O-L-H-C)",
        "bar_fill_guard": True,
        "enable_cycle_tp": False,
        "cycle_tp_pct": 0.35,
        "enable_time_stop": True,
        "time_stop_hours": 48.0,
        "time_stop_mode": "DECAY_TO_TP",
        "time_stop_floor_tp_pct": 0.20,
        "trend_guard_enable": True,
        "trend_guard_lookback": 30,
        "trend_guard_thr_pct": 0.0,
        "auto_optimize": False,
        "opt_target_hit": 0.40,
        "opt_min_range_pct": 0.30,
        "opt_max_range_pct": 8.00,
        "opt_min_levels": 5,
        "opt_max_levels": 25,
        "enable_dyn_os_mult": False,
        "dyn_os_min_mult": 0.30,
        "dyn_os_max_mult": 1.50,
    }

for sym in symbols:
    if sym not in st.session_state[s_live('pair_cfg')]:
        st.session_state[s_live('pair_cfg')][sym] = default_cfg(sym)
    cfg = st.session_state[s_live('pair_cfg')][sym]

    # --- BB mean-reversion state (for buy-filter) ---
    try:
        if bool(cfg.get("bb_mr_enable", False)):
            w = int(cfg.get("bb_mr_window", 20))
            if w >= 2 and "close" in df.columns:
                mid = float(df["close"].rolling(w).mean().iloc[-1])
                std = float(df["close"].rolling(w).std(ddof=0).iloc[-1])
                st.session_state["bb_state"][sym] = {"enable": True, "mid": mid, "std": std, "thr": float(cfg.get("bb_mr_z", 0.75))}
            else:
                st.session_state["bb_state"][sym] = {"enable": False}
        else:
            st.session_state["bb_state"][sym] = {"enable": False}
    except Exception:
        st.session_state["bb_state"][sym] = {"enable": False}

    with st.sidebar.expander(sym, expanded=False):
        cfg["grid_type"] = st.selectbox(
            f"{sym} grid type", ["Linear", "Fibonacci"],
            index=0 if cfg["grid_type"] == "Linear" else 1,
            key=k_live(f"{sym}_grid_type")
        )
        cfg["base_range_pct"] = st.slider(
            f"{sym} base range ± (%)", 0.1, 20.0, float(cfg["base_range_pct"]), step=0.1, key=k_live(f"{sym}_range")
        )
        cfg["dynamic_spacing"] = st.checkbox(
            f"{sym} regime → dynamic spacing", value=bool(cfg["dynamic_spacing"]), key=k_live(f"{sym}_dyn")
        )
        cfg["k_range"] = st.slider(
            f"{sym} range multiplier strength", 0.5, 3.0, float(cfg["k_range"]), step=0.1, key=k_live(f"{sym}_krange")
        )
        cfg["k_levels"] = st.slider(
            f"{sym} levels reduction strength", 0.3, 1.0, float(cfg["k_levels"]), step=0.05, key=k_live(f"{sym}_klevels")
        )
        if cfg["grid_type"] == "Linear":
            cfg["base_levels"] = st.slider(
                f"{sym} base levels", 3, 30, int(cfg["base_levels"]), key=k_live(f"{sym}_levels")
            )
        # --- Regime profiles (live) ---
        st.markdown("**Regime profiles (live)**")
        cfg["use_regime_profiles"] = st.checkbox(
            f"{sym} Enable regime-conditional parameter sets",
            value=bool(cfg.get("use_regime_profiles", False)),
            key=k_live(f"{sym}_use_profiles"),
        )
        cfg["regime_profile_rebuild"] = st.checkbox(
            f"{sym} Rebuild on regime change (flatten + reset cycles)",
            value=bool(cfg.get("regime_profile_rebuild", False)),
            key=k_live(f"{sym}_prof_rebuild"),
            help="When effective regime changes: close that symbol's position (sim), rebuild grid at current price, reset cycles.",
        )

        if bool(cfg.get("use_regime_profiles", False)):
            for reg in ["RANGE", "TREND", "CHAOS", "WARMUP"]:
                prof = cfg.setdefault("regime_profiles", {}).setdefault(reg, {})
                with st.expander(f"{sym} {reg} profile", expanded=(reg == "RANGE")):
                    prof["range_pct"] = st.slider(
                        f"{sym} {reg} range ± (%)",
                        0.1, 20.0,
                        float(prof.get("range_pct", cfg.get("base_range_pct", 1.0))),
                        step=0.1,
                        key=k_live(f"{sym}_{reg}_rp"),
                    )
                    if cfg.get("grid_type") == "Linear":
                        prof["levels"] = st.slider(
                            f"{sym} {reg} levels (Linear)",
                            3, 30,
                            int(prof.get("levels", cfg.get("base_levels", 12))),
                            key=k_live(f"{sym}_{reg}_lv"),
                        )
                    prof["order_size_mult"] = st.slider(
                        f"{sym} {reg} order size mult",
                        0.1, 3.0,
                        float(prof.get("order_size_mult", 1.0)),
                        step=0.1,
                        key=k_live(f"{sym}_{reg}_osm"),
                    )
                    prof["cycle_tp_enable"] = st.checkbox(
                        f"{sym} {reg} enable Cycle TP",
                        value=bool(prof.get("cycle_tp_enable", False)),
                        key=k_live(f"{sym}_{reg}_ctp_en"),
                    )
                    prof["cycle_tp_pct"] = st.slider(
                        f"{sym} {reg} Cycle TP (%)",
                        0.05, 5.0,
                        float(prof.get("cycle_tp_pct", 0.35)),
                        step=0.05,
                        disabled=(not bool(prof.get("cycle_tp_enable", False))),
                        key=k_live(f"{sym}_{reg}_ctp"),
                    )

        st.divider()

        cfg["order_size"] = st.number_input(
            f"{sym} order size (base)", value=float(cfg["order_size"]),
            min_value=0.0, format="%.6f", key=k_live(f"{sym}_osize")
        )

        cfg["price_decimals"] = int(st.number_input(
            f"{sym} grid price decimals", min_value=0, max_value=8, value=int(cfg.get("price_decimals", 2)), step=1, key=k_live(f"{sym}_pdec")
        ))
        cfg["intrabar_replay"] = st.checkbox(
            f"{sym} intrabar replay (closed candle)", value=bool(cfg.get("intrabar_replay", True)), key=k_live(f"{sym}_intrabar")
        )
        cfg["intrabar_path"] = st.selectbox(
            f"{sym} intrabar path", ["Standard (O-L-H-C)", "Conservative (L-H-C)"],
            index=0 if str(cfg.get("intrabar_path", "Standard")).startswith("Standard") else 1,
            key=k_live(f"{sym}_intrapath"), disabled=(not bool(cfg.get("intrabar_replay", True)))
        )
        cfg["bar_fill_guard"] = st.checkbox(
            f"{sym} per-bar fill guard (avoid duplicate fills)", value=bool(cfg.get("bar_fill_guard", True)), key=k_live(f"{sym}_bar_guard")
        )

        st.markdown("**BB mean-reversion buy-filter**")
        cfg["bb_mr_enable"] = st.checkbox(
            f"{sym} BB mean-reversion filter", value=bool(cfg.get("bb_mr_enable", True)), key=k_live(f"{sym}_bbmr_en")
        )
        cfg["bb_mr_window"] = st.slider(
            f"{sym} BB window", 10, 60, int(cfg.get("bb_mr_window", 20)), key=k_live(f"{sym}_bbmr_w")
        )
        cfg["bb_mr_z"] = st.slider(
            f"{sym} Z-threshold (buy only if z <= -thr)", 0.0, 3.0, float(cfg.get("bb_mr_z", 0.75)), step=0.05, key=k_live(f"{sym}_bbmr_z")
        )
        cfg["enable_cycle_tp"] = st.checkbox(
            f"{sym} Cycle take-profit (per cycle)", value=bool(cfg.get("enable_cycle_tp", False)), key=k_live(f"{sym}_ctp_en")
        )
        cfg["cycle_tp_pct"] = st.slider(
            f"{sym} Cycle TP (%)", 0.05, 5.0, float(cfg.get("cycle_tp_pct", 0.35)), step=0.05,
            disabled=(not bool(cfg.get("enable_cycle_tp", False))), key=k_live(f"{sym}_ctp_pct")
        )

        st.markdown("**Inventory management (anti-hang)**")
        cfg["enable_time_stop"] = st.checkbox(
            f"{sym} Time-stop per cycle",
            value=bool(cfg.get("enable_time_stop", True)),
            help="Interpreteerbaar: als een cycle te lang open staat, mag hij sluiten op (net) break-even of met afbouwende TP.",
            key=k_live(f"{sym}_ts_en")
        )
        cfg["time_stop_hours"] = st.slider(
            f"{sym} Time-stop after (hours)", 1.0, 240.0, float(cfg.get("time_stop_hours", 48.0)), step=1.0,
            disabled=(not bool(cfg.get("enable_time_stop", True))), key=k_live(f"{sym}_ts_h")
        )
        cfg["time_stop_mode"] = st.selectbox(
            f"{sym} Time-stop mode", ["BREAK_EVEN_NET", "REDUCE_TO_TP", "DECAY_TO_TP"],
            index=["BREAK_EVEN_NET","REDUCE_TO_TP","DECAY_TO_TP"].index(str(cfg.get("time_stop_mode","DECAY_TO_TP")).upper() if str(cfg.get("time_stop_mode","DECAY_TO_TP")).upper() in ["BREAK_EVEN_NET","REDUCE_TO_TP","DECAY_TO_TP"] else "DECAY_TO_TP"),
            disabled=(not bool(cfg.get("enable_time_stop", True))), key=k_live(f"{sym}_ts_mode")
        )
        cfg["time_stop_floor_tp_pct"] = st.slider(
            f"{sym} Time-stop TP floor (%)", 0.0, 3.0, float(cfg.get("time_stop_floor_tp_pct", 0.20)), step=0.05,
            disabled=(not bool(cfg.get("enable_time_stop", True))) or str(cfg.get("time_stop_mode","")).upper()=="BREAK_EVEN_NET",
            key=k_live(f"{sym}_ts_floor")
        )

        cfg["trend_guard_enable"] = st.checkbox(
            f"{sym} Trend-guard (no buys in TREND-down)",
            value=bool(cfg.get("trend_guard_enable", True)),
            help="Interpreteerbaar: als regime TREND én de prijs over lookback daalt, blokkeer buys (sells blijven toegestaan).",
            key=k_live(f"{sym}_tg_en")
        )
        cfg["trend_guard_lookback"] = st.slider(
            f"{sym} Trend lookback (candles)", 5, 200, int(cfg.get("trend_guard_lookback", 30)), step=5,
            disabled=(not bool(cfg.get("trend_guard_enable", True))), key=k_live(f"{sym}_tg_lb")
        )
        cfg["trend_guard_thr_pct"] = st.slider(
            f"{sym} Downtrend threshold (%)", 0.0, 5.0, float(cfg.get("trend_guard_thr_pct", 0.0)), step=0.1,
            disabled=(not bool(cfg.get("trend_guard_enable", True))), key=k_live(f"{sym}_tg_thr")
        )

st.markdown("---")
cfg["auto_optimize"] = st.checkbox(
    f"{sym} Auto-optimize grid range/levels",
    value=bool(cfg.get("auto_optimize", False)),
    key=k_live(f"{sym}_autoopt"),
)
cfg["opt_target_hit"] = st.slider(
    f"{sym} Target hit-rate",
    0.10, 0.90, float(cfg.get("opt_target_hit", 0.40)),
    step=0.05,
    disabled=(not bool(cfg.get("auto_optimize", False))),
    key=k_live(f"{sym}_opt_hit"),
)
opt_apply = st.button(
    f"Apply suggested params for {sym}",
    key=k_live(f"{sym}_opt_apply"),
    disabled=(not bool(cfg.get("auto_optimize", False))),
    help="Applies the suggested range/levels to this pair's config."
)
if opt_apply:
    st.session_state[f"apply_opt_{sym}"] = True

cfg["enable_dyn_os_mult"] = st.checkbox(
    f"{sym} Dynamic order-size multiplier",
    value=bool(cfg.get("enable_dyn_os_mult", False)),
    key=k_live(f"{sym}_dynos"),
)
cfg["dyn_os_min_mult"] = st.slider(
    f"{sym} Dyn size min mult",
    0.1, 1.0, float(cfg.get("dyn_os_min_mult", 0.30)),
    step=0.05,
    disabled=(not bool(cfg.get("enable_dyn_os_mult", False))),
    key=k_live(f"{sym}_dynos_min"),
)
cfg["dyn_os_max_mult"] = st.slider(
    f"{sym} Dyn size max mult",
    1.0, 3.0, float(cfg.get("dyn_os_max_mult", 1.50)),
    step=0.10,
    disabled=(not bool(cfg.get("enable_dyn_os_mult", False))),
    key=k_live(f"{sym}_dynos_max"),
)

# ----------------------------
# Initialize portfolio trader
# ----------------------------

trader_signature = (exec_mode, maker_fee, taker_fee, slippage_pct, fee_mode, tuple(sorted(per_asset_caps.items())))

# Create or swap trader when configuration changes
if "trader_signature" not in st.session_state or st.session_state.trader_signature != trader_signature:
    st.session_state.trader_signature = trader_signature

    # Default: simulation trader
    if (not exec_mode.startswith("Live")) or (not st.session_state.get("live_enabled", False)):
        st.session_state.trader = PortfolioSimulatorTrader(
            cash_quote=1000.0,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            slippage=slippage_pct,
            fee_mode=fee_mode,
            quote_ccy="EUR",
            max_exposure_quote=per_asset_caps,
        )
    else:
        api_key = st.secrets.get("BITVAVO_API_KEY", "")
        api_secret = st.secrets.get("BITVAVO_API_SECRET", "")
        st.session_state.trader = BitvavoLiveTrader(
            api_key=api_key,
            api_secret=api_secret,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            slippage=slippage_pct,
            fee_mode=fee_mode,
            quote_ccy="EUR",
            max_exposure_quote=per_asset_caps,
            enable_market_orders=True,
            max_order_quote=float(st.session_state.get("live_max_order_quote", 50.0) or 0.0),
            max_retries=3,
        )

trader: PortfolioSimulatorTrader = st.session_state.trader

# Expose live trader ban status in health panel (if applicable)
try:
    if isinstance(trader, BitvavoLiveTrader):
        health_set("live_ban_until_ms", int(getattr(trader, "ban_until_ms", 0) or 0))
except Exception:
    pass

# ----------------------------
# Session state dicts
# ----------------------------
if s_live('engines') not in st.session_state:
    st.session_state[s_live('engines')] = {}
if s_live('regime_state') not in st.session_state:
    st.session_state[s_live('regime_state')] = {}
if s_live('portfolio_peak_eq') not in st.session_state:
    st.session_state[s_live('portfolio_peak_eq')] = None
if "dd_hist" not in st.session_state:
    st.session_state.dd_hist = []  # list of drawdown_pct floats
if "eq_hist" not in st.session_state:
    st.session_state.eq_hist = []  # list of equity floats
if s_live('portfolio_stop_active') not in st.session_state:
    st.session_state[s_live('portfolio_stop_active')] = False
if s_live('asset_halt') not in st.session_state:
    st.session_state[s_live('asset_halt')] = set()  # base assets halted due to stop
if s_live('pair_paused') not in st.session_state:
    st.session_state[s_live('pair_paused')] = set()  # symbols paused manually (no trading)
if s_live('last_bar_ts') not in st.session_state:
    st.session_state[s_live('last_bar_ts')] = {}  # symbol -> last processed closed bar timestamp

# --- Interpretable execution log (per pair)
if "decision_log" not in st.session_state:
    st.session_state.decision_log = {}  # sym -> deque of decision dicts

# ----------------------------
# Fetch data per pair
# ----------------------------
dfs: Dict[str, pd.DataFrame] = {}
last_prices: Dict[str, float] = {}
last_ts_map: Dict[str, pd.Timestamp] = {}
atr_abs: Dict[str, float] = {}  # per symbol
vol_cluster_map: Dict[str, float] = {}  # per symbol

for sym in symbols:
    if _bitvavo_is_banned():
        st.error(f"Data fetch paused due to Bitvavo ban: {sym}")
        continue
    try:
        health_set("last_public_ohlcv_attempt_ts", time.time())
        health_set("last_public_ohlcv_symbol", sym)
        df = fetch_ohlcv_bitvavo_cached(sym, timeframe=timeframe, limit=300)
    except BitvavoRateLimitBan as e:
        st.session_state['bitvavo_banned_until_ms'] = int(e.banned_until_ms)
        health_set("bitvavo_banned_until_ms", int(e.banned_until_ms))
        health_note("Bitvavo public ban", str(e))
        st.error(f"Data error for {sym}: {e}")
        continue
    except Exception as e:
        health_note("Public data error", f"{sym}: {e}")
        st.error(f"Data error for {sym}: {e}")
        continue
    dfs[sym] = df
    health_set("last_public_ohlcv_ok_ts", time.time())
    last_prices[sym] = float(df["close"].iloc[-1])
    if exec_mode.startswith("Dry-run"):
        try:
            health_set("last_public_ticker_attempt_ts", time.time())
            health_set("last_public_ticker_symbol", sym)
            t = fetch_ticker_bitvavo_cached(sym)
            bid = t.get("bid"); ask = t.get("ask"); last = t.get("last")
            if (bid is not None) and (ask is not None):
                last_prices[sym] = float((bid + ask) / 2.0)
            elif last is not None:
                last_prices[sym] = float(last)
            health_set("last_public_ticker_ok_ts", time.time())
        except BitvavoRateLimitBan as e:
            st.session_state['bitvavo_banned_until_ms'] = int(e.banned_until_ms)
            health_set("bitvavo_banned_until_ms", int(e.banned_until_ms))
            health_note("Bitvavo public ban", str(e))
        except Exception:
            pass
    last_ts_map[sym] = df["timestamp"].iloc[-1]
    # per-symbol health
    try:
        _sym_health = _health().get("symbols", {}) or {}
        _sym_health[str(sym)] = {
            "last_bar_ts_utc": str(last_ts_map[sym]),
            "last_close": float(last_prices[sym]),
        }
        health_set("symbols", _sym_health)
    except Exception:
        pass

# --- Bot Manager mode: run isolated bot portfolios (Simulation/Dry-run)
if use_bot_manager:
    if "bot_traders" not in st.session_state:
        st.session_state.bot_traders = {}
    if "bot_engines" not in st.session_state:
        st.session_state.bot_engines = {}

    global_fee_sig = (maker_fee, taker_fee, slippage_pct, fee_mode)
    bots_by_id = {str(x.get('id','')).strip(): x for x in (bots or [])}

    bot_summaries = {}
    for b in bots:
        bot_id = str(b.get("id","")).strip()
        if not bot_id:
            continue
        sym = str(b.get("symbol","")).upper().replace("-", "/").strip()
        if sym not in dfs:
            continue

        base = sym.split("/")[0]
        budget = float(b.get("budget_eur", 0.0) or 0.0)

        sig = (bot_id, budget) + global_fee_sig
        bt_key = f"bottrader:{bot_id}"
        if bt_key not in st.session_state.bot_traders or getattr(st.session_state.bot_traders[bt_key], "_sig", None) != sig:
            t = PortfolioSimulatorTrader(
                cash_quote=budget,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                slippage=slippage_pct,
                fee_mode=fee_mode,
                quote_ccy="EUR",
                max_exposure_quote={base: budget} if budget > 0 else {base: 0.0},
            )
            t._sig = sig
            st.session_state.bot_traders[bt_key] = t

        trader_b = st.session_state.bot_traders[bt_key]
        df = dfs[sym]
        price = float(df["close"].iloc[-1])
        ts = df["timestamp"].iloc[-1]

        grid_type = "Fibonacci" if str(b.get("grid_type","Linear")).lower().startswith("fib") else "Linear"
        base_range_pct = float(b.get("base_range_pct", 1.0))
        order_size = float(b.get("order_size_base", 0.0))
        base_levels = int(b.get("base_levels", 10))

        lower = price * (1 - base_range_pct / 100.0)
        upper = price * (1 + base_range_pct / 100.0)
        if grid_type == "Linear":
            grid = generate_linear_grid(lower, upper, max(3, base_levels))
        else:
            grid = generate_fibonacci_grid(lower, upper)

        eng_key = f"botengine:{bot_id}"
        eng_sig = (sym, grid_type, round(lower,2), round(upper,2), len(grid), float(order_size))
        if eng_key not in st.session_state.bot_engines or getattr(st.session_state.bot_engines[eng_key], "_signature", None) != eng_sig:
            eng = GridEngine(sym, grid, order_size)
            eng._signature = eng_sig
            st.session_state.bot_engines[eng_key] = eng

        eng = st.session_state.bot_engines[eng_key]

        # --- Bot-level trailing stop (inventory protection) + cooldown to avoid immediate re-buy
        if "bot_trailing_state" not in st.session_state:
            st.session_state.bot_trailing_state = {}

        tstate = (st.session_state.bot_trailing_state.get(bot_id, {}) or {}).copy()
        trailing_enabled = bool(b.get("trailing_enabled", False))
        activation_pct = float(b.get("trailing_activation_pct", 1.0) or 0.0)
        trail_pct = float(b.get("trailing_trail_pct", 3.0) or 0.0)
        cooldown_cfg = int(b.get("trailing_cooldown_bars", 12) or 0)

        # Maintain cooldown in bars (decrement when ts advances)
        last_ts_seen = tstate.get("last_ts_seen")
        if last_ts_seen != ts:
            tstate["last_ts_seen"] = ts
            tstate["cooldown_bars"] = max(0, int(tstate.get("cooldown_bars", 0) or 0) - 1)

        allow_buys = True
        if trailing_enabled and int(tstate.get("cooldown_bars", 0) or 0) > 0:
            allow_buys = False

        # Update trailing state + trigger flatten (paper/dry-run only)
        if trailing_enabled and trail_pct > 0:
            pos_amt = float(trader_b.positions.get(base, 0.0) or 0.0)
            avg_entry = trader_b.avg_entry_price(base)

            if pos_amt > 1e-12 and avg_entry is not None and float(avg_entry) > 0:
                # activation
                if not bool(tstate.get("active", False)):
                    if activation_pct <= 0.0 or float(price) >= float(avg_entry) * (1.0 + activation_pct / 100.0):
                        tstate["active"] = True
                        tstate["peak"] = float(price)
                else:
                    tstate["peak"] = max(float(tstate.get("peak", price) or price), float(price))

                # trigger
                if bool(tstate.get("active", False)):
                    peak = float(tstate.get("peak", price) or price)
                    stop_level = peak * (1.0 - float(trail_pct) / 100.0)
                    tstate["stop_level"] = float(stop_level)

                    if st.session_state.trading_enabled and float(price) <= float(stop_level):
                        # Close full position (single base in this bot)
                        pos_before = float(trader_b.positions.get(base, 0.0) or 0.0)
                        cb_before = float(trader_b.cost_basis.get(base, 0.0) or 0.0)
                        avg_cost = (cb_before / pos_before) if pos_before > 0 else 0.0

                        tr = trader_b.sell(sym, float(price), float(pos_before), ts, reason="TRAIL_STOP")
                        if tr is not None:
                            pnl = float(tr.cash_delta_quote) - float(avg_cost) * float(pos_before)
                            # Also push into engine trade log so UI charts/tables show the stop event
                            eng.trades.append({
                                "time": tr.time, "symbol": tr.symbol, "side": tr.side,
                                "price": float(tr.price), "amount": float(tr.amount),
                                "fee_rate": float(tr.fee_rate), "fee_paid": float(tr.fee_paid_quote),
                                "cash_delta": float(tr.cash_delta_quote),
                                "pnl": float(pnl),
                                "reason": tr.reason,
                            })

                        # Reset grid cycles to avoid stale open_cycles after forced flatten
                        try:
                            eng.reset_open_cycles()
                        except Exception:
                            pass

                        # Cooldown: pause buys for N bars; reset trailing tracker
                        tstate["cooldown_bars"] = int(cooldown_cfg)
                        tstate["active"] = False
                        tstate["peak"] = None

            else:
                # No inventory -> reset trailing tracker
                tstate["active"] = False
                tstate["peak"] = None
                tstate["stop_level"] = None
        else:
            # feature disabled
            tstate["active"] = False
            tstate["peak"] = None
            tstate["stop_level"] = None
            tstate["cooldown_bars"] = 0

        st.session_state.bot_trailing_state[bot_id] = tstate

        if st.session_state.trading_enabled:
            eng.check_price(price, trader_b, ts, allow_buys=allow_buys)

        eq_b = trader_b.equity({sym: price})
        bot_summaries[bot_id] = {
            "name": str(b.get("name", bot_id)),
            "symbol": sym,
            "budget": budget,
            "cash": trader_b.cash,
            "equity": eq_b,
            "pos_base": trader_b.positions.get(base, 0.0),
            "avg_entry": trader_b.avg_entry_price(base),
            "trades": len(eng.trades),
            "pnl_realized": sum(float(t.get("pnl",0.0)) for t in eng.trades if t.get("side")=="SELL"),
        }

    st.subheader("Bots (Bot Manager mode)")
    if not bot_summaries:
        st.info("Geen botdata beschikbaar. Controleer symbols en of data opgehaald kan worden.")
        st.stop()

    bot_df = pd.DataFrame([{"bot_id": k, **v} for k,v in bot_summaries.items()]).sort_values(["symbol","name"])
    bot_df["cash"] = bot_df["cash"].round(2)
    bot_df["equity"] = bot_df["equity"].round(2)
    bot_df["pos_base"] = bot_df["pos_base"].astype(float).round(6)
    bot_df["avg_entry"] = bot_df["avg_entry"].astype(float).round(2)
    bot_df["pnl_realized"] = bot_df["pnl_realized"].round(2)
    st.dataframe(bot_df.rename(columns={"bot_id":"bot","pnl_realized":"Realized PnL (EUR)","pos_base":"Pos (base)"}), width='stretch', height=240)

    tab_labels = [f"{row['name']} — {row['symbol']}" for _, row in bot_df.iterrows()]
    tabs = st.tabs(tab_labels)
    for (idx_row, row), tab in zip(bot_df.iterrows(), tabs):
        with tab:
            bot_id = row["bot_id"]
            sym = row["symbol"]
            base = sym.split("/")[0]
            df = dfs[sym]
            trader_b = st.session_state.bot_traders[f"bottrader:{bot_id}"]
            eng = st.session_state.bot_engines[f"botengine:{bot_id}"]

            st.caption(f"Budget: {row['budget']:.2f} EUR | Cash: {row['cash']:.2f} | Equity: {row['equity']:.2f} | Pos: {row['pos_base']:.6f} {base}")

            # Trailing stop status (if enabled for this bot)
            try:
                bcfg = bots_by_id.get(str(bot_id), {}) or {}
                if bool(bcfg.get("trailing_enabled", False)):
                    ts_state = (st.session_state.get("bot_trailing_state", {}) or {}).get(str(bot_id), {}) or {}
                    active = bool(ts_state.get("active", False))
                    peak = ts_state.get("peak")
                    stop_level = ts_state.get("stop_level")
                    cd = int(ts_state.get("cooldown_bars", 0) or 0)
                    act_pct = float(bcfg.get("trailing_activation_pct", 0.0) or 0.0)
                    tr_pct = float(bcfg.get("trailing_trail_pct", 0.0) or 0.0)
                    if peak is not None and stop_level is not None:
                        st.caption(
                            f"Trailing stop: {'ACTIVE' if active else 'idle'} | activation {act_pct:.2f}% | trail {tr_pct:.2f}% | "
                            f"peak {float(peak):.2f} | stop {float(stop_level):.2f} | cooldown {cd} bars"
                        )
                    else:
                        st.caption(
                            f"Trailing stop: {'ACTIVE' if active else 'idle'} | activation {act_pct:.2f}% | trail {tr_pct:.2f}% | cooldown {cd} bars"
                        )
            except Exception:
                pass

            fig = go.Figure(go.Candlestick(
                x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
            ))
            for lvl in eng.grid:
                fig.add_hline(y=lvl, line_dash="dot")
            for t in eng.trades[-300:]:
                symbol_m = "triangle-up" if t["side"]=="BUY" else "triangle-down"
                fig.add_scatter(x=[t["time"]], y=[t["price"]], mode="markers",
                                marker=dict(color="green" if t["side"]=="BUY" else "red", symbol=symbol_m, size=10),
                                name=t["side"])
            fig.update_layout(height=520, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, width='stretch')

            st.subheader("Trades")
            if eng.trades:
                tdf = pd.DataFrame(eng.trades).sort_values("time", ascending=False)
                for c in ["price","fee_paid","cash_delta","pnl"]:
                    if c in tdf.columns:
                        tdf[c] = pd.to_numeric(tdf[c], errors="coerce")
                if "fee_rate" in tdf.columns:
                    tdf["fee_rate_pct"] = (pd.to_numeric(tdf["fee_rate"], errors="coerce")*100).round(3)
                if "price" in tdf.columns:
                    tdf["price"] = tdf["price"].round(2)
                if "amount" in tdf.columns:
                    tdf["amount"] = pd.to_numeric(tdf["amount"], errors="coerce").round(6)
                if "pnl" in tdf.columns:
                    tdf["pnl"] = tdf["pnl"].round(2)
                show_cols = [c for c in ["time","side","price","amount","fee_rate_pct","fee_paid","cash_delta","pnl","reason"] if c in tdf.columns]
                st.dataframe(tdf[show_cols], width='stretch', height=320)
            else:
                st.info("Nog geen trades.")

    st.stop()

def compute_returns(df: pd.DataFrame) -> pd.Series:
    # log returns on close
    s = pd.Series(df["close"]).astype(float)
    return (s.apply(lambda x: math.log(x)).diff()).dropna()

def compute_metrics(df: pd.DataFrame, price: float) -> Tuple[float, float, float, float, float, str]:
    dfm = df.copy()
    dfm["atr"] = atr(dfm, 14)
    dfm["rv"] = realized_vol(dfm, 30)
    dfm["bb"] = bollinger_bandwidth(dfm, 20, 2.0)
    dfm["adx"] = adx(dfm, 14)

    def last_val(col):
        v = float(dfm[col].iloc[-1])
        return v if not math.isnan(v) else float("nan")

    atr_val = last_val("atr")
    rv_val = last_val("rv")
    bb_val = last_val("bb")
    adx_val = last_val("adx")
    atr_pct = (atr_val / price) if not math.isnan(atr_val) else float("nan")
    regime = classify_regime(dfm, atr_pct, rv_val, bb_val, adx_val)
    return atr_val, atr_pct, rv_val, bb_val, adx_val, regime

def apply_hysteresis(symbol: str, raw_regime: str) -> str:
    now = time.time()
    if symbol not in st.session_state[s_live('regime_state')]:
        st.session_state[s_live('regime_state')][symbol] = {
            "hist": deque(maxlen=confirm_n),
            "effective": raw_regime,
            "init_ts": now,
            "last_change": now
        }
    state = st.session_state[s_live('regime_state')][symbol]
    if state["hist"].maxlen != confirm_n:
        state["hist"] = deque(list(state["hist"]), maxlen=confirm_n)

    state["hist"].append(raw_regime)
    hist = list(state["hist"])
    confirmed = (len(hist) == confirm_n) and all(r == hist[0] for r in hist)
    if confirmed:
        candidate = hist[0]
        if candidate != state["effective"] and (now - state["last_change"]) >= cooldown_s:
            state["effective"] = candidate
            state["last_change"] = now
    return state["effective"]

# ----------------------------
# Correlation prep (rolling)
# ----------------------------
corr_matrix = None
if 'enable_corr_filter' in globals() and enable_corr_filter and len(dfs) >= 2:
    rets = {}
    for s, d in dfs.items():
        r = compute_returns(d)
        if len(r) >= corr_window:
            rets[s] = r.tail(corr_window)
    if len(rets) >= 2:
        corr_matrix = pd.DataFrame(rets).corr()

def regime_duration_minutes(symbol: str) -> float:
    state = st.session_state[s_live('regime_state')].get(symbol)
    if not state:
        return float("nan")
    lc = float(state.get("last_change", 0.0))
    if lc <= 0:
        lc = float(state.get("init_ts", 0.0))
    if lc <= 0:
        return float("nan")
    return (time.time() - lc) / 60.0

def compute_grid_hit_rate(df: pd.DataFrame, grid_levels, window_candles: int) -> float:
    """Range efficiency proxy: % of grid levels that were 'touched' by candle ranges in last window."""
    if df is None or df.empty or not grid_levels:
        return float("nan")
    w = int(max(1, window_candles))
    d = df.tail(w)
    if d.empty:
        return float("nan")
    levels = [float(x) for x in grid_levels]
    hits = set()
    # Candle 'touch' if level between low and high
    lows = d["low"].astype(float).to_numpy()
    highs = d["high"].astype(float).to_numpy()
    for lvl in levels:
        # vectorized-ish check
        for lo, hi in zip(lows, highs):
            if lo <= lvl <= hi:
                hits.add(lvl)
                break
    return float(len(hits) / max(1, len(levels)))

def compute_streaks(pnls):
    """Compute win/loss streak metrics from a list of realized PnL values (chronological)."""
    # classify: +1 win, -1 loss, 0 neutral
    cls = []
    for p in pnls:
        try:
            v = float(p)
        except Exception:
            continue
        if v > 1e-12:
            cls.append(1)
        elif v < -1e-12:
            cls.append(-1)
        else:
            cls.append(0)

    # win rate (exclude zeros)
    nz = [c for c in cls if c != 0]
    wins = sum(1 for c in nz if c == 1)
    losses = sum(1 for c in nz if c == -1)
    win_rate = float(wins / (wins + losses)) if (wins + losses) > 0 else float("nan")

    # streaks
    cur_type = 0
    cur_len = 0
    max_win = 0
    max_loss = 0

    def update_max(t, l):
        nonlocal max_win, max_loss
        if t == 1:
            max_win = max(max_win, l)
        elif t == -1:
            max_loss = max(max_loss, l)

    for c in cls:
        if c == 0:
            update_max(cur_type, cur_len)
            cur_type, cur_len = 0, 0
            continue
        if c == cur_type:
            cur_len += 1
        else:
            update_max(cur_type, cur_len)
            cur_type, cur_len = c, 1
    update_max(cur_type, cur_len)

    # current streak from the end
    end_type = 0
    end_len = 0
    for c in reversed(cls):
        if c == 0:
            break
        if end_type == 0:
            end_type = c
            end_len = 1
        elif c == end_type:
            end_len += 1
        else:
            break

    return {
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "cur_streak_type": "WIN" if end_type == 1 else ("LOSS" if end_type == -1 else "—"),
        "cur_streak_len": int(end_len),
        "max_win_streak": int(max_win),
        "max_loss_streak": int(max_loss),
    }

def propose_grid_params(
    df: pd.DataFrame,
    price: float,
    atr_val: float,
    grid_type: str,
    min_range_pct: float,
    max_range_pct: float,
    min_levels: int,
    max_levels: int,
    target_hit: float,
    window_candles: int,
):
    """Heuristic suggestion for grid range (±%) and (for Linear) number of levels.

    Objective:
    - Achieve a target grid "hit-rate" over the last N candles.
    - Keep spacing reasonable vs ATR (avoid too-tight/too-wide grids).
    - Avoid excessive levels (complexity penalty).

    Returns: (range_pct, levels_or_None, hit_rate, spacing_to_atr)
    """
    if df is None or df.empty:
        return float("nan"), None, float("nan"), float("nan")

    price = float(price)
    atr_val = float(atr_val) if atr_val is not None else float("nan")
    target_hit = float(target_hit)

    # Candidate ranges (coarse-to-fine)
    rngs = []
    r = float(min_range_pct)
    while r <= float(max_range_pct) + 1e-9:
        rngs.append(round(r, 2))
        r += 0.25 if r < 2.0 else (0.5 if r < 5.0 else 1.0)

    if str(grid_type) != "Linear":
        # Fibonacci: optimize range only
        best = (float("inf"), float("nan"), None, float("nan"), float("nan"))
        for rp in rngs:
            lo = price * (1.0 - rp / 100.0)
            hi = price * (1.0 + rp / 100.0)
            grid = generate_fibonacci_grid(lo, hi)
            hr = compute_grid_hit_rate(df, grid, window_candles=int(window_candles))
            score = abs(float(hr) - target_hit) if not math.isnan(hr) else 9e9
            if score < best[0]:
                best = (score, rp, None, hr, float("nan"))
        return best[1], best[2], best[3], best[4]

    # Linear: optimize range + levels
    levels_list = list(range(int(min_levels), int(max_levels) + 1))
    best = (float("inf"), float("nan"), None, float("nan"), float("nan"))

    for rp in rngs:
        lo = price * (1.0 - rp / 100.0)
        hi = price * (1.0 + rp / 100.0)
        width = hi - lo

        for lv in levels_list:
            if lv < 3:
                continue
            spacing = width / float(lv - 1)

            # ATR spacing penalty
            s2a = float("nan")
            pen = 0.0
            if (not math.isnan(atr_val)) and atr_val > 0:
                s2a = spacing / atr_val
                if s2a < 0.35:
                    pen += (0.35 - s2a) * 2.0
                if s2a > 2.5:
                    pen += (s2a - 2.5) * 1.0
            else:
                pen += 0.25

            grid = generate_linear_grid(lo, hi, int(lv))
            hr = compute_grid_hit_rate(df, grid, window_candles=int(window_candles))
            if math.isnan(hr):
                continue

            complexity = max(0.0, (float(lv) - 12.0) / 20.0)
            score = abs(float(hr) - target_hit) + pen + complexity

            if score < best[0]:
                best = (score, rp, int(lv), hr, s2a)

    return best[1], best[2], best[3], best[4]

def dyn_order_size_multiplier(
    eff_regime: str,
    hit_rate: float,
    vol_cluster: float,
    min_mult: float,
    max_mult: float,
) -> float:
    """Dynamic order size multiplier (risk-aware heuristic).

    - Reduce size in CHAOS and when volatility clustering is high.
    - Slightly reduce in TREND (avoid pyramiding trend risk).
    - Slightly increase in RANGE when hit-rate is healthy.

    Returns a clamped multiplier in [min_mult, max_mult].
    """
    m = 1.0

    if eff_regime == "CHAOS":
        m *= 0.60
    elif eff_regime == "TREND":
        m *= 0.80
    elif eff_regime == "RANGE":
        m *= 1.10

    if (not math.isnan(vol_cluster)) and vol_cluster >= 0.35:
        # stronger clustering => smaller size
        m *= max(0.5, 1.0 - min(0.5, (vol_cluster - 0.35)))

    if not math.isnan(hit_rate):
        if hit_rate < 0.20:
            m *= 0.70
        elif hit_rate > 0.55:
            m *= 1.10

    m = float(max(float(min_mult), min(float(max_mult), m)))
    return m

# ----------------------------
# STOP-LOSS CHECKS + PANIC FLATTEN
# ----------------------------
ts_any = next(iter(last_ts_map.values())) if last_ts_map else pd.Timestamp.utcnow()
eq = trader.equity(last_prices)
# --- Asset drawdown (unrealized vs avg entry) ---
asset_dd = {}  # base -> dd%
assets_in_dd = set()
for sym_, px_ in last_prices.items():
    base_ = sym_.split("/")[0]
    pos_ = trader.positions.get(base_, 0.0)
    if pos_ <= 1e-12:
        continue
    avg_ = trader.avg_entry_price(base_)
    if avg_ is None or avg_ <= 0:
        continue
    dd_pct_asset = max(0.0, (avg_ - float(px_)) / float(avg_) * 100.0)
    asset_dd[base_] = dd_pct_asset
    if 'enable_dd_limit' in globals() and enable_dd_limit and dd_pct_asset >= dd_asset_threshold_pct:
        assets_in_dd.add(base_)
dd_assets_count = len(assets_in_dd)

if st.session_state.start_equity is None:
    st.session_state.start_equity = float(eq) if eq > 0 else 1.0

# Execute panic flatten once prices are known (always)
if st.session_state.get("panic_flatten", False):
    trader.close_all(last_prices, ts_any, reason="PANIC_FLATTEN")
    for eng in st.session_state[s_live('engines')].values():
        eng.reset_open_cycles()
    st.session_state.panic_flatten = False
    st.session_state[s_live('portfolio_stop_active')] = True
    st.session_state.trading_enabled = False  # auto-pause after panic

# Peak equity / drawdown
if st.session_state[s_live('portfolio_peak_eq')] is None:
    st.session_state[s_live('portfolio_peak_eq')] = eq
else:
    st.session_state[s_live('portfolio_peak_eq')] = max(st.session_state[s_live('portfolio_peak_eq')], eq)

peak = st.session_state[s_live('portfolio_peak_eq')] or eq
dd = (peak - eq) / peak if peak > 0 else 0.0

# --- Equity / drawdown debug metrics
dd_eur = (peak - eq) if (peak is not None and peak > 0) else 0.0

# Exposure (mark-to-market) and unrealized PnL
total_exposure_eur = 0.0
total_unrealized_eur = 0.0
for sym, px in last_prices.items():
    base = sym.split("/")[0]
    pos = float(trader.positions.get(base, 0.0))
    if pos <= 1e-12:
        continue
    total_exposure_eur += pos * float(px)
    avg_entry = trader.avg_entry_price(base)
    if avg_entry is not None:
        total_unrealized_eur += (float(px) - float(avg_entry)) * pos

# History (for sparkline)
try:
    st.session_state.eq_hist.append(float(eq))
    st.session_state.dd_hist.append(float(dd * 100.0))
    # keep last 240 points
    if len(st.session_state.eq_hist) > 240:
        st.session_state.eq_hist = st.session_state.eq_hist[-240:]
    if len(st.session_state.dd_hist) > 240:
        st.session_state.dd_hist = st.session_state.dd_hist[-240:]
except Exception:
    pass

# Portfolio drawdown stop (auto-pause)
portfolio_stop_triggered = False
if enable_portfolio_dd and (dd * 100.0) >= max_dd_pct:
    st.session_state[s_live('portfolio_stop_active')] = True
    st.session_state.trading_enabled = False  # auto-pause on portfolio stop
    portfolio_stop_triggered = True

# Per-asset stop checks
asset_stops_triggered = []
if enable_asset_stop:
    for sym, px in last_prices.items():
        base = sym.split("/")[0]
        pos = trader.positions.get(base, 0.0)
        if pos <= 1e-12:
            continue
        avg_entry = trader.avg_entry_price(base)
        if avg_entry is None:
            continue

        stop_by_pct = px <= avg_entry * (1.0 - asset_stop_pct / 100.0)
        atr_val, _, _, _, _, _ = compute_metrics(dfs[sym], px)
        atr_abs[sym] = atr_val
        stop_by_atr = False
        if use_atr_stop and not math.isnan(atr_val):
            stop_by_atr = px <= (avg_entry - atr_mult * atr_val)

        if stop_by_pct or stop_by_atr:
            asset_stops_triggered.append((sym, base, px, avg_entry, atr_val))

# Execute stop actions
if portfolio_stop_triggered and dd_action_flatten:
    trader.close_all(last_prices, ts_any, reason="STOPLOSS_PORTFOLIO")
    for eng in st.session_state[s_live('engines')].values():
        eng.reset_open_cycles()

if asset_stops_triggered:
    for sym, base, px, avg_entry, atr_val in asset_stops_triggered:
        amt = trader.positions.get(base, 0.0)
        if amt <= 1e-12:
            continue
        trader.sell(sym, px, amt, last_ts_map.get(sym, ts_any), reason="STOPLOSS_ASSET")
        st.session_state[s_live('asset_halt')].add(base)
        if sym in st.session_state[s_live('engines')]:
            st.session_state[s_live('engines')][sym].reset_open_cycles()

# If portfolio stop active: disallow new buys globally
global_allow_buys = not st.session_state[s_live('portfolio_stop_active')]

# --- BUY guard: portfolio-level pre-trade filters ---
def buy_guard(symbol: str, amount_base: float, limit_price: float, ts):
    # 1) Max assets-in-drawdown (hard block on new buys once limit reached)
    if 'enable_dd_limit' in globals() and enable_dd_limit and max_assets_in_dd > 0:
        if dd_assets_count >= max_assets_in_dd:
            return False, f"DRAWDOWN_LIMIT: {dd_assets_count} >= {max_assets_in_dd} assets in drawdown"

    # 2) Correlation filter vs currently held assets (base positions > 0)
    if 'enable_corr_filter' in globals() and enable_corr_filter and corr_matrix is not None:
        held_symbols = []
        for b, amt in trader.positions.items():
            if amt > 1e-12:
                held_symbols.append(f"{b}/EUR")
        for hs in held_symbols:
            if hs == symbol:
                continue
            if (symbol in corr_matrix.index) and (hs in corr_matrix.columns):
                c = float(corr_matrix.loc[symbol, hs])
                if (not math.isnan(c)) and c >= corr_threshold:
                    return False, f"CORRELATION_LIMIT: corr({symbol},{hs})={c:.2f} >= {corr_threshold:.2f}"

    # 3) BB mean-reversion buy-filter (interpretable)
    try:
        bs = st.session_state["bb_state"].get(symbol)
        if bs and bool(bs.get("enable", False)):
            mid = float(bs.get("mid"))
            std = float(bs.get("std"))
            thr = float(bs.get("thr"))
            if (not math.isnan(mid)) and (not math.isnan(std)) and std > 0 and thr > 0:
                z = (float(limit_price) - mid) / std
                if z > -thr:
                    return False, f"BB_MR_BLOCK: z={z:.2f} > -{thr:.2f}"
    except Exception:
        pass

    return True, "OK"

# ----------------------------
# Run engines per pair
# ----------------------------
pair_summaries = {}

for sym, df in dfs.items():
    price = last_prices[sym]
    ts = last_ts_map[sym]
    cfg = st.session_state[s_live('pair_cfg')][sym]

    # --- BB mean-reversion state (for buy-filter) ---
    try:
        if bool(cfg.get("bb_mr_enable", False)):
            w = int(cfg.get("bb_mr_window", 20))
            if w >= 2 and "close" in df.columns:
                mid = float(df["close"].rolling(w).mean().iloc[-1])
                std = float(df["close"].rolling(w).std(ddof=0).iloc[-1])
                st.session_state["bb_state"][sym] = {"enable": True, "mid": mid, "std": std, "thr": float(cfg.get("bb_mr_z", 0.75))}
            else:
                st.session_state["bb_state"][sym] = {"enable": False}
        else:
            st.session_state["bb_state"][sym] = {"enable": False}
    except Exception:
        st.session_state["bb_state"][sym] = {"enable": False}

    # --- Effective order size (equity scaling) ---
    eff_order_size = float(cfg["order_size"])
    if "enable_scaling" in globals() and enable_scaling:
        if scaling_mode == "Simple equity scaling":
            start_eq = float(st.session_state.start_equity or 1.0)
            scale = (eq / start_eq) if start_eq > 0 else 1.0
            eff_order_size = float(cfg["order_size"]) * max(0.0, scale)
        else:
            # ATR risk sizing: size = (equity * risk%) / (ATR * multiplier)
            atr_tmp, *_ = compute_metrics(df, price)
            if atr_tmp is not None and (not math.isnan(float(atr_tmp))) and float(atr_tmp) > 0:
                risk_eur = float(eq) * (float(risk_per_trade_pct) / 100.0)
                eff_order_size = risk_eur / (float(atr_tmp) * float(atr_risk_mult))
        # clamps
        eff_order_size = max(float(min_order_size), min(float(max_order_size), float(eff_order_size)))

    atr_val, atr_pct, rv_val, bb_val, adx_val, raw_regime = compute_metrics(df, price)
    atr_abs[sym] = atr_val
    eff_regime = apply_hysteresis(sym, raw_regime)
    # --- Apply regime profile (interpretable, rule-based)
    profile = None
    if bool(cfg.get("use_regime_profiles", False)):
        profile = (cfg.get("regime_profiles") or {}).get(str(eff_regime))

    # Detect regime change for optional rebuild (flatten + reset cycles)
    if "last_eff_regime" not in st.session_state:
        st.session_state.last_eff_regime = {}
    prev_eff = st.session_state.last_eff_regime.get(sym)
    st.session_state.last_eff_regime[sym] = str(eff_regime)

    vc = vol_cluster_acf1(df, window=int(vc_window))
    vol_cluster_map[sym] = vc

    range_mult = 1.0
    levels_mult = 1.0

    if profile and bool(cfg.get('regime_profile_rebuild', False)) and prev_eff and prev_eff != str(eff_regime):
        base = sym.split('/')[0]
        amt = float(trader.positions.get(base, 0.0))
        if amt > 1e-12:
            trader.sell(sym, float(price), amt, ts, reason='REGIME_REBUILD_FLATTEN')
        # reset cycles; grid will be rebuilt via signature change below
        if sym in st.session_state[s_live('engines')]:
            st.session_state[s_live('engines')][sym].reset_open_cycles()
    if cfg["dynamic_spacing"] and eff_regime != "WARMUP":
        if eff_regime == "TREND":
            range_mult = cfg["k_range"]
            levels_mult = cfg["k_levels"]
        elif eff_regime == "CHAOS":
            range_mult = cfg["k_range"] * 1.5
            levels_mult = max(0.3, cfg["k_levels"] * 0.7)

    atr_floor_pct = 0.0
    if cfg["dynamic_spacing"] and not math.isnan(atr_pct):
        atr_floor_pct = max(0.0, 3.0 * atr_pct * 100.0)

    if profile:
        cfg_base_range = float(profile.get('range_pct', cfg.get('base_range_pct', 1.0)))
    else:
        cfg_base_range = float(cfg.get('base_range_pct', 1.0))
    eff_range_pct = max(cfg_base_range * range_mult, atr_floor_pct) if cfg['dynamic_spacing'] else cfg_base_range
    lower = price * (1 - eff_range_pct / 100.0)
    upper = price * (1 + eff_range_pct / 100.0)

    if cfg["grid_type"] == "Linear":
        if profile:
            cfg_base_levels = int(profile.get('levels', cfg.get('base_levels', 10)))
        else:
            cfg_base_levels = int(cfg.get('base_levels', 10))
        eff_levels = cfg_base_levels if not cfg['dynamic_spacing'] else max(3, int(round(cfg_base_levels * levels_mult)))
        grid = generate_linear_grid(lower, upper, eff_levels)
    else:
        eff_levels = None
        grid = generate_fibonacci_grid(lower, upper)

    # Dedupe/quantize grid levels to exchange-like price precision (reduces float duplicates).
    try:
        pdec = int(cfg.get("price_decimals", 2))
    except Exception:
        pdec = 2
    qgrid = sorted({round(float(x), pdec) for x in grid})
    if len(qgrid) >= 2:
        grid = qgrid

    # --- Engine/grid reuse: keep grid fixed and only rebuild when config/regime changes or price leaves bounds.
    cfg_sig = (
        sym, timeframe, cfg['grid_type'],
        float(cfg.get('base_range_pct', 1.0)),
        int(cfg.get('base_levels', 10)) if cfg['grid_type'] == 'Linear' else -1,
        float(cfg.get('order_size', 0.0)),
        bool(cfg.get('dynamic_spacing', True)),
        float(cfg.get('k_range', 1.5)),
        float(cfg.get('k_levels', 0.7)),
        str(eff_regime),
    )
    need_rebuild = False
    rebuild_reason = 'OK'
    if sym not in st.session_state[s_live('engines')]:
        need_rebuild = True
        rebuild_reason = 'INIT'
    else:
        eng_existing = st.session_state[s_live('engines')][sym]
        prev_sig = getattr(eng_existing, '_cfg_sig', None)
        if prev_sig != cfg_sig:
            need_rebuild = True
            rebuild_reason = 'CFG_OR_REGIME_CHANGE'
        else:
            bounds = getattr(eng_existing, '_bounds', None)
            if bounds and isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                lo_b, hi_b = float(bounds[0]), float(bounds[1])
                span = max(hi_b - lo_b, 1e-9)
                buf = 0.10 * span  # 10% buffer
                if float(price) < (lo_b - buf) or float(price) > (hi_b + buf):
                    need_rebuild = True
                    rebuild_reason = 'PRICE_OUTSIDE_BOUNDS'
            else:
                need_rebuild = True
                rebuild_reason = 'MISSING_BOUNDS'

    if need_rebuild:
        eng = GridEngine(sym, grid, cfg['order_size'])
        eng._cfg_sig = cfg_sig
        eng._bounds = (lower, upper)
        eng._last_rebuild_reason = rebuild_reason
        st.session_state[s_live('engines')][sym] = eng
    else:
        # keep existing grid; keep order_size updated if you tweak it live
        eng_existing = st.session_state[s_live('engines')][sym]
        eng_existing.order_size = float(cfg.get('order_size', eng_existing.order_size))

    eng: GridEngine = st.session_state[s_live('engines')][sym]
    # --- Range efficiency (hit-rate) ---
    hr = compute_grid_hit_rate(df, grid, window_candles=int(hit_window)) if "hit_window" in globals() else float("nan")
    st.session_state.last_hit_rate[sym] = float(hr)

    # --- Order size scaling (dynamic multiplier) ---
    dyn_mult = 1.0
    if bool(cfg.get("enable_dyn_os_mult", False)):
        dyn_mult = dyn_order_size_multiplier(
            eff_regime=str(eff_regime),
            hit_rate=float(hr),
            vol_cluster=float(vc),
            min_mult=float(cfg.get("dyn_os_min_mult", 0.30)),
            max_mult=float(cfg.get("dyn_os_max_mult", 1.50)),
        )

    eng.order_size = float(eff_order_size) * float(dyn_mult)

    eng.enable_cycle_tp = bool(cfg.get("enable_cycle_tp", False))
    eng.cycle_tp_pct = float(cfg.get("cycle_tp_pct", 0.35))

    eng.enable_time_stop = bool(cfg.get("enable_time_stop", True))
    eng.enable_bar_fill_guard = bool(cfg.get("bar_fill_guard", True))
    eng.time_stop_hours = float(cfg.get("time_stop_hours", 48.0))
    eng.time_stop_mode = str(cfg.get("time_stop_mode", "DECAY_TO_TP")).upper()
    eng.time_stop_floor_tp_pct = float(cfg.get("time_stop_floor_tp_pct", 0.20))

    base = sym.split("/")[0]
    trend_block = False
    trend_ret_pct = float("nan")
    if bool(cfg.get("trend_guard_enable", True)) and eff_regime == "TREND":
        lb = int(cfg.get("trend_guard_lookback", 30))
        thr = float(cfg.get("trend_guard_thr_pct", 0.0))
        if lb >= 2 and len(df) >= lb + 1:
            try:
                trend_ret_pct = (float(df["close"].iloc[-1]) / float(df["close"].iloc[-lb-1]) - 1.0) * 100.0
                if trend_ret_pct <= -thr:
                    trend_block = True
            except Exception:
                trend_block = False

    allow_buys = global_allow_buys and (base not in st.session_state[s_live('asset_halt')]) and (not trend_block)

    pair_is_paused = sym in st.session_state[s_live('pair_paused')]

    # --- Interpretable decision snapshot (before execution)
    if sym not in st.session_state.decision_log:
        st.session_state.decision_log[sym] = deque(maxlen=int(decision_log_rows) if "decision_log_rows" in globals() else 500)

    base_cap = per_asset_caps.get(base)
    exposure = float(trader.positions.get(base, 0.0)) * float(price)
    cap_remaining = (float(base_cap) - exposure) if (base_cap is not None) else float("nan")

    reasons = []
    if not st.session_state.trading_enabled:
        reasons.append("GLOBAL_STOPPED")
    if pair_is_paused:
        reasons.append("PAIR_PAUSED")
    if not global_allow_buys:
        reasons.append("PORTFOLIO_STOP_ACTIVE")
    if base in st.session_state[s_live('asset_halt')]:
        reasons.append("ASSET_HALT")
    if (base_cap is not None) and (cap_remaining <= 1e-6):
        reasons.append("CAP_REACHED")

    # Drawdown-limit state (hard block on new buys)
    if enable_dd_limit and max_assets_in_dd > 0:
        if dd_assets_count >= max_assets_in_dd:
            reasons.append("DRAWDOWN_LIMIT")

    # Correlation summary vs held assets
    corr_max = float("nan")
    corr_with = ""
    if enable_corr_filter and corr_matrix is not None:
        held_symbols = []
        for b, amt in trader.positions.items():
            if amt > 1e-12:
                held_symbols.append(f"{b}/EUR")
        for hs in held_symbols:
            if hs == sym:
                continue
            if (sym in corr_matrix.index) and (hs in corr_matrix.columns):
                c = float(corr_matrix.loc[sym, hs])
                if not math.isnan(c):
                    if math.isnan(corr_max) or c > corr_max:
                        corr_max = c
                        corr_with = hs
        if (not math.isnan(corr_max)) and corr_max >= float(corr_threshold):
            reasons.append(f"CORR_BLOCK:{corr_with}:{corr_max:.2f}")

    st.session_state.decision_log[sym].append({
        "time": ts,
        "price": float(price),
        "raw_regime": raw_regime,
        "eff_regime": eff_regime,
        "allow_buys": bool(allow_buys),
        "pos_base": float(trader.positions.get(base, 0.0)),
        "avg_entry": float(trader.avg_entry_price(base) or 0.0),
        "exposure_eur": exposure,
        "cap_eur": float(base_cap) if base_cap is not None else float("nan"),
        "cap_remaining_eur": cap_remaining,
        "assets_in_dd": int(dd_assets_count),
        "corr_max": corr_max,
        "corr_with": corr_with,
        "reasons": ";".join(reasons) if reasons else "OK",
    })

    if st.session_state.trading_enabled and (not pair_is_paused):
        # --- Intrabar bar-cross processing (captures level crossings that occur between refresh ticks)
        # We process the last CLOSED candle once per bar. This avoids missing crossings when using slower refresh intervals.
        if len(df) >= 3:
            closed = df.iloc[-2]  # penultimate candle is closed
            bar_ts = closed["timestamp"]
            prev_bar_ts = st.session_state[s_live('last_bar_ts')].get(sym)
            if (prev_bar_ts is None) or (bar_ts > prev_bar_ts):
                st.session_state[s_live('last_bar_ts')][sym] = bar_ts
                o = float(closed["open"]); h = float(closed["high"]); l = float(closed["low"]); c = float(closed["close"])
                # Intrabar replay: process the last CLOSED candle once per bar to capture level crossings.
                if bool(cfg.get("intrabar_replay", True)):
                    path = str(cfg.get("intrabar_path", "Standard (O-L-H-C)"))
                    if path.startswith("Conservative"):
                        pts = (l, h, c)  # fewer touches; avoids optimistic churn
                    else:
                        pts = (o, l, h, c)
                    for px in pts:
                        if st.session_state.trading_enabled and (not pair_is_paused):
                            eng.check_price(px, trader, bar_ts, allow_buys=allow_buys, buy_guard=buy_guard)
        

    # Buy guard: LIVE per-order cap (quote notional)
    def buy_guard(symbol: str, amount_base: float, limit_price: float, ts_):
        if exec_mode.startswith("Live") and st.session_state.get("live_enabled", False):
            cap = float(st.session_state.get("live_max_order_quote", 0.0) or 0.0)
            if cap > 0 and (float(limit_price) * float(amount_base)) > cap + 1e-9:
                return False, "LIVE_ORDER_CAP"
        return True, "OK"

        eng.check_price(price, trader, ts, allow_buys=allow_buys, buy_guard=buy_guard)

    # --- Range efficiency & streaks ---
    pnls = [c.get('pnl', 0.0) for c in eng.closed_cycles]
    if 'streak_scope' in globals():
        pnls = pnls[-int(streak_scope):]
    streak = compute_streaks(pnls)

    pair_summaries[sym] = {
        "price": price,
        "raw_regime": raw_regime,
        "eff_regime": eff_regime,
        "regime_dur_min": regime_duration_minutes(sym),
        "vol_cluster_acf1": float(vol_cluster_map.get(sym, float("nan"))),
        "hit_rate": float(hr),
        "dyn_os_mult": float(dyn_mult),
        "win_rate": float(streak["win_rate"]),
        "cur_streak": f"{streak['cur_streak_type']} {streak['cur_streak_len']}",
        "max_win_streak": int(streak["max_win_streak"]),
        "max_loss_streak": int(streak["max_loss_streak"]),
        "hit_rate": float(hr),
        "win_rate": float(streak["win_rate"]),
        "cur_streak": f"{streak['cur_streak_type']} {streak['cur_streak_len']}",
        "max_win_streak": int(streak["max_win_streak"]),
        "max_loss_streak": int(streak["max_loss_streak"]),
        "eff_range_pct": eff_range_pct,
        "levels": eff_levels,
        "order_size": float(eff_order_size),
        "cycle_tp_on": bool(cfg.get("enable_cycle_tp", False)),
        "cycle_tp_pct": float(cfg.get("cycle_tp_pct", 0.35)) if bool(cfg.get("enable_cycle_tp", False)) else 0.0,
        "pos_base": trader.positions.get(base, 0.0),
        "avg_entry": trader.avg_entry_price(base),
        "closed_pnl": sum(c["pnl"] for c in eng.closed_cycles),
        "trades": len(eng.trades),
        "halted": base in st.session_state[s_live('asset_halt')],
        "trend_blocked": bool(trend_block),
        "trend_ret_pct": trend_ret_pct,
        "paused": pair_is_paused,
        "asset_dd_pct": float(asset_dd.get(base, 0.0)),
        "in_drawdown": base in assets_in_dd,
    }

# ----------------------------
# Portfolio header
# ----------------------------
st.subheader("Portfolio")
colA, colB, colC, colD, colE, colF, colG = st.columns(7)
colA.metric("Cash (EUR)", f"{trader.cash:.2f}")
colB.metric("Equity (EUR)", f"{eq:.2f}")
colC.metric("Peak equity (EUR)", f"{peak:.2f}")
colD.metric("DD (EUR)", f"{dd_eur:.2f}")
colE.metric("DD (%)", f"{dd*100:.2f}%")
colF.metric("Exposure (EUR)", f"{total_exposure_eur:.2f}")
colG.metric("Unrealized PnL (EUR)", f"{total_unrealized_eur:.2f}")

with st.expander("Drawdown history (last points)", expanded=False):
    if st.session_state.dd_hist:
        st.line_chart(st.session_state.dd_hist, height=140)
    else:
        st.caption("No history yet.")

# --- Bitvavo LIVE account snapshot (balances + open orders)
# Show this panel whenever the user is in Live mode.
# Fetching balances/orders is read-only and useful even when the live executor itself
# is not armed yet.
if exec_mode.startswith("Live"):
    st.subheader("Bitvavo account (LIVE)")

    if "acc_last_balance" not in st.session_state:
        st.session_state["acc_last_balance"] = None
    if "acc_last_open_orders" not in st.session_state:
        st.session_state["acc_last_open_orders"] = None
    # Secrets presence indicator (helps debug "No balance data" situations)
    _has_key = bool(st.secrets.get("BITVAVO_API_KEY", ""))
    _has_secret = bool(st.secrets.get("BITVAVO_API_SECRET", ""))
    if not (_has_key and _has_secret):
        st.warning("Bitvavo API keys not found in Streamlit secrets. Add BITVAVO_API_KEY and BITVAVO_API_SECRET to enable authenticated account snapshot.")

    st.caption("Balances and open orders are fetched via authenticated API with caching/backoff. Open orders may be empty because Live executor v1 uses MARKET orders.")

    # Account snapshot controls
    acc_only_relevant = st.checkbox(
        "Show only relevant assets (EUR + bases from selected pairs)",
        value=bool(st.session_state.get("acc_only_relevant", True)),
        key="acc_only_relevant",
    )
    acc_auto_refresh = st.checkbox(
        "Auto-refresh account snapshot (on page refresh)",
        value=bool(st.session_state.get("acc_auto_refresh", False)),
        help="When OFF, balances/open orders stay cached until you click 'Fetch account snapshot'.",
        key="acc_auto_refresh",
    )
    acc_fetch = st.button(
        "Fetch account snapshot",
        help="Fetch balances + open orders now (uses caching/backoff).",
        key="acc_fetch_now",
        width="content",
    )

    # Remember the last snapshot/error so the user isn't left guessing.
    if "acc_last_error" not in st.session_state:
        st.session_state["acc_last_error"] = None
    if "acc_last_ok_ts" not in st.session_state:
        st.session_state["acc_last_ok_ts"] = None

    with st.expander("Account snapshot diagnostics", expanded=False):
        st.write({
            "trader_type": type(trader).__name__,
            "live_enabled": bool(st.session_state.get("live_enabled", False)),
            "has_api_key": bool(st.secrets.get("BITVAVO_API_KEY", "")),
            "has_api_secret": bool(st.secrets.get("BITVAVO_API_SECRET", "")),
            "auto_refresh": bool(st.session_state.get("acc_auto_refresh", False)),
            "last_ok_ts": st.session_state.get("acc_last_ok_ts"),
            "last_error": st.session_state.get("acc_last_error"),
        })

    cA, cB = st.columns([1, 1])

    # Use a dedicated authenticated client for account snapshot. This allows you to view balances
    # even if the live executor is not enabled yet.
    acc_trader = trader
    try:
        if not isinstance(acc_trader, BitvavoLiveTrader) and (_has_key and _has_secret):
            sig = (
                str(st.secrets.get("BITVAVO_API_KEY", "")),
                float(maker_fee), float(taker_fee), float(slippage_pct), str(fee_mode),
            )
            if st.session_state.get("acc_trader_sig") != sig or ("acc_trader" not in st.session_state):
                st.session_state["acc_trader_sig"] = sig
                st.session_state["acc_trader"] = BitvavoLiveTrader(
                    api_key=st.secrets.get("BITVAVO_API_KEY", ""),
                    api_secret=st.secrets.get("BITVAVO_API_SECRET", ""),
                    maker_fee=maker_fee,
                    taker_fee=taker_fee,
                    slippage=slippage_pct,
                    fee_mode=fee_mode,
                    quote_ccy="EUR",
                    max_exposure_quote=per_asset_caps,
                    enable_market_orders=False,  # read-only use in this panel
                    max_order_quote=0.0,
                    max_retries=3,
                )
            acc_trader = st.session_state.get("acc_trader") or trader
    except Exception:
        acc_trader = trader

    # Balances
    with cA:
        try:
            # Fetch balances only when requested, otherwise show cached snapshot
            do_fetch = bool(st.session_state.get("acc_auto_refresh", False)) or bool(acc_fetch)
            bal = None
            if hasattr(acc_trader, "get_balances"):
                if do_fetch or st.session_state.get("acc_last_balance") is None:
                    bal = acc_trader.get_balances()
                    st.session_state["acc_last_balance"] = bal
                    # Mark fetch attempt outcome
                    if bal:
                        st.session_state["acc_last_ok_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["acc_last_error"] = None
                    else:
                        st.session_state["acc_last_error"] = "Balance fetch returned empty result."
                else:
                    bal = st.session_state.get("acc_last_balance")
            else:
                bal = None
            if not bal:
                st.info("No balance data available.")
                st.caption(f"Debug: trader={type(trader).__name__}, live_enabled={bool(st.session_state.get('live_enabled', False))}, has_keys={_has_key and _has_secret}")
            else:
                free = (bal.get("free", {}) or {})
                used = (bal.get("used", {}) or {})
                total = (bal.get("total", {}) or {})
                # Some exchange implementations may omit 'total'. Compute it defensively.
                if not total and (free or used):
                    total = {}
                    for a in set(list(free.keys()) + list(used.keys())):
                        try:
                            total[a] = float(free.get(a, 0.0) or 0.0) + float(used.get(a, 0.0) or 0.0)
                        except Exception:
                            continue

                want_assets = set([quote_ccy])
                for sym in symbols:
                    want_assets.add(sym.split("/")[0].upper())

                rows = []
                for asset, tot in total.items():
                    try:
                        t = float(tot or 0.0)
                    except Exception:
                        continue
                    if (bool(st.session_state.get("acc_only_relevant", True)) and asset.upper() in want_assets) or (not bool(st.session_state.get("acc_only_relevant", True))):
                        rows.append({
                            "asset": asset.upper(),
                            "free": float(free.get(asset, 0.0) or 0.0),
                            "used": float(used.get(asset, 0.0) or 0.0),
                            "total": t,
                        })
                if rows:
                    bdf = pd.DataFrame(rows).sort_values("asset")
                    st.dataframe(bdf, width="stretch", height=260)
                else:
                    st.info("No balances to display.")
        except BitvavoRateLimitBanned as e:
            st.error(f"Rate-limit ban active (errorCode 105). Ban until: {e.ban_until_ms} (ms since epoch).")
        except Exception as e:
            st.error(f"Failed to fetch balances: {e}")

    # Open orders
    with cB:
        try:
            do_fetch = bool(st.session_state.get("acc_auto_refresh", False)) or bool(acc_fetch)
            orders = []
            if hasattr(acc_trader, "list_open_orders"):
                if do_fetch or st.session_state.get("acc_last_open_orders") is None:
                    orders = acc_trader.list_open_orders(symbols=symbols)
                    st.session_state["acc_last_open_orders"] = orders
                    st.session_state["acc_last_ok_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    # Note: empty open orders is not an error.
                else:
                    orders = st.session_state.get("acc_last_open_orders") or []
            else:
                orders = []
            if not orders:
                st.info("No open orders.")
            else:
                rows = []
                for o in orders:
                    try:
                        rows.append({
                            "symbol": o.get("symbol"),
                            "side": o.get("side"),
                            "type": o.get("type"),
                            "status": o.get("status"),
                            "price": o.get("price"),
                            "amount": o.get("amount"),
                            "filled": o.get("filled"),
                            "remaining": o.get("remaining"),
                            "time": o.get("datetime") or o.get("timestamp"),
                            "id": o.get("id"),
                        })
                    except Exception:
                        continue
                odf = pd.DataFrame(rows)
                if "symbol" in odf.columns:
                    odf = odf.sort_values(["symbol"])
                st.dataframe(odf, width="stretch", height=260)
        except BitvavoRateLimitBanned as e:
            st.error(f"Rate-limit ban active (errorCode 105). Ban until: {e.ban_until_ms} (ms since epoch).")
        except Exception as e:
            st.error(f"Failed to fetch open orders: {e}")

if st.session_state[s_live('asset_halt')]:
    st.warning(f"Asset halt active (no new buys): {', '.join(sorted(st.session_state[s_live('asset_halt')]))}")

summary_df = pd.DataFrame([{"symbol": k, **v} for k, v in pair_summaries.items()])
if (not summary_df.empty) and ("symbol" in summary_df.columns):
    summary_df = summary_df.sort_values("symbol")
else:
    # Defensive: avoid KeyError on empty/partial frames (e.g., if no pairs loaded successfully).
    summary_df = summary_df.copy()
if not summary_df.empty:
    # Defensive: ensure optional ML columns exist (older session states / partial data)
    for col in ["regime_dur_min", "vol_cluster_acf1", "hit_rate", "win_rate", "cur_streak", "max_win_streak", "max_loss_streak", "dyn_os_mult"]:
        if col not in summary_df.columns:
            summary_df[col] = float("nan")

    cols = [
        "symbol", "price", "eff_regime", "regime_dur_min", "vol_cluster_acf1", "hit_rate", "win_rate", "cur_streak",
        "eff_range_pct", "levels", "order_size", "dyn_os_mult",
        "pos_base", "avg_entry", "asset_dd_pct", "in_drawdown", "halted", "paused", "closed_pnl", "trades"
    ]
    show = summary_df.reindex(columns=cols).copy()

    show["price"] = show["price"].round(2)
    show["eff_range_pct"] = show["eff_range_pct"].round(2)
    show["regime_dur_min"] = show["regime_dur_min"].astype(float).round(1)
    show["vol_cluster_acf1"] = show["vol_cluster_acf1"].astype(float).round(2)
    show["hit_rate"] = (show["hit_rate"].astype(float) * 100.0).round(1)
    show["win_rate"] = (show["win_rate"].astype(float) * 100.0).round(1)
    show["pos_base"] = show["pos_base"].astype(float).round(6)
    show["avg_entry"] = show["avg_entry"].astype(float).round(2)
    show["dyn_os_mult"] = show["dyn_os_mult"].astype(float).round(2)
    show["asset_dd_pct"] = show["asset_dd_pct"].astype(float).round(2)
    show["closed_pnl"] = show["closed_pnl"].round(2)

    st.dataframe(
        show.rename(columns={
            "eff_range_pct": "Eff range ± (%)",
            "closed_pnl": "Realized PnL (EUR)",
            "pos_base": "Pos (base)",
            "asset_dd_pct": "Asset DD (%)",
            "in_drawdown": "In DD?",
            "regime_dur_min": "Regime duration (min)",
            "vol_cluster_acf1": "Vol cluster ACF1"
        }),
        width="stretch",
        height=240
    )

with st.expander("Correlation matrix (rolling)", expanded=False):
    if corr_matrix is None:
        st.caption("Correlation filter disabled or insufficient data.")
    else:
        st.dataframe(corr_matrix.round(2), width="stretch", height=220)

# ----------------------------
# Tabs per pair
# ----------------------------
if not dfs:
    st.error("No market data loaded for the selected pairs/timeframe. Check Bitvavo availability / rate limits, then try again.")
    st.stop()

tabs = st.tabs(list(dfs.keys()))
for i, sym in enumerate(dfs.keys()):
    with tabs[i]:
        df = dfs[sym]
        price = last_prices[sym]
        if sym not in st.session_state[s_live('engines')]:
            st.warning("Engine not initialized for this pair (data/initialization issue).")
            continue

        eng: GridEngine = st.session_state[s_live('engines')][sym]
        grid = eng.grid
        base = sym.split("/")[0]

        # --- Per-pair pause / resume ---
        p1, p2, p3 = st.columns([1, 1, 2])
        is_paused = sym in st.session_state[s_live('pair_paused')]
        with p1:
            if st.button("⏸ Pause pair", key=f"pause_{sym}", disabled=is_paused, width="stretch"):
                st.session_state[s_live('pair_paused')].add(sym)
                st.rerun()
        with p2:
            if st.button("▶ Resume pair", key=f"resume_{sym}", disabled=(not is_paused), width="stretch"):
                st.session_state[s_live('pair_paused')].discard(sym)
                st.rerun()
        with p3:
            pair_state = "PAUSED" if is_paused else "ACTIVE"
            global_state = "RUNNING" if st.session_state.trading_enabled else "STOPPED"
            st.caption(f"Pair status: {pair_state}  |  Global trading: {global_state}")
            with st.expander("Debug (why no trades?)", expanded=False):
                grid_sorted = sorted([float(x) for x in grid])
                p = float(price)
                next_buy = max([x for x in grid_sorted if x < p], default=None)
                next_sell = min([x for x in grid_sorted if x > p], default=None)
                gmin = (min(grid_sorted) if grid_sorted else None)
                gmax = (max(grid_sorted) if grid_sorted else None)
                gstep = (grid_sorted[1] - grid_sorted[0] if len(grid_sorted) >= 2 else None)
                bounds = getattr(eng, "_bounds", None)
                st.write("price:", p)
                st.write("grid_min/max:", gmin, gmax, "grid_step:", gstep)
                st.write("next_buy(<price):", next_buy, "next_sell(>price):", next_sell)
                st.write("bounds:", bounds, "last_rebuild_reason:", getattr(eng, "_last_rebuild_reason", "OK"))
                st.write("allow_buys:", bool(allow_buys), "pair_paused:", bool(is_paused), "portfolio_stop:", bool(st.session_state[s_live('portfolio_stop_active')]), "asset_halt:", (base in st.session_state[s_live('asset_halt')]))
                st.write("trades_logged:", len(getattr(eng, "trades", [])), "open_cycles:", len(getattr(eng, "open_cycles", dict())))
                st.write("active_buys:", len(getattr(eng, "active_buys", set())), "active_sells:", len(getattr(eng, "active_sells", set())))

        fig = go.Figure(go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
        ))
        for lvl in grid:
            fig.add_hline(y=lvl, line_dash="dot")

        for t in eng.trades[-400:]:
            marker_symbol = "triangle-up" if t["side"] == "BUY" else "triangle-down"
            fig.add_scatter(
                x=[t["time"]], y=[t["price"]],
                mode="markers",
                marker=dict(
                    color="green" if t["side"] == "BUY" else "red",
                    symbol=marker_symbol,
                    size=10
                ),
                name=f'{t["side"]}'
            )

        fig.update_layout(height=580, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")
        st.subheader("Open cycles / grid state")
        oc_rows = []
        for buy_level, oc in eng.open_cycles.items():
            buy_level = float(buy_level)
            sell_level = eng._next(buy_level) if buy_level in eng.grid[:-1] else float("nan")
            tp_price = (
                float(oc.buy_price) * (1.0 + float(getattr(eng, "cycle_tp_pct", 0.0)) / 100.0)
                if getattr(eng, "enable_cycle_tp", False)
                else float("nan")
            )
            unreal = (float(price) - float(oc.buy_price)) * float(oc.amount)
            oc_rows.append({
                "buy_level": buy_level,
                "buy_price": float(oc.buy_price),
                "amount": float(oc.amount),
                "cash_out": float(oc.cash_out),
                "sell_level": float(sell_level),
                "tp_price": (float(tp_price) if not math.isnan(tp_price) else None),
                "unreal_pnl": float(unreal),
                "buy_time": oc.buy_time,
            })

        if oc_rows:
            ocd = pd.DataFrame(oc_rows).sort_values("buy_time", ascending=False)
            ocd["buy_level"] = ocd["buy_level"].round(2)
            ocd["buy_price"] = ocd["buy_price"].round(2)
            ocd["sell_level"] = ocd["sell_level"].round(2)
            if "tp_price" in ocd.columns:
                ocd["tp_price"] = pd.to_numeric(ocd["tp_price"], errors="coerce").round(2)
            ocd["amount"] = ocd["amount"].round(6)
            ocd["cash_out"] = ocd["cash_out"].round(2)
            ocd["unreal_pnl"] = ocd["unreal_pnl"].round(2)
            st.dataframe(ocd, width="stretch", height=220)
        else:
            st.caption("No open cycles for this pair.")

        st.caption(
            f"Grid state: active buys={len(eng.active_buys)} | active sells={len(eng.active_sells)} | open cycles={len(eng.open_cycles)}"
        )

        st.subheader("Trades (executed, exact realized PnL on SELL)")
        if eng.trades:
            tdf = pd.DataFrame(eng.trades).sort_values("time", ascending=False)
            tdf["fee_rate_pct"] = (tdf["fee_rate"] * 100).round(3)
            tdf["price"] = tdf["price"].round(2)
            tdf["amount"] = tdf["amount"].round(6)
            tdf["fee_paid"] = tdf["fee_paid"].round(2)
            tdf["cash_delta"] = tdf["cash_delta"].round(2)
            tdf["pnl"] = tdf["pnl"].round(2)

            # cumulative realized pnl (SELL rows only, chronological)
            tdf_ch = tdf.iloc[::-1].copy()
            running = 0.0
            cum = []
            for _, row in tdf_ch.iterrows():
                if row["side"] == "SELL":
                    running += float(row["pnl"])
                cum.append(running)
            tdf_ch["cum_realized_pnl"] = pd.Series(cum, index=tdf_ch.index).round(2)
            tdf = tdf_ch.iloc[::-1]

            show_cols = ["time", "side", "price", "amount", "fee_rate_pct", "fee_paid", "cash_delta", "pnl", "cum_realized_pnl", "reason"]
            st.dataframe(
                tdf[show_cols].rename(columns={
                    "fee_rate_pct": "fee (%)",
                    "fee_paid": "fee paid (EUR)",
                    "cash_delta": "cash Δ (EUR)",
                    "pnl": "realized PnL (EUR)",
                    "cum_realized_pnl": "cum PnL (EUR)",
                }),
                width="stretch",
                height=320
            )
        else:
            st.info("Nog geen grid trades.")

        st.subheader("Order attempts blocked (risk/insufficient)")
        blocked = [t for t in trader.trades if t.symbol == sym and t.reason != "OK" and t.cash_delta_quote == 0.0]
        if blocked:
            bdf = pd.DataFrame([{
                "time": t.time, "side": t.side, "price": t.price, "amount": t.amount, "reason": t.reason
            } for t in blocked]).sort_values("time", ascending=False)
            st.dataframe(bdf, width="stretch", height=180)
        else:
            st.caption("Geen geblokkeerde orders voor deze pair.")

        avg_entry = trader.avg_entry_price(base)
        if avg_entry:
            st.caption(f"Position: {trader.positions.get(base, 0.0):.6f} {base} | Avg entry: {avg_entry:.2f} EUR")
        else:
            st.caption(f"Position: {trader.positions.get(base, 0.0):.6f} {base}")

st.caption("Stop-loss in simulatie: portfolio drawdown stop (optioneel flatten) + per-asset stop (avg-entry % en optioneel ATR). Reset session om stops te clearen.")

# ----------------------------
# Health panel (sidebar)
# ----------------------------
try:
    # Update order/trade telemetry (best-effort)
    tlist = list(getattr(trader, "trades", []) or [])
    health_set("trade_count_total", int(len(tlist)))
    if tlist:
        lt = tlist[-1]
        health_set("last_trade_ts", str(getattr(lt, "time", "")))
        health_set("last_trade_symbol", str(getattr(lt, "symbol", "")))
        health_set("last_trade_side", str(getattr(lt, "side", "")))
        health_set("last_trade_reason", str(getattr(lt, "reason", "")))
        try:
            health_set("last_trade_price", float(getattr(lt, "price", 0.0) or 0.0))
            health_set("last_trade_amount", float(getattr(lt, "amount", 0.0) or 0.0))
        except Exception:
            pass
except Exception:
    pass

with st.sidebar.expander("Health / diagnostics", expanded=False):
    h = _health()
    now = time.time()
    def _fmt_age(ts):
        try:
            ts = float(ts)
            if ts <= 0:
                return "-"
            return f"{int(max(0, now-ts))}s ago"
        except Exception:
            return "-"

    st.write({
        "last_rerun": _fmt_age(h.get("last_rerun_ts", 0)),
        "public_ohlcv": {
            "last_ok": _fmt_age(h.get("last_public_ohlcv_ok_ts", 0)),
            "last_attempt": _fmt_age(h.get("last_public_ohlcv_attempt_ts", 0)),
            "symbol": h.get("last_public_ohlcv_symbol", ""),
        },
        "public_ticker": {
            "last_ok": _fmt_age(h.get("last_public_ticker_ok_ts", 0)),
            "last_attempt": _fmt_age(h.get("last_public_ticker_attempt_ts", 0)),
            "symbol": h.get("last_public_ticker_symbol", ""),
        },
        "bitvavo_ban": {
            "public_ban_until_ms": int(h.get("bitvavo_banned_until_ms", 0) or 0),
            "live_ban_until_ms": int(h.get("live_ban_until_ms", 0) or 0),
        },
        "trades": {
            "count_total": int(h.get("trade_count_total", 0) or 0),
            "last": {
                "ts": h.get("last_trade_ts", ""),
                "symbol": h.get("last_trade_symbol", ""),
                "side": h.get("last_trade_side", ""),
                "reason": h.get("last_trade_reason", ""),
                "price": h.get("last_trade_price", None),
                "amount": h.get("last_trade_amount", None),
            },
        },
        "last_event": {
            "event": h.get("last_event", ""),
            "age": _fmt_age(h.get("last_event_ts", 0)),
            "detail": h.get("last_event_detail", ""),
        },
    })

    # Per-symbol snapshot (last candle / close)
    sym_info = h.get("symbols", {}) or {}
    if sym_info:
        st.caption("Per-symbol snapshot")
        st.dataframe(pd.DataFrame.from_dict(sym_info, orient="index"), width="stretch", height=180)