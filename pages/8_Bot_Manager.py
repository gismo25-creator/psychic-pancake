import time
import uuid
from typing import Any, Dict, List

import streamlit as st

from core.bots.store import load_bots, save_bots, upsert_bot, delete_bot

st.set_page_config(layout="wide")
st.title("Bot Manager (Virtual budgets, paper/dry-run)")

st.caption(
    "Hier beheer je bots als aparte objecten (naam, enable/disable, eigen budget en grid-parameters). "
    "In Live pagina kun je daarna 'Use Bot Manager bots' aanzetten om deze bots te draaien in Simulation/Dry-run."
)

if "bots" not in st.session_state:
    st.session_state.bots = load_bots()

bots: List[Dict[str, Any]] = st.session_state.bots

def _default_bot(symbol: str = "BTC/EUR") -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4())[:8],
        "name": f"bot_{int(time.time())}",
        "symbol": symbol.upper().replace("-", "/"),
        "enabled": True,
        "budget_eur": 500.0,
        "grid_type": "Linear",
        "base_range_pct": 1.0,
        "base_levels": 12,
        "order_size_base": 0.001,
        "dynamic_spacing": True,
        "tp_floor_pct": 0.35,
        "trailing_enabled": False,
        "trailing_activation_pct": 1.0,
        "trailing_trail_pct": 3.0,
        "trailing_cooldown_bars": 12,
    }

def _sanitize(bot: Dict[str, Any]) -> Dict[str, Any]:
    b = dict(bot)
    b["symbol"] = str(b.get("symbol","")).upper().replace("-", "/").strip()
    if "/" not in b["symbol"]:
        b["symbol"] = "BTC/EUR"
    b["name"] = str(b.get("name","")).strip() or b["symbol"]
    b["enabled"] = bool(b.get("enabled", True))
    b["budget_eur"] = float(b.get("budget_eur", 0.0) or 0.0)
    b["grid_type"] = "Fibonacci" if str(b.get("grid_type","Linear")).lower().startswith("fib") else "Linear"
    b["base_range_pct"] = float(b.get("base_range_pct", 1.0))
    b["base_levels"] = int(b.get("base_levels", 10))
    b["order_size_base"] = float(b.get("order_size_base", 0.0))
    b["dynamic_spacing"] = bool(b.get("dynamic_spacing", True))
    b["tp_floor_pct"] = float(b.get("tp_floor_pct", 0.0))
    b["trailing_enabled"] = bool(b.get("trailing_enabled", False))
    b["trailing_activation_pct"] = float(b.get("trailing_activation_pct", 1.0) or 0.0)
    b["trailing_trail_pct"] = float(b.get("trailing_trail_pct", 3.0) or 0.0)
    b["trailing_cooldown_bars"] = int(b.get("trailing_cooldown_bars", 12) or 0)
    return b

c1, c2, c3 = st.columns([1,1,3])
with c1:
    if st.button("Reload from disk", width="stretch"):
        st.session_state.bots = load_bots()
        st.rerun()
with c2:
    if st.button("Save to disk", width="stretch"):
        save_bots(st.session_state.bots)
        st.success("Saved to data/bots.json")

st.divider()

st.subheader("Add bot")
ac1, ac2, ac3, ac4, ac5 = st.columns([2,2,1,1,1])
with ac1:
    new_name = st.text_input("Name", value="")
with ac2:
    new_symbol = st.text_input("Symbol", value="BTC/EUR")
with ac3:
    new_budget = st.number_input("Budget (EUR)", min_value=0.0, value=500.0, step=50.0)
with ac4:
    new_enabled = st.checkbox("Enabled", value=True)
with ac5:
    if st.button("Add", width="stretch"):
        b = _default_bot(new_symbol)
        if new_name.strip():
            b["name"] = new_name.strip()
        b["enabled"] = bool(new_enabled)
        b["budget_eur"] = float(new_budget)
        b = _sanitize(b)
        st.session_state.bots = upsert_bot(st.session_state.bots, b)
        save_bots(st.session_state.bots)
        st.success(f"Added {b['name']} ({b['symbol']})")
        st.rerun()

st.divider()

st.subheader("Bots")
if not bots:
    st.info("Nog geen bots. Voeg er één toe hierboven.")
    st.stop()

rows = []
for b in bots:
    bb = _sanitize(b)
    rows.append({
        "id": bb["id"],
        "name": bb["name"],
        "symbol": bb["symbol"],
        "enabled": bb["enabled"],
        "budget_eur": bb["budget_eur"],
        "grid_type": bb["grid_type"],
        "range_pct": bb["base_range_pct"],
        "levels": bb["base_levels"],
        "order_size_base": bb["order_size_base"],
        "tp_floor_pct": bb["tp_floor_pct"],
        "trailing_enabled": bb.get("trailing_enabled", False),
        "trail_pct": bb.get("trailing_trail_pct", 0.0),
        "activation_pct": bb.get("trailing_activation_pct", 0.0),
    })
st.dataframe(rows, use_container_width=True, height=240)

for b in bots:
    bb = _sanitize(b)
    with st.expander(f"{bb['name']} — {bb['symbol']} ({'ENABLED' if bb['enabled'] else 'DISABLED'})", expanded=False):
        ec1, ec2, ec3, ec4 = st.columns([2,2,1,1])
        with ec1:
            bb["name"] = st.text_input("Name", value=bb["name"], key=f"name_{bb['id']}")
            bb["symbol"] = st.text_input("Symbol", value=bb["symbol"], key=f"sym_{bb['id']}")
        with ec2:
            bb["budget_eur"] = st.number_input("Budget (EUR)", min_value=0.0, value=float(bb["budget_eur"]), step=50.0, key=f"bud_{bb['id']}")
            bb["enabled"] = st.checkbox("Enabled", value=bool(bb["enabled"]), key=f"ena_{bb['id']}")
        with ec3:
            bb["grid_type"] = st.selectbox("Grid type", ["Linear","Fibonacci"], index=0 if bb["grid_type"]=="Linear" else 1, key=f"gt_{bb['id']}")
            bb["dynamic_spacing"] = st.checkbox("Dynamic spacing", value=bool(bb["dynamic_spacing"]), key=f"dyn_{bb['id']}")
        with ec4:
            bb["tp_floor_pct"] = st.number_input("TP floor (%)", min_value=0.0, value=float(bb["tp_floor_pct"]), step=0.05, key=f"tpf_{bb['id']}")

        ec5, ec6, ec7 = st.columns([1,1,2])
        with ec5:
            bb["base_range_pct"] = st.slider("Base range ± (%)", 0.1, 20.0, float(bb["base_range_pct"]), step=0.1, key=f"rng_{bb['id']}")
        with ec6:
            if bb["grid_type"] == "Linear":
                bb["base_levels"] = st.slider("Levels", 3, 40, int(bb["base_levels"]), key=f"lvl_{bb['id']}")
            else:
                st.caption("Fibonacci levels are implicit.")
        with ec7:
            bb["order_size_base"] = st.number_input("Order size (base)", min_value=0.0, value=float(bb["order_size_base"]), format="%.6f", key=f"os_{bb['id']}")

        st.markdown("### Trailing stop (inventory bescherming)")
        tc1, tc2, tc3, tc4 = st.columns([1.2, 1.2, 1.2, 1.4])
        with tc1:
            bb["trailing_enabled"] = st.checkbox(
                "Enable trailing stop",
                value=bool(bb.get("trailing_enabled", False)),
                key=f"tr_en_{bb['id']}",
                help="Als de prijs na activatie X% terugvalt vanaf de piek, wordt de volledige inventory gesloten (paper/dry-run)."
            )
        with tc2:
            bb["trailing_activation_pct"] = st.number_input(
                "Activation (%)",
                min_value=0.0, max_value=25.0,
                value=float(bb.get("trailing_activation_pct", 1.0) or 0.0),
                step=0.1,
                key=f"tr_act_{bb['id']}",
                help="Trailing start pas nadat de prijs ≥ avg-entry * (1 + activation%). 0 = direct actief zodra er inventory is."
            )
        with tc3:
            bb["trailing_trail_pct"] = st.number_input(
                "Trail distance (%)",
                min_value=0.1, max_value=50.0,
                value=float(bb.get("trailing_trail_pct", 3.0) or 0.0),
                step=0.1,
                key=f"tr_pct_{bb['id']}",
                help="Stop = peak_price * (1 - trail%). Bijvoorbeeld 3.0%."
            )
        with tc4:
            bb["trailing_cooldown_bars"] = st.number_input(
                "Cooldown (bars)",
                min_value=0, max_value=500,
                value=int(bb.get("trailing_cooldown_bars", 12) or 0),
                step=1,
                key=f"tr_cd_{bb['id']}",
                help="Aantal candles waarin BUYs gepauzeerd worden na een trailing-stop (voorkomt direct re-buy)."
            )


        bc1, bc2, bc3, bc4 = st.columns([1,1,1,3])
        with bc1:
            if st.button("Save", key=f"save_{bb['id']}", width="stretch"):
                bb = _sanitize(bb)
                st.session_state.bots = upsert_bot(st.session_state.bots, bb)
                save_bots(st.session_state.bots)
                st.success("Saved.")
                st.rerun()
        with bc2:
            if st.button("Duplicate", key=f"dup_{bb['id']}", width="stretch"):
                nb = dict(bb)
                nb["id"] = str(uuid.uuid4())[:8]
                nb["name"] = nb["name"] + "_copy"
                st.session_state.bots = upsert_bot(st.session_state.bots, nb)
                save_bots(st.session_state.bots)
                st.success("Duplicated.")
                st.rerun()
        with bc3:
            if st.button("Delete", key=f"del_{bb['id']}", width="stretch"):
                st.session_state.bots = delete_bot(st.session_state.bots, bb["id"])
                save_bots(st.session_state.bots)
                st.warning("Deleted.")
                st.rerun()
        with bc4:
            st.caption(f"ID: {bb['id']}")
