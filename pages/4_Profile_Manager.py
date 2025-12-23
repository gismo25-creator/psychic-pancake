import json
from pathlib import Path

import streamlit as st

from core.profiles.registry import list_bundles, load_bundle, validate_bundle, diff_profiles, save_bundle, ensure_store_dir

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

st.subheader("Apply to session")
confirm = st.checkbox("I understand this overwrites per-symbol settings in this session.")
if st.button("Apply bundle", disabled=(not ok or not confirm)):
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
