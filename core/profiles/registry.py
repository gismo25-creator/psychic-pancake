import json
import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, List, Optional

import pandas as pd

DEFAULT_STORE_DIR = os.path.join("data", "profiles")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_store_dir(path: str = DEFAULT_STORE_DIR) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def stable_hash_df(df: pd.DataFrame) -> str:
    """Hash a candle dataframe in a stable way (timestamp + close)."""
    if df is None or df.empty:
        return "EMPTY"
    tmp = df[["timestamp", "close"]].copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce").fillna(0.0).round(8)
    payload = tmp.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def make_bundle(trained: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "meta": meta,
        "profiles": trained,
    }


def save_bundle(bundle: Dict[str, Any], store_dir: str = DEFAULT_STORE_DIR, name: Optional[str] = None) -> str:
    ensure_store_dir(store_dir)
    safe = (name or f"profile_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}").replace(" ", "_")
    fname = safe if safe.endswith(".json") else safe + ".json"
    path = os.path.join(store_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, sort_keys=False, default=str)
    return path


def list_bundles(store_dir: str = DEFAULT_STORE_DIR) -> List[str]:
    if not os.path.isdir(store_dir):
        return []
    files = [f for f in os.listdir(store_dir) if f.lower().endswith(".json")]
    files.sort(reverse=True)
    return [os.path.join(store_dir, f) for f in files]


def load_bundle(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_bundle(bundle: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Return (ok, errors, warnings)."""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(bundle, dict):
        return False, ["Bundle is not a dict"], []
    if bundle.get("schema_version") != 1:
        warnings.append(f"Unknown schema_version: {bundle.get('schema_version')}")
    if "profiles" not in bundle or not isinstance(bundle["profiles"], dict) or not bundle["profiles"]:
        errors.append("Missing or empty 'profiles' dict")

    meta = bundle.get("meta", {})
    if not isinstance(meta, dict):
        warnings.append("meta is not a dict")

    prof = bundle.get("profiles", {})
    for sym, cfg in prof.items():
        if not isinstance(cfg, dict):
            errors.append(f"{sym}: cfg is not a dict")
            continue

        if "order_size" in cfg:
            try:
                osz = float(cfg["order_size"])
                if osz <= 0:
                    errors.append(f"{sym}: order_size <= 0")
                if osz > 10:
                    warnings.append(f"{sym}: unusually large order_size ({osz})")
            except Exception:
                errors.append(f"{sym}: order_size not numeric")

        if "base_range_pct" in cfg:
            try:
                r = float(cfg["base_range_pct"])
                if r <= 0:
                    errors.append(f"{sym}: base_range_pct <= 0")
                if r > 50:
                    warnings.append(f"{sym}: base_range_pct > 50% ({r})")
            except Exception:
                errors.append(f"{sym}: base_range_pct not numeric")

        if "base_levels" in cfg:
            try:
                lv = int(cfg["base_levels"])
                if lv < 2:
                    errors.append(f"{sym}: base_levels < 2")
                if lv > 200:
                    warnings.append(f"{sym}: base_levels > 200 ({lv})")
            except Exception:
                warnings.append(f"{sym}: base_levels not int")

        if cfg.get("use_regime_profiles") and "regime_profiles" not in cfg:
            errors.append(f"{sym}: use_regime_profiles enabled but regime_profiles missing")

    ok = len(errors) == 0
    return ok, errors, warnings


def diff_profiles(current_pair_cfg: Dict[str, Any], new_profiles: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"added_symbols": [], "removed_symbols": [], "changed": {}}
    cur_syms = set((current_pair_cfg or {}).keys())
    new_syms = set((new_profiles or {}).keys())

    out["added_symbols"] = sorted(list(new_syms - cur_syms))
    out["removed_symbols"] = sorted(list(cur_syms - new_syms))

    for sym in sorted(list(cur_syms & new_syms)):
        cur = current_pair_cfg.get(sym, {}) or {}
        new = new_profiles.get(sym, {}) or {}
        changed = {}
        keys = set(cur.keys()) | set(new.keys())
        for k in sorted(list(keys)):
            if cur.get(k) != new.get(k):
                changed[k] = {"from": cur.get(k), "to": new.get(k)}
        if changed:
            out["changed"][sym] = changed
    return out
