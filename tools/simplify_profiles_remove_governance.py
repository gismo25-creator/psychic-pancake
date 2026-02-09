#!/usr/bin/env python3
"""
Simplify profiles workflow:
1) Migrate data/profiles/active.json into the simple Profile Library so Live can select it.
2) Remove/disable the Profile Manager (Governance) page from Streamlit navigation.

Backups:
- streamlit_app.py.bak
- data/profile_library/index.json.bak (if exists)

Run from project root (folder containing streamlit_app.py):
    python tools/simplify_profiles_remove_governance.py
"""
from __future__ import annotations
from pathlib import Path
import json, re, shutil, datetime

ROOT = Path(".")
ACTIVE_PATH = ROOT / "data" / "profiles" / "active.json"
LIB_DIR = ROOT / "data" / "profile_library"
LIB_PROFILES_DIR = LIB_DIR / "profiles"
LIB_INDEX_PATH = LIB_DIR / "index.json"
APP_PATH = ROOT / "streamlit_app.py"

def _utcnow():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def migrate_active_to_library() -> bool:
    if not ACTIVE_PATH.exists():
        print(f"SKIP: {ACTIVE_PATH} not found.")
        return False

    active = json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))
    meta = active.get("meta") or {}
    tf = str(meta.get("timeframe") or "15m")
    profiles = active.get("profiles") or {}

    LIB_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    # load existing index
    if LIB_INDEX_PATH.exists():
        try:
            idx = json.loads(LIB_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            idx = {"version": 1, "items": []}
    else:
        idx = {"version": 1, "items": []}

    items = idx.get("items") or []
    existing_ids = {it.get("id") for it in items if isinstance(it, dict)}

    wrote_any = False
    for symbol, cfg in profiles.items():
        entry_id = f"ACTIVE_MIGRATED::{symbol}::{tf}"
        filename = f"{symbol.replace('/','-')}__{tf}__ACTIVE_MIGRATED.json"
        out_path = LIB_PROFILES_DIR / filename

        out = {
            "id": entry_id,
            "created_at": _utcnow(),
            "source": "active.json",
            "symbol": symbol,
            "timeframe": tf,
            "gates_passed": bool(meta.get("gates_passed", True)),
            "meta": {
                "schema_version": active.get("schema_version", 1),
                "active_created_at": active.get("created_at"),
                "trainer_gates": meta.get("gates", {}),
                "fees": meta.get("fees", {}),
            },
            "profile": cfg,
        }

        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

        if entry_id not in existing_ids:
            items.append({
                "id": entry_id,
                "symbol": symbol,
                "timeframe": tf,
                "gates_passed": out["gates_passed"],
                "path": str(out_path.as_posix()),
                "created_at": out["created_at"],
                "source": "active.json",
            })
            existing_ids.add(entry_id)

        wrote_any = True
        print(f"WROTE: {out_path}")

    # persist index with backup
    if LIB_INDEX_PATH.exists():
        shutil.copy2(LIB_INDEX_PATH, LIB_INDEX_PATH.with_suffix(".json.bak"))
    idx["items"] = items
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    LIB_INDEX_PATH.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: updated {LIB_INDEX_PATH}")
    return wrote_any

def patch_streamlit_app_remove_governance() -> bool:
    if not APP_PATH.exists():
        print(f"SKIP: {APP_PATH} not found.")
        return False

    src = APP_PATH.read_text(encoding="utf-8", errors="replace")
    backup = APP_PATH.with_suffix(".py.bak")
    shutil.copy2(APP_PATH, backup)

    patterns = [
        re.compile(r"^.*Profile Manager.*$\n?", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^.*Governance.*$\n?", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^.*pages/.*profile.*manager.*$\n?", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^.*pages\\.*profile.*manager.*$\n?", re.IGNORECASE | re.MULTILINE),
    ]
    patched = src
    for pat in patterns:
        patched = pat.sub("", patched)

    patched = re.sub(r"\n{3,}", "\n\n", patched)

    if patched != src:
        APP_PATH.write_text(patched, encoding="utf-8")
        print(f"OK: patched {APP_PATH} (backup at {backup})")
        return True
    else:
        print("NOTE: No governance page registration lines matched in streamlit_app.py (nothing removed).")
        return False

def disable_page_file_if_exists() -> bool:
    pages_dir = ROOT / "pages"
    if not pages_dir.exists():
        return False

    changed = False
    for p in pages_dir.glob("*.py"):
        name = p.name.lower()
        if ("profile" in name and "manager" in name) or ("governance" in name):
            newp = p.with_suffix(".py.disabled")
            try:
                p.rename(newp)
                print(f"RENAMED: {p} -> {newp}")
                changed = True
            except Exception as e:
                print(f"WARN: could not rename {p}: {e}")
    return changed

def main():
    print("== Migrating active.json -> Profile Library ==")
    migrate_active_to_library()
    print("\n== Disabling Profile Manager (Governance) page ==")
    patch_streamlit_app_remove_governance()
    disable_page_file_if_exists()
    print("\nDONE. Restart Streamlit:\n  streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
