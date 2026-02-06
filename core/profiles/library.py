"""Simple profile library.

Lightweight alternative to the governance/ACTIVE workflow.

- Trainer can save good profiles here.
- Live can load a profile for a selected pair.

Files:
  data/profile_library/index.json
  data/profile_library/profiles/<symbol>__<timeframe>__<created_at>__<id>.json
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path("data/profile_library")
INDEX_PATH = BASE_DIR / "index.json"
PROFILES_DIR = BASE_DIR / "profiles"


@dataclass
class LibraryEntry:
    id: str
    symbol: str
    timeframe: str
    created_at: str
    source_bundle: str
    gates_passed: bool
    score_hint: Optional[float]
    filename: str


def ensure_dirs() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text(json.dumps({"entries": []}, indent=2), encoding="utf-8")


def _read_index() -> Dict[str, Any]:
    ensure_dirs()
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"entries": []}


def _write_index(doc: Dict[str, Any]) -> None:
    ensure_dirs()
    INDEX_PATH.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def list_entries(symbol: Optional[str] = None, timeframe: Optional[str] = None) -> List[LibraryEntry]:
    doc = _read_index()
    out: List[LibraryEntry] = []
    for raw in (doc.get("entries") or []):
        try:
            e = LibraryEntry(**raw)
        except Exception:
            continue
        if symbol and e.symbol.upper() != symbol.upper():
            continue
        if timeframe and e.timeframe != timeframe:
            continue
        out.append(e)
    out.sort(key=lambda x: x.created_at, reverse=True)
    return out


def save_profile(
    symbol: str,
    timeframe: str,
    profile_cfg: Dict[str, Any],
    created_at: str,
    source_bundle: str,
    gates_passed: bool,
    score_hint: Optional[float] = None,
) -> LibraryEntry:
    ensure_dirs()
    sym = str(symbol).upper()
    tf = str(timeframe)
    eid = str(uuid.uuid4())[:8]
    safe_ts = str(created_at).replace(":", "").replace(" ", "_")
    fname = f"{sym}__{tf}__{safe_ts}__{eid}.json"
    payload = {
        "symbol": sym,
        "timeframe": tf,
        "created_at": created_at,
        "source_bundle": source_bundle,
        "gates_passed": bool(gates_passed),
        "score_hint": score_hint,
        "profile": profile_cfg,
    }
    (PROFILES_DIR / fname).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    entry = LibraryEntry(
        id=eid,
        symbol=sym,
        timeframe=tf,
        created_at=str(created_at),
        source_bundle=str(source_bundle),
        gates_passed=bool(gates_passed),
        score_hint=score_hint,
        filename=fname,
    )
    doc = _read_index()
    entries = doc.get("entries") or []
    entries.append(asdict(entry))
    doc["entries"] = entries
    _write_index(doc)
    return entry


def load_profile(entry: LibraryEntry) -> Dict[str, Any]:
    ensure_dirs()
    doc = json.loads((PROFILES_DIR / entry.filename).read_text(encoding="utf-8"))
    return doc.get("profile") or {}


def load_profile_by_id(entry_id: str) -> Optional[Dict[str, Any]]:
    for e in list_entries():
        if e.id == entry_id:
            return load_profile(e)
    return None
