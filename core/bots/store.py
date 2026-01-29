from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_PATH = Path("data") / "bots.json"

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def load_bots(path: Path | str = DEFAULT_PATH) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except Exception:
        return []

def save_bots(bots: List[Dict[str, Any]], path: Path | str = DEFAULT_PATH) -> None:
    p = Path(path)
    _ensure_parent(p)
    p.write_text(json.dumps(bots, indent=2, ensure_ascii=False), encoding="utf-8")

def upsert_bot(bots: List[Dict[str, Any]], bot: Dict[str, Any]) -> List[Dict[str, Any]]:
    bot_id = str(bot.get("id", "")).strip()
    if not bot_id:
        return bots
    out = []
    replaced = False
    for b in bots:
        if str(b.get("id","")) == bot_id:
            out.append(bot)
            replaced = True
        else:
            out.append(b)
    if not replaced:
        out.append(bot)
    return out

def delete_bot(bots: List[Dict[str, Any]], bot_id: str) -> List[Dict[str, Any]]:
    return [b for b in bots if str(b.get("id","")) != str(bot_id)]
