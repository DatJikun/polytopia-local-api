"""Turn-to-turn control layer: structured observe diff + one PNG.

Not a replacement for the API. Saves ``turn_snapshot/T{n}.json`` + ``.png``
on End Turn. Alerts if turn jumps >1 or stars jump by income with no turn change.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

_LAST_SAVED: dict[str, Any] | None = None
_LAST_DIFF: dict[str, Any] | None = None


def snapshot_dir() -> Path:
    d = Path(os.environ.get("POLYTOPIA_SNAPSHOTS", "turn_snapshot"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def slim(obs: dict[str, Any]) -> dict[str, Any]:
    hud = obs.get("hud") or {}

    def _ents(items: list | None) -> list[dict[str, Any]]:
        out = []
        for it in items or []:
            out.append(
                {
                    "id": it.get("id"),
                    "x": it.get("x"),
                    "y": it.get("y"),
                    "tribe": it.get("tribe"),
                    "owner": it.get("owner"),
                }
            )
        return out

    return {
        "hud": {
            "turn": hud.get("turn"),
            "stars": hud.get("stars"),
            "score": hud.get("score"),
            "income": hud.get("income"),
            "stale": hud.get("stale"),
        },
        "units": _ents(obs.get("units")),
        "cities_own": _ents(obs.get("cities_own")),
        "cities_enemy": _ents(obs.get("cities_enemy")),
        "fog_edge": [
            {"id": g.get("id"), "x": g.get("x"), "y": g.get("y")}
            for g in (obs.get("fog_edge") or [])
        ],
        "screenshot": obs.get("screenshot"),
    }


def _ids(items: list[dict[str, Any]]) -> set[str]:
    return {str(it.get("id")) for it in items if it.get("id") is not None}


def diff(prev: dict[str, Any] | None, cur: dict[str, Any], reason: str = "observe") -> dict[str, Any]:
    global _LAST_DIFF
    a = slim(prev) if prev else None
    b = slim(cur)
    alerts: list[dict[str, Any]] = []
    turn0 = (a or {}).get("hud", {}).get("turn") if a else None
    turn1 = b["hud"].get("turn")
    stars0 = (a or {}).get("hud", {}).get("stars") if a else None
    stars1 = b["hud"].get("stars")
    income = b["hud"].get("income")
    turn_delta = None
    if isinstance(turn0, int) and isinstance(turn1, int):
        turn_delta = turn1 - turn0
        if turn_delta > 1:
            alerts.append(
                {
                    "kind": "turn_jump",
                    "from": turn0,
                    "to": turn1,
                    "delta": turn_delta,
                    "hint": "named dock click may have hit End Turn (TECH_TREE)",
                }
            )
    stars_delta = None
    if isinstance(stars0, int) and isinstance(stars1, int):
        stars_delta = stars1 - stars0
        if (
            turn_delta == 0
            and isinstance(income, int)
            and income > 0
            and stars_delta >= income
        ):
            alerts.append(
                {
                    "kind": "stars_income_without_turn",
                    "from": stars0,
                    "to": stars1,
                    "income": income,
                    "hint": "★ jumped by income while turn did not change",
                }
            )

    def _moved(old: list, new: list) -> list[dict[str, Any]]:
        by_id = {str(it.get("id")): it for it in old}
        out = []
        for it in new:
            prev_it = by_id.get(str(it.get("id")))
            if not prev_it:
                continue
            if prev_it.get("x") != it.get("x") or prev_it.get("y") != it.get("y"):
                out.append(
                    {
                        "id": it.get("id"),
                        "from": [prev_it.get("x"), prev_it.get("y")],
                        "to": [it.get("x"), it.get("y")],
                    }
                )
        return out

    own0 = (a or {}).get("cities_own") or []
    own1 = b.get("cities_own") or []
    en0 = (a or {}).get("cities_enemy") or []
    en1 = b.get("cities_enemy") or []
    payload = {
        "reason": reason,
        "from_turn": turn0,
        "to_turn": turn1,
        "turn_delta": turn_delta,
        "stars_delta": stars_delta,
        "alerts": alerts,
        "units": {
            "added": sorted(_ids(b["units"]) - _ids((a or {}).get("units") or [])),
            "removed": sorted(_ids((a or {}).get("units") or []) - _ids(b["units"])),
            "moved": _moved((a or {}).get("units") or [], b["units"]),
        },
        "cities_own": {
            "added": sorted(_ids(own1) - _ids(own0)),
            "removed": sorted(_ids(own0) - _ids(own1)),
        },
        "cities_enemy": {
            "added": sorted(_ids(en1) - _ids(en0)),
            "removed": sorted(_ids(en0) - _ids(en1)),
        },
        "fog_edge": b.get("fog_edge") or [],
        "screenshot": b.get("screenshot"),
    }
    _LAST_DIFF = payload
    return payload


def last_diff() -> dict[str, Any] | None:
    return _LAST_DIFF


def last_saved() -> dict[str, Any] | None:
    global _LAST_SAVED
    if _LAST_SAVED is not None:
        return _LAST_SAVED
    p = snapshot_dir() / "latest.json"
    if p.is_file():
        try:
            _LAST_SAVED = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return _LAST_SAVED


def save(obs: dict[str, Any], reason: str = "end_turn") -> dict[str, Any]:
    """Write JSON + PNG for this turn. Returns paths + diff vs previous snapshot."""
    global _LAST_SAVED
    prev = last_saved()
    d = diff(prev, obs, reason=reason)
    turn = (obs.get("hud") or {}).get("turn")
    tag = f"T{turn}" if turn is not None else "Tunknown"
    folder = snapshot_dir()
    body = slim(obs)
    body["diff"] = d
    json_path = folder / f"{tag}.json"
    json_path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (folder / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    png_path = folder / f"{tag}.png"
    src = obs.get("screenshot")
    if src and Path(src).is_file():
        shutil.copy2(src, png_path)
    _LAST_SAVED = obs
    return {
        "ok": True,
        "json": str(json_path),
        "png": str(png_path) if png_path.is_file() else None,
        "diff": d,
    }
