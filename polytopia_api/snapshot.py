"""Turn-to-turn control layer: structured observe diff + one PNG.

Not a replacement for the API. Saves ``turn_snapshot/T{n}.json`` + ``.png``
on End Turn. Alerts if turn jumps >1 or stars jump by income with no turn change.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

_LAST_SAVED: dict[str, Any] | None = None
_LAST_DIFF: dict[str, Any] | None = None
_UNTRUSTED_I = 0


def reset() -> None:
    """Tests only."""
    global _LAST_SAVED, _LAST_DIFF, _UNTRUSTED_I
    _LAST_SAVED = None
    _LAST_DIFF = None
    _UNTRUSTED_I = 0


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
            "stale_fields": hud.get("stale_fields") or [],
            "turn_trusted": hud.get("turn_trusted"),
            "turn_source": hud.get("turn_source"),
            "turn_ocr": hud.get("turn_ocr"),
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


def hud_turn_trusted(hud: dict[str, Any] | None) -> bool:
    """Stale/cached turn must not become T{n} (T24 was saved as T4.json)."""
    hud = hud or {}
    if hud.get("turn_trusted") is False:
        return False
    if not isinstance(hud.get("turn"), int):
        return False
    fields = hud.get("stale_fields")
    if "turn" in (fields or []):
        return False
    if hud.get("stale") and not fields:
        return False
    return True


def last_trusted_turn(obs: dict[str, Any] | None = None) -> int | None:
    src = obs if obs is not None else last_saved()
    if not src:
        return None
    if src.get("turn_trusted") is True and isinstance(src.get("turn"), int):
        return int(src["turn"])
    hud = src.get("hud") or {}
    if hud_turn_trusted(hud):
        return int(hud["turn"])
    return None


def resolve_turn(
    obs: dict[str, Any],
    label: int | None = None,
    reason: str = "observe",
) -> dict[str, Any]:
    hud = obs.get("hud") or {}
    ocr = hud.get("turn_ocr") if isinstance(hud.get("turn_ocr"), int) else hud.get("turn")
    if not isinstance(ocr, int):
        ocr = None
    if label is not None:
        return {"turn": int(label), "source": "label", "trusted": True, "ocr": ocr}
    prev = last_trusted_turn()
    if reason == "end_turn" and isinstance(prev, int):
        return {"turn": prev + 1, "source": "clock", "trusted": True, "ocr": ocr}
    if hud_turn_trusted(hud) and isinstance(hud.get("turn"), int):
        if isinstance(prev, int) and hud["turn"] < prev:
            return {
                "turn": None,
                "source": "rejected_ocr",
                "trusted": False,
                "ocr": hud["turn"],
                "reason": "ocr_turn_behind",
            }
        return {"turn": int(hud["turn"]), "source": "ocr", "trusted": True, "ocr": ocr}
    return {
        "turn": None,
        "source": "stale",
        "trusted": False,
        "ocr": ocr,
        "reason": "stale_hud",
        "hint": "pass turn=N (manual label) — refusing to write T{ocr} from stale HUD",
    }


def diff(prev: dict[str, Any] | None, cur: dict[str, Any], reason: str = "observe") -> dict[str, Any]:
    global _LAST_DIFF
    a = slim(prev) if prev else None
    b = slim(cur)
    alerts: list[dict[str, Any]] = []
    turn0 = last_trusted_turn(a) if a else None
    if turn0 is None and a:
        t = (a.get("hud") or {}).get("turn")
        turn0 = t if hud_turn_trusted(a.get("hud")) and isinstance(t, int) else None
    turn1 = last_trusted_turn(b)
    if turn1 is None:
        t = b["hud"].get("turn")
        turn1 = t if hud_turn_trusted(b.get("hud")) and isinstance(t, int) else None
    stars0 = (a or {}).get("hud", {}).get("stars") if a else None
    stars1 = b["hud"].get("stars")
    income = b["hud"].get("income")
    a_ok = hud_turn_trusted((a or {}).get("hud")) if a else False
    b_ok = hud_turn_trusted(b.get("hud"))
    turn_delta = None
    if isinstance(turn0, int) and isinstance(turn1, int) and a_ok and b_ok:
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
    if not b_ok:
        alerts.append(
            {
                "kind": "stale_hud",
                "ocr": b["hud"].get("turn_ocr", b["hud"].get("turn")),
                "hint": "HUD turn is stale/untrusted — pass turn=N or use End Turn clock",
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


def save(obs: dict[str, Any], reason: str = "end_turn", turn: int | None = None) -> dict[str, Any]:
    """Write JSON + PNG. Never names the file from a stale HUD turn."""
    global _LAST_SAVED, _UNTRUSTED_I
    resolved = resolve_turn(obs, label=turn, reason=reason)
    tagged = dict(obs)
    hud = dict(obs.get("hud") or {})
    hud["turn_ocr"] = hud.get("turn_ocr", hud.get("turn"))
    hud["turn_source"] = resolved["source"]
    hud["turn_trusted"] = resolved["trusted"]
    if resolved["trusted"]:
        hud["turn"] = resolved["turn"]
    tagged["hud"] = hud
    tagged["turn"] = resolved["turn"]
    tagged["turn_source"] = resolved["source"]
    tagged["turn_trusted"] = resolved["trusted"]

    prev = last_saved()
    d = diff(prev, tagged, reason=reason)
    d["turn_source"] = resolved["source"]
    d["turn_trusted"] = resolved["trusted"]

    folder = snapshot_dir()
    if resolved["trusted"] and resolved["turn"] is not None:
        tag = f"T{resolved['turn']}"
        ok = True
        reason_txt = None
    else:
        _UNTRUSTED_I += 1
        tag = f"untrusted-{int(time.time())}-{_UNTRUSTED_I}"
        ok = False
        reason_txt = resolved.get("reason") or "stale_hud"
        d.setdefault("alerts", []).append(
            {
                "kind": "stale_hud",
                "ocr": resolved.get("ocr"),
                "hint": resolved.get("hint") or "pass turn=N — will not write T4.json from stale HUD",
            }
        )

    body = slim(tagged)
    body["turn"] = resolved["turn"]
    body["turn_source"] = resolved["source"]
    body["turn_trusted"] = resolved["trusted"]
    body["diff"] = d
    json_path = folder / f"{tag}.json"
    json_path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    png_path = folder / f"{tag}.png"
    src = obs.get("screenshot")
    if src and Path(src).is_file():
        shutil.copy2(src, png_path)
    if resolved["trusted"]:
        (folder / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
        _LAST_SAVED = tagged
    else:
        tagged["untrusted"] = True
    return {
        "ok": ok,
        "json": str(json_path),
        "png": str(png_path) if png_path.is_file() else None,
        "tag": tag,
        "turn": resolved["turn"],
        "turn_source": resolved["source"],
        "turn_trusted": resolved["trusted"],
        "reason": reason_txt,
        "hint": resolved.get("hint"),
        "ocr_turn": resolved.get("ocr"),
        "diff": d,
    }
