"""Playbook step: observe → one legal action → re-observe HUD.

Priority (from polytopia-playbook.md):
1. Close accidental Settings / refuse Clear Forest.
2. Confirm harvest / train if that modal is already open.
3. If a unit can move, click a detected move mark (never water).
4. Harvest fruit if we have ≥2★.
5. End Turn only when spend/moves are exhausted and ★ < 20.
"""

from __future__ import annotations

import time
from typing import Any

from . import coords
from . import driver
from .observe import observe

# Coords we already tried this turn that did not consume a move/stars.
_blacklist: set[tuple[int, int]] = set()
_last_turn: int | None = None


def plan(obs: dict[str, Any]) -> dict[str, Any]:
    """Pure playbook choice. No clicks. Used by step() and tests."""
    stars = obs["hud"].get("stars")
    unit = obs["unit"]
    overlay = obs["overlay"]
    confirm_ready = obs.get("confirm_ready")

    if overlay.get("back_lit") and not confirm_ready and unit.get("settings"):
        return {"name": "back", "reason": "settings/tech overlay"}
    if unit.get("clear_forest"):
        return {"name": "back", "reason": "refuse Clear Forest"}
    if confirm_ready and (unit.get("harvest") or overlay.get("do_it_pixel") or overlay.get("do_it_blobs")):
        blobs = overlay.get("do_it_blobs") or []
        x, y = (blobs[0]["x"], blobs[0]["y"]) if blobs else tuple(coords.DO_IT)
        return {"name": "confirm_harvest", "x": x, "y": y}
    if confirm_ready and (unit.get("train") or overlay.get("train_pixel") or overlay.get("train_blobs")):
        blobs = overlay.get("train_blobs") or []
        x, y = (blobs[0]["x"], blobs[0]["y"]) if blobs else tuple(coords.TRAIN)
        return {"name": "confirm_train", "x": x, "y": y}
    if overlay.get("do_it_pixel"):
        x, y = tuple(coords.DO_IT)
        return {"name": "confirm_do_it", "x": x, "y": y}
    if unit.get("can_move") and obs.get("move_marks"):
        mark = _pick(obs["move_marks"])
        if mark:
            return {"name": "move", "x": mark["x"], "y": mark["y"], "n": mark["n"]}
    if isinstance(stars, int) and stars >= 2 and obs.get("fruit"):
        fruit = _pick(obs["fruit"])
        if fruit:
            return {"name": "click_fruit", "x": fruit["x"], "y": fruit["y"]}
    if unit.get("no_actions"):
        return {"name": "deselect", "reason": "unit has no actions left"}
    if isinstance(stars, int) and stars >= 20:
        return {
            "name": "blocked_end_turn",
            "ok": False,
            "reason": "playbook: do not End Turn with ≥20★ while spend may exist",
            "stars": stars,
        }
    return {"name": "end_turn", "reason": "no move/harvest/confirm left"}


def _round(obs: dict[str, Any]) -> tuple[int | None, int | None]:
    h = obs["hud"]
    return h.get("stars"), h.get("turn")


def _black(x: int, y: int) -> None:
    _blacklist.add((int(x) // 8 * 8, int(y) // 8 * 8))


def _ok_coord(x: int, y: int) -> bool:
    key = (int(x) // 8 * 8, int(y) // 8 * 8)
    return key not in _blacklist


def _pick(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    for it in items:
        if _ok_coord(it["x"], it["y"]):
            return it
    return None


def _sleep() -> None:
    time.sleep(0.45)


def _execute(name: str, planned: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    action = dict(planned)
    if name == "back":
        driver.back()
        _sleep()
        return _result(before, action)
    if name in {"confirm_harvest", "confirm_train", "confirm_do_it"}:
        driver.click(int(action["x"]), int(action["y"]))
        _sleep()
        return _result(before, action)
    if name == "move":
        driver.click(int(action["x"]), int(action["y"]))
        _sleep()
        after = observe()
        if _round(after) == _round(before) and after["unit"].get("can_move"):
            _black(action["x"], action["y"])
            action["ok"] = False
            action["reason"] = "move mark did not consume action (likely water/nameplate)"
        else:
            action["ok"] = True
        return _result(before, action, after=after)
    if name == "click_fruit":
        driver.click(int(action["x"]), int(action["y"]))
        _sleep()
        mid = observe()
        if mid["confirm_ready"] or mid["unit"].get("harvest"):
            action["confirm"] = driver.confirm()
            _sleep()
        elif mid["overlay"].get("do_it_blobs"):
            b = mid["overlay"]["do_it_blobs"][0]
            driver.click(b["x"], b["y"])
            _sleep()
            driver.confirm()
        else:
            _black(action["x"], action["y"])
            action["ok"] = False
            action["reason"] = "fruit click did not open harvest"
        return _result(before, action)
    if name == "deselect":
        driver.click(*coords.DESELECT)
        _sleep()
        return _result(before, action)
    if name == "blocked_end_turn":
        return _result(before, action)
    if name == "end_turn":
        driver.end_turn()
        time.sleep(1.0)
        return _result(before, action)
    action["ok"] = False
    action["reason"] = f"unknown action {name}"
    return _result(before, action)


def step() -> dict[str, Any]:
    """Observe, pick one playbook action, click, re-read HUD."""
    global _last_turn, _blacklist
    if not driver.find_window().get("found"):
        raise driver.PolytopiaError("Polytopia window not found")

    before = observe()
    turn = before["hud"].get("turn")
    if turn != _last_turn:
        _blacklist.clear()
        _last_turn = turn

    planned = plan(before)
    return _execute(planned["name"], planned, before)


def _result(before: dict[str, Any], action: dict[str, Any], after: dict[str, Any] | None = None) -> dict[str, Any]:
    after = after or observe()
    action.setdefault("ok", True)
    action["before"] = before["hud"]
    action["after"] = after["hud"]
    action["unit_before"] = {
        k: before["unit"][k]
        for k in ("unit", "can_move", "no_actions", "harvest", "clear_forest")
    }
    action["move_marks"] = len(before["move_marks"])
    action["fruit_seen"] = len(before["fruit"])
    return action


def play_n(n: int = 1) -> dict[str, Any]:
    steps = []
    for i in range(max(1, n)):
        steps.append(step())
        name = steps[-1].get("name")
        if name == "end_turn":
            break
        if name == "blocked_end_turn":
            break
    return {"ok": True, "count": len(steps), "steps": steps}
