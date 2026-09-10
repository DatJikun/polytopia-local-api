"""High-level map commands. Screen pixels; think_turn is the brain."""

from __future__ import annotations

import time
from typing import Any

from . import coords
from . import driver
from . import snapshot
from .observe import observe, remember, lookup


def _sleep() -> None:
    time.sleep(0.4)


def _click(x: int, y: int, space: str = "screen", repeats: int = 1) -> dict[str, Any]:
    return driver.click(int(x), int(y), space=space, repeats=repeats)


def _resolve(ident: str | None, space: str = "screen") -> dict[str, Any] | None:
    if not ident:
        return None
    obs = remember() or observe()
    return lookup(obs, str(ident))


def select_unit(
    x: int | None = None,
    y: int | None = None,
    id: str | None = None,
    city_id: str | None = None,
    space: str = "screen",
) -> dict[str, Any]:
    """Click a unit (or city/village) so the panel / move marks appear."""
    hit = None
    ident = id or city_id
    if ident:
        obs = remember() or observe()
        hit = lookup(obs, ident)
        if not hit:
            return {"ok": False, "name": "select_unit", "reason": f"no entity {ident}"}
        x, y = int(hit["x"]), int(hit["y"])
        space = "screen"
    if x is None or y is None:
        return {"ok": False, "name": "select_unit", "reason": "need x,y or id/city_id"}
    clicked = _click(x, y, space=space)
    _sleep()
    after = observe()
    return {
        "ok": True,
        "name": "select_unit",
        "id": ident,
        "city_id": city_id or (ident if ident and str(ident).startswith("c") else None),
        "hit": hit,
        "x": clicked["x"],
        "y": clicked["y"],
        "unit": after.get("unit"),
        "move_marks": after.get("move_marks"),
        "ready": after.get("ready"),
        "hud": after.get("hud"),
    }


def move_to(
    x: int | None = None,
    y: int | None = None,
    from_x: int | None = None,
    from_y: int | None = None,
    from_id: str | None = None,
    city_id: str | None = None,
    to_id: str | None = None,
    space: str = "screen",
) -> dict[str, Any]:
    """Click destination. ``city_id`` / ``to_id`` resolve an observe entity."""
    dest = _resolve(city_id or to_id, space)
    if dest:
        x, y = int(dest["x"]), int(dest["y"])
        space = "screen"
    if x is None or y is None:
        return {"ok": False, "name": "move_to", "reason": "need x,y or city_id/to_id"}
    selected = None
    if from_id or from_x is not None:
        selected = select_unit(x=from_x, y=from_y, id=from_id, space=space)
        if not selected.get("ok"):
            return {**selected, "name": "move_to"}
    clicked = _click(x, y, space=space)
    _sleep()
    after = observe()
    return {
        "ok": True,
        "name": "move_to",
        "x": clicked["x"],
        "y": clicked["y"],
        "city_id": city_id,
        "to_id": to_id or city_id,
        "dest": dest,
        "selected": selected,
        "unit": after.get("unit"),
        "hud": after.get("hud"),
        "ready": after.get("ready"),
        "turn_diff": after.get("turn_diff"),
    }


def capture_target(obs: dict[str, Any]) -> tuple[int, int] | None:
    blobs = (obs.get("overlay") or {}).get("capture_blobs") or (obs.get("overlay") or {}).get("do_it_blobs") or []
    if blobs:
        return int(blobs[0]["x"]), int(blobs[0]["y"])
    if obs.get("ready", {}).get("capture") or (obs.get("unit") or {}).get("capture"):
        return tuple(coords.DO_IT)
    overlay = obs.get("overlay") or {}
    if overlay.get("do_it_pixel") or overlay.get("capture_pixel"):
        return tuple(coords.DO_IT)
    return None


def capture(city_id: str | None = None, space: str = "screen") -> dict[str, Any]:
    """Press Capture / DO IT. Pass city_id to stand-select the city first."""
    opened = None
    if city_id:
        opened = select_unit(id=city_id, city_id=city_id, space=space)
        if not opened.get("ok"):
            return {**opened, "name": "capture"}
    before = remember() or observe()
    target = capture_target(before)
    if target is None:
        return {
            "ok": False,
            "name": "capture",
            "reason": "capture not ready (no confirm blob / panel)",
            "city_id": city_id,
            "opened": opened,
            "unit": before.get("unit"),
            "ready": before.get("ready"),
        }
    clicked = _click(target[0], target[1], space="screen")
    _sleep()
    after = observe()
    return {
        "ok": True,
        "name": "capture",
        "city_id": city_id,
        "x": clicked["x"],
        "y": clicked["y"],
        "opened": opened,
        "hud": after.get("hud"),
        "unit": after.get("unit"),
        "ready": after.get("ready"),
        "turn_diff": after.get("turn_diff"),
    }


def recruit_target(obs: dict[str, Any]) -> tuple[int, int] | None:
    overlay = obs.get("overlay") or {}
    blobs = overlay.get("train_blobs") or []
    if blobs:
        return int(blobs[0]["x"]), int(blobs[0]["y"])
    if overlay.get("train_pixel") or (obs.get("unit") or {}).get("train") or obs.get("ready", {}).get("train"):
        return tuple(coords.TRAIN)
    return None


def recruit(
    x: int | None = None,
    y: int | None = None,
    id: str | None = None,
    city_id: str | None = None,
    unit: str | None = None,
    space: str = "screen",
) -> dict[str, Any]:
    """Open a city (city_id) and hit TRAIN. Radial portraits stay on-map clicks."""
    opened = None
    ident = city_id or id
    if ident or x is not None:
        opened = select_unit(x=x, y=y, id=ident, city_id=city_id, space=space)
        if not opened.get("ok"):
            return {**opened, "name": "recruit"}
    obs = remember() or observe()
    target = recruit_target(obs)
    if target is None:
        return {
            "ok": False,
            "name": "recruit",
            "reason": "TRAIN not visible — pass city_id or open the city first",
            "city_id": city_id or ident,
            "opened": opened,
            "unit_panel": obs.get("unit"),
            "ready": obs.get("ready"),
        }
    clicked = _click(target[0], target[1], space="screen")
    _sleep()
    after = observe()
    return {
        "ok": True,
        "name": "recruit",
        "city_id": city_id or ident,
        "x": clicked["x"],
        "y": clicked["y"],
        "want_unit": unit,
        "hint": "city radial portraits are on the map; click one or omit unit",
        "opened": opened,
        "hud": after.get("hud"),
        "unit": after.get("unit"),
        "ready": after.get("ready"),
        "turn_diff": after.get("turn_diff"),
    }


def end_turn() -> dict[str, Any]:
    """End Turn, then snapshot JSON+PNG and return structured diff."""
    before = remember() or observe()
    clicked = driver.end_turn()
    time.sleep(1.0)
    after = observe()
    snap = snapshot.save(after, reason="end_turn")
    d = snapshot.diff(before, after, reason="end_turn")
    return {
        **clicked,
        "name": "end_turn",
        "hud": after.get("hud"),
        "diff": d,
        "snapshot": {"json": snap.get("json"), "png": snap.get("png")},
        "alerts": d.get("alerts") or [],
    }
