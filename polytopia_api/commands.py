"""High-level map commands. Screen pixels; think_turn is the brain."""

from __future__ import annotations

import time
from typing import Any

from . import coords
from . import driver
from .observe import observe, remember, lookup


def _sleep() -> None:
    time.sleep(0.4)


def _click(x: int, y: int, space: str = "screen", repeats: int = 1) -> dict[str, Any]:
    return driver.click(int(x), int(y), space=space, repeats=repeats)


def select_unit(
    x: int | None = None,
    y: int | None = None,
    id: str | None = None,
    space: str = "screen",
) -> dict[str, Any]:
    """Click a unit (or city/village) so the panel / move marks appear."""
    hit = None
    if id:
        obs = remember() or observe()
        hit = lookup(obs, id)
        if not hit:
            return {"ok": False, "name": "select_unit", "reason": f"no entity {id}"}
        x, y = int(hit["x"]), int(hit["y"])
        space = "screen"
    if x is None or y is None:
        return {"ok": False, "name": "select_unit", "reason": "need x,y or id"}
    clicked = _click(x, y, space=space)
    _sleep()
    after = observe()
    return {
        "ok": True,
        "name": "select_unit",
        "id": id,
        "hit": hit,
        "x": clicked["x"],
        "y": clicked["y"],
        "unit": after.get("unit"),
        "move_marks": after.get("move_marks"),
        "ready": after.get("ready"),
        "hud": after.get("hud"),
    }


def move_to(
    x: int,
    y: int,
    from_x: int | None = None,
    from_y: int | None = None,
    from_id: str | None = None,
    space: str = "screen",
) -> dict[str, Any]:
    """Click destination. Optionally select a unit first — not blob-gated."""
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
        "selected": selected,
        "unit": after.get("unit"),
        "hud": after.get("hud"),
        "ready": after.get("ready"),
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


def capture() -> dict[str, Any]:
    """Press Capture / DO IT if the panel says the tile is ready."""
    before = observe()
    target = capture_target(before)
    if target is None:
        return {
            "ok": False,
            "name": "capture",
            "reason": "capture not ready (no confirm blob / panel)",
            "unit": before.get("unit"),
            "ready": before.get("ready"),
        }
    clicked = _click(target[0], target[1], space="screen")
    _sleep()
    after = observe()
    return {
        "ok": True,
        "name": "capture",
        "x": clicked["x"],
        "y": clicked["y"],
        "hud": after.get("hud"),
        "unit": after.get("unit"),
        "ready": after.get("ready"),
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
    unit: str | None = None,
    space: str = "screen",
) -> dict[str, Any]:
    """Open a city and hit TRAIN. Radial unit portraits stay on-map clicks."""
    opened = None
    if id or x is not None:
        opened = select_unit(x=x, y=y, id=id, space=space)
        if not opened.get("ok"):
            return {**opened, "name": "recruit"}
    obs = remember() or observe()
    target = recruit_target(obs)
    if target is None:
        return {
            "ok": False,
            "name": "recruit",
            "reason": "TRAIN not visible — pass city x,y/id or open the city first",
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
        "x": clicked["x"],
        "y": clicked["y"],
        "want_unit": unit,
        "hint": "city radial portraits are on the map; click one or omit unit",
        "opened": opened,
        "hud": after.get("hud"),
        "unit": after.get("unit"),
        "ready": after.get("ready"),
    }
