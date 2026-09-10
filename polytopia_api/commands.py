"""High-level map commands. Screen pixels; think_turn is the brain."""

from __future__ import annotations

import time
from typing import Any

from . import coords
from . import driver
from . import snapshot
from .observe import observe, remember, lookup


def _sleep(seconds: float = 0.4) -> None:
    time.sleep(seconds)


def _click(x: int, y: int, space: str = "screen", repeats: int = 1) -> dict[str, Any]:
    return driver.click(int(x), int(y), space=space, repeats=repeats)


def _resolve(ident: str | None, space: str = "screen") -> dict[str, Any] | None:
    if not ident:
        return None
    obs = remember() or observe()
    return lookup(obs, str(ident))


def _entity_click_xy(hit: dict[str, Any]) -> tuple[int, int]:
    """Cities: nameplate selects the city. Building often has a unit on the tile."""
    x = int(hit["x"])
    if hit.get("kind") == "city" or hit.get("plate_y") is not None:
        return x, int(hit.get("plate_y") or hit["y"])
    return x, int(hit["y"])


def _city_click_points(hit: dict[str, Any]) -> list[tuple[int, int, str]]:
    x = int(hit["x"])
    plate = int(hit.get("plate_y") or hit["y"])
    building = int(hit["y"])
    points = [(x, plate, "plate")]
    if abs(building - plate) >= 6:
        points.append((x, building, "building"))
    return points


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
        x, y = _entity_click_xy(hit)
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
    """Prefer a live TRAIN blob. Never click scaled TRAIN just because OCR said 'Train'."""
    overlay = obs.get("overlay") or {}
    blobs = overlay.get("train_blobs") or []
    if blobs:
        best = max(blobs, key=lambda b: int(b.get("n") or 0))
        return int(best["x"]), int(best["y"])
    if overlay.get("train_pixel"):
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
    """Select city_id (nameplate first) and click a detected TRAIN blob."""
    ident = city_id or id
    hit = _resolve(ident, space) if ident else None
    if ident and not hit:
        return {"ok": False, "name": "recruit", "reason": f"no entity {ident}", "city_id": ident}
    tried: list[dict[str, Any]] = []
    opened = None
    last_obs = remember()
    points: list[tuple[int, int, str]] = []
    if hit:
        points = _city_click_points(hit)
    elif x is not None and y is not None:
        points = [(int(x), int(y), "xy")]
    target: tuple[int, int] | None = None
    if not points:
        last_obs = last_obs or observe()
        target = recruit_target(last_obs)
        if target is None:
            return {
                "ok": False,
                "name": "recruit",
                "reason": "need city_id (or x,y) — TRAIN not already visible",
                "city_id": ident,
                "unit_panel": last_obs.get("unit"),
                "ready": last_obs.get("ready"),
                "overlay": {
                    "train_pixel": (last_obs.get("overlay") or {}).get("train_pixel"),
                    "train_blobs": (last_obs.get("overlay") or {}).get("train_blobs"),
                    "train_rgb": (last_obs.get("overlay") or {}).get("train_rgb"),
                },
            }
    for cx, cy, where in points:
        opened = {
            **_click(cx, cy, space="screen"),
            "where": where,
            "city_id": ident,
        }
        _sleep(0.7)
        last_obs = observe()
        target = recruit_target(last_obs)
        tried.append({"where": where, "x": cx, "y": cy, "train": target is not None})
        if target is not None:
            break
    if target is None:
        return {
            "ok": False,
            "name": "recruit",
            "reason": "TRAIN not visible after city_id click — calibrate TRAIN or click the city",
            "city_id": ident,
            "hit": hit,
            "tried": tried,
            "opened": opened,
            "unit_panel": (last_obs or {}).get("unit"),
            "ready": (last_obs or {}).get("ready"),
            "overlay": {
                "train_pixel": ((last_obs or {}).get("overlay") or {}).get("train_pixel"),
                "train_blobs": ((last_obs or {}).get("overlay") or {}).get("train_blobs"),
                "train_rgb": ((last_obs or {}).get("overlay") or {}).get("train_rgb"),
            },
        }
    clicked = _click(target[0], target[1], space="screen")
    _sleep(0.5)
    after = observe()
    return {
        "ok": True,
        "name": "recruit",
        "city_id": ident,
        "x": clicked["x"],
        "y": clicked["y"],
        "want_unit": unit,
        "tried": tried,
        "hint": "radial portraits are on the map; click one after TRAIN",
        "opened": opened,
        "hud": after.get("hud"),
        "unit": after.get("unit"),
        "ready": after.get("ready"),
        "turn_diff": after.get("turn_diff"),
    }


def end_turn(turn: int | None = None) -> dict[str, Any]:
    """End Turn, then snapshot JSON+PNG. Stale HUD cannot name the file T{ocr}."""
    before = remember() or observe()
    clicked = driver.end_turn()
    time.sleep(1.0)
    after = observe()
    snap = snapshot.save(after, reason="end_turn", turn=turn)
    d = snap.get("diff") or snapshot.diff(before, after, reason="end_turn")
    alerts = list(d.get("alerts") or [])
    if not snap.get("ok"):
        alerts.append(
            {
                "kind": "stale_hud",
                "ocr": snap.get("ocr_turn"),
                "hint": "pass turn=N on /end-turn — snapshot kept as untrusted-*.json, not T4.json",
            }
        )
    return {
        **clicked,
        "name": "end_turn",
        "hud": after.get("hud"),
        "diff": d,
        "snapshot": snap,
        "alerts": alerts,
        "turn": snap.get("turn"),
        "turn_source": snap.get("turn_source"),
        "turn_trusted": snap.get("turn_trusted"),
    }
