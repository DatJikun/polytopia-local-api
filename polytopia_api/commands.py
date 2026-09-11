"""High-level map commands. Screen pixels; think_turn is the brain."""

from __future__ import annotations

import time
from typing import Any

from . import combat
from . import coords
from . import driver
from . import snapshot
from .observe import lookup, lookup_city, observe, remember

ATTACK_BUDGET_S = 7.5


def _sleep(seconds: float = 0.4) -> None:
    time.sleep(min(seconds, 1.2))


def _frame(obs: dict[str, Any] | None = None) -> tuple[int, int]:
    fr = ((obs or {}).get("layout") or {}).get("frame") or coords.frame()
    return int(fr[0]), int(fr[1])


def _click(x: int, y: int, space: str = "screen", repeats: int = 1) -> dict[str, Any]:
    return driver.click(int(x), int(y), space=space, repeats=repeats)


def _resolve(ident: str | None, space: str = "screen") -> dict[str, Any] | None:
    """Resolve city_id/unit id from the last observe, then the turn-sticky index.

    A later frame that misses a frost plate must not forget c22_17 mid-turn.
    """
    if not ident:
        return None
    ident = str(ident)
    mem = remember()
    if mem:
        hit = lookup(mem, ident)
        if hit:
            return hit
    hit = lookup_city(ident)
    if hit:
        return hit
    return lookup(observe(), ident)


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
    """Click a blue move mark on the city tile — never the nameplate."""
    dest = _resolve(city_id or to_id, space)
    selected = None
    if from_id or from_x is not None:
        selected = select_unit(x=from_x, y=from_y, id=from_id, space=space)
        if not selected.get("ok"):
            return {**selected, "name": "move_to"}
    after_select = remember() or observe()
    frame = _frame(after_select)
    marks = after_select.get("move_marks") or []
    tile_xy = None
    is_city = bool(
        dest
        and (
            dest.get("kind") == "city"
            or dest.get("tile") is not None
            or dest.get("plate_y") is not None
        )
    )
    if is_city:
        tile_xy = combat.city_tile_center(dest, frame)
        if tile_xy is None:
            return {"ok": False, "name": "move_to", "reason": "need x,y or city_id/to_id"}
        x, y = tile_xy
        space = "screen"
        mark, hex_d = combat.nearest_mark_hex(marks, x, y, frame, 0.6)
        if mark is None:
            return {
                "ok": False,
                "name": "move_to",
                "reason": "no_mark_on_city_tile",
                "hint": "no blue mark within 0.6 hex of the city tile — unit cannot stand ON this turn",
                "city_id": city_id,
                "to_id": to_id or city_id,
                "city_xy": [x, y],
                "tile": dest.get("tile"),
                "move_marks": marks,
                "selected": selected,
                "stood_on_city": False,
            }
        clicked = _click(int(mark["x"]), int(mark["y"]), space="screen")
        _sleep(0.45)
        after = observe()
        on = combat.unit_on_city(after.get("units") or [], dest, frame, after.get("unit") or {}, max_hex=0.55)
        return {
            "ok": True,
            "name": "move_to",
            "x": clicked["x"],
            "y": clicked["y"],
            "city_id": city_id,
            "to_id": to_id or city_id,
            "dest": dest,
            "clicked_mark": mark,
            "mark_hex_dist": hex_d,
            "selected": selected,
            "stood_on_city": bool(on.get("ok")),
            "on_city": on,
            "unit": after.get("unit"),
            "hud": after.get("hud"),
            "ready": after.get("ready"),
            "turn_diff": after.get("turn_diff"),
        }
    if dest:
        x, y = int(dest["x"]), int(dest["y"])
        space = "screen"
    if x is None or y is None:
        return {"ok": False, "name": "move_to", "reason": "need x,y or city_id/to_id"}
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
        "stood_on_city": False,
        "unit": after.get("unit"),
        "hud": after.get("hud"),
        "ready": after.get("ready"),
        "turn_diff": after.get("turn_diff"),
    }


def _in_unit_panel(x: int, y: int, frame: tuple[int, int]) -> bool:
    w, h = frame
    return int(y) >= int(h * 0.78) and int(x) <= int(w * 0.52)


def capture_target(obs: dict[str, Any]) -> tuple[int, int] | None:
    """One Capture blob in UNIT_CROP / bottom-left panel. Never a map blue."""
    unit = obs.get("unit") or {}
    ready = obs.get("ready") or {}
    if not unit.get("capture") and not ready.get("capture"):
        return None
    overlay = obs.get("overlay") or {}
    frame = _frame(obs)
    blobs = overlay.get("capture_blobs") or []
    panel = [b for b in blobs if _in_unit_panel(int(b["x"]), int(b["y"]), frame)]
    if panel:
        best = max(panel, key=lambda b: int(b.get("n") or 0))
        return int(best["x"]), int(best["y"])
    return None


def _captured(before: dict[str, Any], after: dict[str, Any], city_id: str | None, dest: dict[str, Any] | None) -> bool:
    ident = str(city_id or (dest or {}).get("id") or "")
    if not ident:
        return False
    en0 = {str(c.get("id")) for c in (before.get("cities_enemy") or [])}
    en1 = {str(c.get("id")) for c in (after.get("cities_enemy") or [])}
    if ident in en0 and ident not in en1:
        return True
    hit = lookup(after, ident)
    if hit and hit.get("owner") == "own":
        return True
    tile = (dest or {}).get("tile")
    if tile:
        for c in after.get("cities_own") or []:
            if c.get("tile") == tile:
                return True
    return False


def capture(city_id: str | None = None, space: str = "screen") -> dict[str, Any]:
    """Capture only when a unit already stands ON the city tile. No map click."""
    hit = _resolve(city_id, space) if city_id else None
    if city_id and not hit:
        return {"ok": False, "name": "capture", "reason": f"no entity {city_id}", "city_id": city_id}
    before = remember() or observe()
    frame = _frame(before)
    on = combat.unit_on_city(
        before.get("units") or [],
        hit,
        frame,
        before.get("unit") or {},
        max_hex=0.4,
    )
    panel_capture = bool((before.get("unit") or {}).get("capture") or (before.get("ready") or {}).get("capture"))
    if not on.get("ok"):
        return {
            "ok": False,
            "name": "capture",
            "reason": "not_standing_on_city",
            "hint": "POST /move-to with city_id until stood_on_city; do not computerUse Capture",
            "city_id": city_id,
            "on_city": on,
            "unit": before.get("unit"),
            "captured": False,
        }
    if not panel_capture:
        return {
            "ok": False,
            "name": "capture",
            "reason": "capture not ready (no panel Capture blob)",
            "hint": "unit is on the tile but panel does not say Capture yet",
            "city_id": city_id,
            "on_city": on,
            "unit": before.get("unit"),
            "captured": False,
        }
    target = capture_target(before)
    if target is None:
        return {
            "ok": False,
            "name": "capture",
            "reason": "capture not ready (no panel Capture blob)",
            "city_id": city_id,
            "on_city": on,
            "unit": before.get("unit"),
            "ready": before.get("ready"),
            "overlay": {
                "capture_blobs": (before.get("overlay") or {}).get("capture_blobs"),
            },
            "captured": False,
        }
    clicked = _click(target[0], target[1], space="screen")
    _sleep(0.55)
    after = observe()
    captured = _captured(before, after, city_id, hit)
    return {
        "ok": True,
        "name": "capture",
        "city_id": city_id,
        "x": clicked["x"],
        "y": clicked["y"],
        "on_city": on,
        "captured": captured,
        "hud": after.get("hud"),
        "unit": after.get("unit"),
        "ready": after.get("ready"),
        "turn_diff": after.get("turn_diff"),
    }


def _hp_snapshot(obs: dict[str, Any], ident: str | None, x: int | None, y: int) -> dict[str, Any]:
    """Garrison HP bar on the city tile — never city nameplate ``n`` / ``w``."""
    frame = _frame(obs)
    hit = lookup(obs, ident) if ident else None
    city = None
    if hit and (
        hit.get("kind") == "city"
        or hit.get("plate_y") is not None
        or hit.get("tile") is not None
    ):
        city = hit
    garrison = combat.garrison_unit(obs.get("units") or [], city, frame) if city else None
    if garrison is None and x is not None and y is not None:
        garrison = combat.nearest_unit(
            obs.get("units") or [],
            int(x),
            int(y),
            max(48, combat.hex_pitch(*frame)),
        )
    if not garrison or garrison.get("kind") == "city":
        return {}
    return {
        "id": garrison.get("id"),
        "kind": garrison.get("kind") or "unit",
        "hp": garrison.get("hp"),
        "n": garrison.get("n"),
        "x": garrison.get("x"),
        "y": garrison.get("y"),
    }


def attack(
    from_id: str | None = None,
    to_id: str | None = None,
    from_x: int | None = None,
    from_y: int | None = None,
    x: int | None = None,
    y: int | None = None,
    city_id: str | None = None,
    space: str = "screen",
) -> dict[str, Any]:
    """Red mark nearest garrison unit xy. HP from that bar, not the city plate."""
    t0 = time.time()

    def _elapsed() -> float:
        return round(time.time() - t0, 3)

    ident = to_id or city_id
    dest = _resolve(ident, space) if ident else None
    if dest:
        space = "screen"
        dest = dict(dest)
    if _elapsed() >= ATTACK_BUDGET_S:
        return {
            "ok": False,
            "name": "attack",
            "reason": "timeout",
            "hint": "budget <8s — not looping observe",
            "from_id": from_id,
            "to_id": ident,
            "dest": dest,
            "elapsed_s": _elapsed(),
        }
    selected = None
    if from_id or from_x is not None:
        selected = select_unit(x=from_x, y=from_y, id=from_id, space=space)
        if not selected.get("ok"):
            return {**selected, "name": "attack", "elapsed_s": _elapsed()}
    if _elapsed() >= ATTACK_BUDGET_S:
        return {
            "ok": False,
            "name": "attack",
            "reason": "timeout",
            "hint": "select/observe blew the 8s budget — not a second observe",
            "from_id": from_id,
            "to_id": ident,
            "dest": dest,
            "selected": selected,
            "elapsed_s": _elapsed(),
        }
    before = remember() or (selected or {}).get("state")
    if before is None:
        before = observe()
    elapsed = _elapsed()
    panel = before.get("unit") or {}
    if panel.get("no_actions"):
        return {
            "ok": False,
            "name": "attack",
            "reason": "no_actions",
            "hint": "selected unit has no actions — do not reuse stale red marks",
            "from_id": from_id,
            "to_id": to_id or city_id,
            "dest": dest,
            "selected": selected,
            "attack_marks": [],
            "elapsed_s": elapsed,
            "unit": panel,
        }
    frame = _frame(before)
    garrison = combat.garrison_unit(before.get("units") or [], dest, frame) if dest else None
    if garrison:
        x, y = int(garrison["x"]), int(garrison["y"])
    elif dest:
        center = combat.city_tile_center(dest, frame)
        if center:
            x, y = center
        else:
            x, y = int(dest["x"]), int(dest["y"])
    if x is None or y is None:
        return {
            "ok": False,
            "name": "attack",
            "reason": "need to_id/city_id or x,y",
            "elapsed_s": elapsed,
        }
    marks = list(before.get("attack_marks") or (before.get("overlay") or {}).get("attack_marks") or [])
    fx = fy = None
    if from_id:
        src = lookup(before, from_id)
        if src:
            fx, fy = int(src["x"]), int(src["y"])
    elif from_x is not None:
        fx, fy = int(from_x), int(from_y)
    verdict = combat.can_strike(
        panel.get("unit"),
        (fx, fy) if fx is not None else None,
        (x, y),
        (dest or {}).get("kind"),
        frame,
        marks,
    )
    hp_before = _hp_snapshot(before, to_id or city_id, x, y)
    mark, d_hex = combat.nearest_mark_hex(marks, x, y, frame, 1.15)
    if mark is None:
        mark = verdict.get("red_mark")
        d_hex = None
    if mark is None:
        return {
            "ok": False,
            "name": "attack",
            "reason": verdict.get("reason") or "no_mark_on_target",
            "hint": verdict.get("hint") or "no red hex on the garrison tile",
            "from_id": from_id,
            "to_id": to_id or city_id,
            "dest": dest,
            "garrison": garrison,
            "aim": [x, y],
            "selected": selected,
            "attack_marks": marks,
            "can_strike": verdict,
            "hp_before": hp_before,
            "elapsed_s": round(time.time() - t0, 3),
            "unit": panel,
        }
    if verdict.get("reason") == "naval_no_land":
        return {
            "ok": False,
            "name": "attack",
            "reason": "naval_no_land",
            "hint": verdict.get("hint"),
            "from_id": from_id,
            "to_id": to_id or city_id,
            "can_strike": verdict,
            "elapsed_s": round(time.time() - t0, 3),
            "unit": panel,
        }
    clicked = _click(int(mark["x"]), int(mark["y"]), space="screen")
    if _elapsed() >= ATTACK_BUDGET_S:
        return {
            "ok": True,
            "name": "attack",
            "from_id": from_id,
            "to_id": ident,
            "city_id": dest.get("id") if dest else city_id,
            "x": clicked["x"],
            "y": clicked["y"],
            "clicked_mark": mark,
            "hp_before": hp_before,
            "hp_dropped": False,
            "reason": "timeout",
            "hint": "clicked red mark but skipped post-observe to stay under 8s",
            "elapsed_s": _elapsed(),
            "garrison": garrison,
            "aim": [x, y],
        }
    _sleep(0.25)
    after = observe()
    hp_after = _hp_snapshot(after, to_id or city_id, x, y)
    hp_dropped = False
    if hp_before.get("id") and not hp_after.get("id"):
        hp_dropped = True
    elif hp_before.get("hp") is not None and hp_after.get("hp") is not None and hp_before.get("hp") != hp_after.get("hp"):
        hp_dropped = True
    elif (
        hp_before.get("kind") != "city"
        and isinstance(hp_before.get("n"), int)
        and isinstance(hp_after.get("n"), int)
        and hp_after["n"] < hp_before["n"]
    ):
        hp_dropped = True
    return {
        "ok": True,
        "name": "attack",
        "from_id": from_id,
        "to_id": to_id or city_id,
        "city_id": dest.get("id") if dest else city_id,
        "x": clicked["x"],
        "y": clicked["y"],
        "red_marks": marks,
        "clicked_mark": mark,
        "mark_hex_dist": d_hex,
        "garrison": garrison,
        "aim": [x, y],
        "can_strike": verdict,
        "hp_before": hp_before,
        "hp_after": hp_after,
        "hp_dropped": hp_dropped,
        "elapsed_s": round(time.time() - t0, 3),
        "selected": selected,
        "unit": after.get("unit"),
        "hud": after.get("hud"),
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
