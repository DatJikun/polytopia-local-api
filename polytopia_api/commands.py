"""High-level map commands. Screen pixels; think_turn is the brain."""

from __future__ import annotations

import time
from typing import Any

from . import combat
from . import coords
from . import detect
from . import driver
from . import snapshot
from .observe import all_cities, lookup, lookup_city, lookup_unit, observe, remember

ATTACK_BUDGET_S = 7.5
SELECT_LIGHT_SLEEP_S = 0.18


def _sleep(seconds: float = 0.4) -> None:
    time.sleep(min(seconds, 1.2))


def _frame(obs: dict[str, Any] | None = None) -> tuple[int, int]:
    fr = ((obs or {}).get("layout") or {}).get("frame") or coords.frame()
    return int(fr[0]), int(fr[1])


def _click(x: int, y: int, space: str = "screen", repeats: int = 1) -> dict[str, Any]:
    return driver.click(int(x), int(y), space=space, repeats=repeats)


def _fail(name: str, reason: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "name": name, "reason": reason}
    out.update(extra)
    return out


def _shot_arr(obs: dict[str, Any] | None):
    path = (obs or {}).get("screenshot")
    if not path:
        return None
    try:
        from PIL import Image

        return detect.as_rgb(Image.open(path))
    except Exception:
        return None


def _resolve(ident: str | None, space: str = "screen") -> dict[str, Any] | None:
    """Resolve city_id/unit id from the last observe, then the turn-sticky index.

    A later frame that misses a frost plate must not forget c22_17 mid-turn.
    A later frame that misses an HP bar must not forget u2 after /attack.
    """
    if not ident:
        return None
    ident = str(ident)
    mem = remember()
    if mem:
        hit = lookup(mem, ident)
        if hit:
            return hit
    hit = lookup_city(ident) or lookup_unit(ident)
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
    below = plate + 10
    if all(abs(below - p[1]) >= 6 for p in points):
        points.append((x, below, "plate_below"))
    return points


def _panel_capture(obs: dict[str, Any] | None) -> bool:
    obs = obs or {}
    unit = obs.get("unit") or {}
    ready = obs.get("ready") or {}
    return bool(unit.get("capture") or ready.get("capture"))


def _city_near_xy(
    x: int,
    y: int,
    obs: dict[str, Any] | None,
    max_hex: float = 0.85,
) -> dict[str, Any] | None:
    """Treat a raw x,y as a city when it lands on a known city hex/plate."""
    frame = _frame(obs)
    pitch = combat.hex_pitch(*frame)
    best = None
    best_d = max_hex
    cities = []
    seen: set[str] = set()
    for c in list((obs or {}).get("cities") or []) + all_cities():
        cid = str(c.get("id") or c.get("city_id") or "")
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        cities.append(c)
    for c in cities:
        for px, py in combat.city_stand_points(c, frame) or [(int(c.get("x") or 0), int(c.get("y") or 0))]:
            d = combat.tile_dist(int(x), int(y), px, py, pitch)
            if d <= best_d:
                best_d = d
                best = c
        try:
            plate = int(c.get("plate_y") or c["y"])
            d = combat.tile_dist(int(x), int(y), int(c["x"]), plate, pitch)
        except (KeyError, TypeError, ValueError):
            continue
        if d <= best_d:
            best_d = d
            best = c
    return best


def _on_city_mark(
    marks: list[dict[str, Any]],
    dest: dict[str, Any],
    frame: tuple[int, int],
) -> tuple[dict[str, Any] | None, float | None]:
    """Blue mark on the city hex only — never an adjacent tile (~1.0 hex).

    Prefer the walkable ground (building foot). Scoring every stand point and
    taking the nearest let the roof cluster (~tile_xy) beat the live Disrof
    ring (~0.9 below the roof).
    """
    walk = combat.city_walk_center(dest, frame)
    if walk:
        mark, d = combat.nearest_mark_hex(
            marks, walk[0], walk[1], frame, combat.ON_CITY_MARK_HEX
        )
        if mark is not None:
            return mark, d
    best = None
    best_d = combat.ON_CITY_MARK_HEX
    for px, py in combat.city_stand_points(dest, frame):
        mark, d = combat.nearest_mark_hex(marks, px, py, frame, combat.ON_CITY_MARK_HEX)
        if mark is not None and d is not None and d <= best_d:
            best = mark
            best_d = d
    return best, (None if best is None else round(best_d, 3))


def _find_walk_mark(
    marks: list[dict[str, Any]],
    dest: dict[str, Any],
    frame: tuple[int, int],
    arr=None,
) -> tuple[dict[str, Any] | None, float | None]:
    """Prefer a clustered blue, then leftover ring pixels on the city hex."""
    mark, hex_d = _on_city_mark(marks, dest, frame)
    if mark is not None:
        return mark, hex_d
    if arr is None:
        return None, None
    pitch = combat.hex_pitch(*frame)
    points: list[tuple[int, int]] = []
    walk = combat.city_walk_center(dest, frame)
    if walk:
        points.append(walk)
    for px, py in combat.city_stand_points(dest, frame):
        if (px, py) not in points:
            points.append((px, py))
    for px, py in points:
        hit = detect.move_on_hex(arr, px, py, pitch, min_n=5)
        if hit.get("ok"):
            return hit, 0.0
    for px, py in points:
        tint = detect.move_tint_at(arr, px, py, radius=max(14, pitch // 2), min_n=5)
        if tint.get("ok"):
            return tint, 0.0
    return None, None


def _unit_xy(obs: dict[str, Any] | None, ident: str | None) -> tuple[int, int] | None:
    if not ident:
        return None
    hit = lookup(obs, ident) if obs else None
    if not hit:
        hit = _resolve(ident)
    if not hit:
        return None
    try:
        return int(hit["x"]), int(hit["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _city_click_misfire(
    before: dict[str, Any],
    after: dict[str, Any],
    from_id: str | None,
) -> bool:
    """True when the click selected the city instead of walking the unit."""
    if _panel_capture(after):
        return False
    panel = after.get("unit") or {}
    if panel.get("train") and not panel.get("capture"):
        return True
    xy0 = _unit_xy(before, from_id)
    xy1 = _unit_xy(after, from_id)
    if xy0 and xy1 and xy0 == xy1:
        return True
    return False


def _stood_on_city(
    obs: dict[str, Any],
    dest: dict[str, Any] | None,
    frame: tuple[int, int],
    allow_panel: bool = False,
) -> dict[str, Any]:
    """Pixel ON-tile. Panel Capture only counts after we clicked this city hex."""
    on = combat.unit_on_city(
        obs.get("units") or [],
        dest,
        frame,
        obs.get("unit") or {},
        max_hex=0.55,
        want_owner="own",
    )
    if on.get("ok"):
        return on
    if allow_panel:
        wide = combat.unit_on_city(
            obs.get("units") or [],
            dest,
            frame,
            obs.get("unit") or {},
            max_hex=combat.STANDING_HEX,
            want_owner="own",
        )
        if wide.get("ok"):
            return wide
    if not _panel_capture(obs):
        return on
    # Capture panel + a unit still on THIS hex (sprite offset), not adjacent.
    relaxed = combat.unit_on_city(
        obs.get("units") or [],
        dest,
        frame,
        obs.get("unit") or {},
        max_hex=combat.STANDING_HEX,
        want_owner="own",
    )
    if relaxed.get("ok"):
        relaxed["reason"] = "panel_capture"
        return relaxed
    if not allow_panel:
        return on
    # After a city-hex click, Capture is ground truth even if the HP bar
    # clustering missed the rider on Disrof.
    center = combat.city_tile_center(dest, frame) if dest else None
    return {
        "ok": True,
        "reason": "panel_capture",
        "hex_pitch": combat.hex_pitch(*frame),
        "tile_xy": [center[0], center[1]] if center else None,
        "hex_dist": on.get("hex_dist"),
    }


def select_unit(
    x: int | None = None,
    y: int | None = None,
    id: str | None = None,
    city_id: str | None = None,
    space: str = "screen",
    light: bool = False,
) -> dict[str, Any]:
    """Click a unit (or city/village) so the panel / move marks appear."""
    hit = None
    ident = id or city_id
    if ident:
        hit = _resolve(ident, space)
        if not hit:
            return _fail("select_unit", f"no entity {ident}")
        x, y = _entity_click_xy(hit)
        space = "screen"
    if x is None or y is None:
        return _fail("select_unit", "need x,y or id/city_id")
    clicked = _click(x, y, space=space)
    _sleep(SELECT_LIGHT_SLEEP_S if light else 0.4)
    after = observe(mode="marks" if light else "full")
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
        "attack_marks": after.get("attack_marks"),
        "ready": after.get("ready"),
        "hud": after.get("hud"),
        "observe_mode": after.get("observe_mode"),
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
    try:
        return _move_to(
            x=x, y=y, from_x=from_x, from_y=from_y,
            from_id=from_id, city_id=city_id, to_id=to_id, space=space,
        )
    except Exception as e:
        return _fail(
            "move_to",
            "error",
            error=str(e),
            city_id=city_id,
            from_id=from_id,
            stood_on_city=False,
        )


def _move_to(
    x: int | None = None,
    y: int | None = None,
    from_x: int | None = None,
    from_y: int | None = None,
    from_id: str | None = None,
    city_id: str | None = None,
    to_id: str | None = None,
    space: str = "screen",
) -> dict[str, Any]:
    dest = _resolve(city_id or to_id, space)
    selected = None
    if from_id or from_x is not None:
        selected = select_unit(x=from_x, y=from_y, id=from_id, space=space, light=True)
        if not selected.get("ok"):
            return _fail(
                "move_to",
                str(selected.get("reason") or "select_failed"),
                from_id=from_id,
                city_id=city_id,
                selected=selected,
                stood_on_city=False,
            )
    after_select = remember() or observe(mode="marks")
    frame = _frame(after_select)
    marks = list(after_select.get("move_marks") or [])
    panel = after_select.get("unit") or {}
    if dest is None and x is not None and y is not None:
        dest = _city_near_xy(int(x), int(y), after_select)
        if dest and not city_id:
            city_id = str(dest.get("id") or dest.get("city_id") or "") or None
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
            try:
                tile_xy = (int(dest["x"]), int(dest["y"]))
            except (KeyError, TypeError, ValueError):
                return _fail(
                    "move_to",
                    "need x,y or city_id/to_id",
                    city_id=city_id,
                    stood_on_city=False,
                )
        x, y = tile_xy
        space = "screen"
        already = _stood_on_city(after_select, dest, frame)
        if already.get("ok"):
            return {
                "ok": True,
                "name": "move_to",
                "x": x,
                "y": y,
                "city_id": city_id,
                "to_id": to_id or city_id,
                "dest": dest,
                "selected": selected,
                "stood_on_city": True,
                "on_city": already,
                "reason": "already_on_city",
                "suggested_tile_xy": [x, y],
                "unit": panel,
                "hud": after_select.get("hud"),
                "ready": after_select.get("ready"),
                "turn_diff": after_select.get("turn_diff"),
            }
        src = _resolve(from_id) if from_id else None
        fx = fy = None
        if src:
            fx, fy = int(src["x"]), int(src["y"])
        elif from_x is not None and from_y is not None:
            fx, fy = int(from_x), int(from_y)
        reach = combat.can_reach_tile(
            panel.get("unit"),
            (fx, fy) if fx is not None else None,
            (x, y),
            frame,
        )
        # Adjacent blue hexes (~1.0) used to win nearest_mark_hex(1.15) and
        # walk NEXT TO Disrof while still returning ok:true.
        arr = _shot_arr(after_select)
        mark, hex_d = _find_walk_mark(marks, dest, frame, arr)
        garrison = combat.garrison_unit(after_select.get("units") or [], dest, frame, max_hex=0.9)
        enemy_on = bool(garrison and str(garrison.get("owner") or "") == "enemy")
        atk_marks = list(after_select.get("attack_marks") or [])
        red_on, _ = combat.nearest_mark_hex(atk_marks, x, y, frame, 0.85)
        if red_on is not None:
            enemy_on = True
        step, _ = combat.nearest_mark_toward(
            marks,
            (fx, fy) if fx is not None else None,
            (x, y),
            frame,
            combat.move_range(panel.get("unit")),
        )
        suggested = [int(step["x"]), int(step["y"])] if step else [x, y]
        if mark is None:
            path_reason = "no_mark_on_city_tile"
            hint = "no blue mark on the city tile — unit cannot stand ON this turn"
            reason = "no_mark_on_city_tile"
            if enemy_on:
                path_reason = "city_occupied"
                reason = "city_occupied"
                hint = "attack the garrison first (hp_dropped) then move-to"
            elif not reach.get("ok"):
                path_reason = "out_of_range"
                reason = "out_of_range"
                hint = "city is beyond this unit's walk this turn — step toward suggested_tile_xy"
            elif not marks:
                path_reason = "no_move_marks"
                reason = "no_move_marks"
                hint = "select did not light blue hexes — unit may have no moves"
            return _fail(
                "move_to",
                reason,
                hint=hint,
                path_reason=path_reason,
                city_id=city_id,
                to_id=to_id or city_id,
                city_xy=[x, y],
                suggested_tile_xy=suggested,
                tile=dest.get("tile"),
                tile_dist=reach.get("tile_dist"),
                move_range=reach.get("move_range"),
                move_marks=marks,
                selected=selected,
                garrison=garrison,
                can_reach=reach,
                stood_on_city=False,
            )
        clicked = _click(int(mark["x"]), int(mark["y"]), space="screen")
        _sleep(0.55)
        after = observe(mode="marks")
        on = _stood_on_city(after, dest, frame, allow_panel=True)
        # Building / plate clicks select the city (TRAIN) and never walk.
        # Re-select the unit and click the blue ring (often at the building foot).
        if not on.get("ok") and _city_click_misfire(after_select, after, from_id):
            if from_id or from_x is not None:
                selected = select_unit(
                    x=from_x, y=from_y, id=from_id, space=space, light=True,
                )
                retry_obs = remember() or after
            else:
                retry_obs = after
            retry_frame = _frame(retry_obs)
            retry_marks = list(retry_obs.get("move_marks") or [])
            retry_arr = _shot_arr(retry_obs)
            mark2, hex_d2 = _find_walk_mark(retry_marks, dest, retry_frame, retry_arr)
            aims: list[tuple[int, int]] = []
            if mark2 is not None:
                aims.append((int(mark2["x"]), int(mark2["y"])))
            walk = combat.city_walk_center(dest, retry_frame)
            if walk and walk not in aims:
                aims.append(walk)
            orig = (int(mark["x"]), int(mark["y"]))
            if orig not in aims:
                aims.append(orig)
            first_xy = (int(clicked["x"]), int(clicked["y"]))
            clicked_retry = False
            for ax, ay in aims:
                if clicked_retry and (ax, ay) == first_xy:
                    continue
                clicked = _click(ax, ay, space="screen")
                clicked_retry = True
                _sleep(0.5)
                after = observe(mode="marks")
                on = _stood_on_city(after, dest, retry_frame, allow_panel=True)
                if on.get("ok"):
                    mark = mark2 or {"x": ax, "y": ay, "n": 0, "kind": "city_walk_retry"}
                    hex_d = hex_d2 if mark2 is not None else 0.0
                    break
        stood = bool(on.get("ok"))
        out = {
            "ok": stood,
            "name": "move_to",
            "x": clicked["x"],
            "y": clicked["y"],
            "city_id": city_id,
            "to_id": to_id or city_id,
            "dest": dest,
            "clicked_mark": mark,
            "mark_hex_dist": hex_d,
            "selected": selected,
            "stood_on_city": stood,
            "on_city": on,
            "suggested_tile_xy": [x, y],
            "unit": after.get("unit"),
            "hud": after.get("hud"),
            "ready": after.get("ready"),
            "turn_diff": after.get("turn_diff"),
        }
        if not stood:
            out["reason"] = "not_stood_on_city"
            out["path_reason"] = "not_stood_on_city"
            out["hint"] = "clicked the city hex but unit is still adjacent — Capture needs ON tile"
        return out
    if dest:
        x, y = int(dest["x"]), int(dest["y"])
        space = "screen"
    if x is None or y is None:
        return _fail("move_to", "need x,y or city_id/to_id", city_id=city_id, stood_on_city=False)
    clicked = _click(x, y, space=space)
    _sleep()
    after = observe(mode="marks")
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
    on = _stood_on_city(before, hit, frame)
    panel_capture = _panel_capture(before)
    if not on.get("ok"):
        occupant = combat.unit_on_city(
            before.get("units") or [],
            hit,
            frame,
            before.get("unit") or {},
            max_hex=combat.STANDING_HEX,
            want_owner="own",
        )
        uid = (occupant.get("unit") or {}).get("id") if occupant.get("ok") else None
        if uid:
            select_unit(id=str(uid), light=True)
            before = remember() or before
            frame = _frame(before)
            on = _stood_on_city(before, hit, frame)
            panel_capture = _panel_capture(before)
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
        selected = select_unit(x=from_x, y=from_y, id=from_id, space=space, light=True)
        if not selected.get("ok"):
            return _fail(
                "attack",
                str(selected.get("reason") or "select_failed"),
                from_id=from_id,
                city_id=city_id,
                selected=selected,
                elapsed_s=_elapsed(),
            )
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
        before = observe(mode="marks")
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
    # Red hexes sit on the TILE. Aiming at a tall giant's HP bar (above the
    # hex) used to miss the mark and return no_mark_on_target.
    center = combat.city_tile_center(dest, frame) if dest else None
    if center:
        x, y = center
    elif garrison:
        x, y = int(garrison["x"]), int(garrison["y"])
    elif dest:
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
        src = lookup(before, from_id) or lookup_unit(from_id)
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
    mark, d_hex = None, None
    aims: list[tuple[int, int]] = [(int(x), int(y))]
    if garrison:
        aims.append((int(garrison["x"]), int(garrison["y"])))
    if dest:
        for px, py in combat.city_stand_points(dest, frame):
            aims.append((px, py))
    seen_aim: set[tuple[int, int]] = set()
    best_d = 1.4
    for ax, ay in aims:
        key = (ax, ay)
        if key in seen_aim:
            continue
        seen_aim.add(key)
        hit_m, hit_d = combat.nearest_mark_hex(marks, ax, ay, frame, 1.4)
        if hit_m is not None and hit_d is not None and hit_d <= best_d:
            mark, d_hex = hit_m, hit_d
            best_d = hit_d
            x, y = ax, ay
    if mark is None:
        mark = verdict.get("red_mark")
        d_hex = None
    if mark is None:
        arr = _shot_arr(before)
        if arr is not None:
            for ax, ay in aims:
                tint = detect.attack_tint_at(arr, ax, ay)
                if tint.get("ok"):
                    mark, d_hex = tint, 0.0
                    x, y = ax, ay
                    break
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
            "suggested_tile_xy": [x, y],
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
    after = observe(mode="marks")
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
