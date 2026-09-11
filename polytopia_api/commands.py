"""High-level map commands. Screen pixels; think_turn is the brain."""

from __future__ import annotations

import time
from typing import Any

from . import combat
from . import coords
from . import detect
from . import driver
from . import snapshot
from .observe import all_cities, lookup, lookup_city, lookup_unit, observe, register_cities, remember

ATTACK_BUDGET_S = 7.5
RECRUIT_BUDGET_S = 7.5
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


def _merge_city(hit: dict[str, Any] | None, sticky: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep plate/tile/x from the turn index when a later frame is incomplete."""
    if not sticky and not hit:
        return None
    if not sticky:
        out = dict(hit or {})
    elif not hit:
        out = dict(sticky)
    else:
        out = dict(sticky)
        out.update({k: v for k, v in hit.items() if v is not None})
        for key in ("tile", "tile_xy", "plate_y", "x", "y", "kind"):
            if out.get(key) is None and sticky.get(key) is not None:
                out[key] = sticky[key]
    cid = str((sticky or {}).get("id") or (hit or {}).get("id") or (hit or {}).get("city_id") or "")
    if cid:
        out["id"] = cid
        out["city_id"] = cid
    if out.get("kind") is None and (out.get("tile") is not None or out.get("plate_y") is not None or cid.startswith("c")):
        out["kind"] = "city"
    return out


def _resolve(ident: str | None, space: str = "screen") -> dict[str, Any] | None:
    """Resolve city_id/unit id from the last observe, then the turn-sticky index.

    A later frame that misses a frost plate must not forget c22_17 mid-turn.
    A later frame that misses an HP bar must not forget u2 after /attack.
    """
    if not ident:
        return None
    ident = str(ident)
    mem = remember()
    if ident.startswith("c"):
        hit = lookup(mem, ident) if mem else None
        sticky = lookup_city(ident)
        merged = _merge_city(hit if hit and (hit.get("kind") == "city" or hit.get("plate_y") is not None or hit.get("tile") is not None or str(hit.get("id") or "").startswith("c")) else None, sticky)
        if merged:
            register_cities([merged])
            return merged
        fresh = lookup(observe(), ident)
        if fresh:
            register_cities([fresh])
        return fresh
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


def _is_city_hit(hit: dict[str, Any] | None) -> bool:
    if not hit:
        return False
    if hit.get("kind") == "unit":
        return False
    return bool(hit.get("kind") == "city" or hit.get("plate_y") is not None)


def _unit_click_points(hit: dict[str, Any]) -> list[tuple[int, int, str]]:
    """Mountain sprites sit well below the HP bar; one y often misses."""
    x, y = int(hit["x"]), int(hit["y"])
    pts = [(x, y, "body")]
    for dy, name in ((12, "below"), (24, "foot"), (-8, "above")):
        ny = max(0, y + dy)
        if all(abs(ny - p[1]) >= 6 for p in pts):
            pts.append((x, ny, name))
    bar = hit.get("bar_y")
    if bar is not None:
        by = int(bar) + 10
        if all(abs(by - p[1]) >= 6 for p in pts):
            pts.append((x, by, "bar_below"))
    return pts


def _panel_has_unit(panel: dict[str, Any] | None) -> bool:
    panel = panel or {}
    if panel.get("settings"):
        return False
    if panel.get("can_move") or panel.get("no_actions") or panel.get("capture"):
        return True
    if panel.get("unit"):
        return True
    return False


def _dismiss_settings(panel: dict[str, Any] | None) -> None:
    panel = panel or {}
    raw = str(panel.get("raw") or "").lower()
    if panel.get("settings") or "settings" in raw:
        try:
            driver.back()
            _sleep(0.22)
        except Exception:
            pass


def _relive_unit(sticky: dict[str, Any], ident: str | None) -> dict[str, Any] | None:
    """Sticky unseen units click empty terrain / Settings. Prefer a live HP bar."""
    obs = remember()
    if obs:
        live = lookup(obs, ident) if ident else None
        if live and live.get("seen") is not False and not _is_city_hit(live):
            return live
        for u in obs.get("units") or []:
            if ident and str(u.get("id")) == str(ident) and u.get("seen") is not False:
                return u
    fresh = observe(mode="marks")
    if ident:
        live = lookup(fresh, ident)
        if live and live.get("seen") is not False and not _is_city_hit(live):
            return live
    try:
        sx, sy = int(sticky["x"]), int(sticky["y"])
    except (KeyError, TypeError, ValueError):
        return None
    best = None
    best_d = 110.0
    for u in fresh.get("units") or []:
        if u.get("seen") is False or _is_city_hit(u):
            continue
        if str(u.get("owner") or "") == "enemy":
            continue
        try:
            d = ((int(u["x"]) - sx) ** 2 + (int(u["y"]) - sy) ** 2) ** 0.5
        except (KeyError, TypeError, ValueError):
            continue
        if d <= best_d:
            best_d = d
            best = u
    return best


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


def _recruit_click_points(
    hit: dict[str, Any],
    frame: tuple[int, int],
    obs: dict[str, Any] | None = None,
) -> list[tuple[int, int, str]]:
    """Clicks that SELECT THE CITY (TRAIN panel), not a unit on the tile (Disband).

    Roof / nameplate open the city. The standable hex and walk foot select a
    garrison (live 99876c5: tile-first still returned TRAIN not visible).
    Never click the foot — that is Disband. Skip the tile when occupied.
    """
    x = int(hit["x"])
    plate = int(hit.get("plate_y") or hit["y"])
    building = int(hit["y"])
    points: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    def _add(px: int, py: int, where: str) -> None:
        key = (int(px), int(py))
        if coords.in_dock_zone(key[0], key[1]):
            return
        if any(abs(key[0] - sx) < 6 and abs(key[1] - sy) < 6 for sx, sy in seen):
            return
        points.append((key[0], key[1], where))
        seen.add(key)

    tile = combat.city_tile_center(hit, frame)
    if tile:
        _add(tile[0], max(0, tile[1] - 10), "roof")
    _add(x, plate, "plate")
    _add(x, max(0, plate - 10), "plate_above")
    _add(x - 14, plate, "plate_left")
    _add(x, building, "building")
    occupied = False
    if obs is not None:
        garrison = combat.garrison_unit(obs.get("units") or [], hit, frame, max_hex=0.9)
        occupied = garrison is not None
    if tile and not occupied:
        _add(tile[0], tile[1], "tile")
    return points


def _game_stats_open(obs: dict[str, Any] | None) -> bool:
    """Game Stats overlay (hud → Game Mode: Domination). Not the TRAIN panel."""
    hud = (obs or {}).get("hud") or {}
    unit = (obs or {}).get("unit") or {}
    if hud.get("game_stats") or unit.get("game_stats"):
        return True
    raw = f"{hud.get('raw') or ''} {unit.get('raw') or ''}".lower()
    return "game mode" in raw or "game stats" in raw


def _recruit_panel_block(obs: dict[str, Any] | None) -> str | None:
    """Wrong overlay for TRAIN — BACK these. A leftover unit *name* is not a block.

    Live: city panel OCR often reads a nearby warrior; treating that as ``unit``
    dismissed the city and TRAIN never appeared. UNIT_CROP at 1280 also leaks
    the dock "Settings" label while TRAIN is up — do not BACK that.
    """
    if _game_stats_open(obs):
        return "game_stats"
    panel = (obs or {}).get("unit") or {}
    raw = str(panel.get("raw") or "").lower()
    if (
        panel.get("train")
        or "choose a unit" in raw
        or "choose unit" in raw
        or _panel_train_blobs(obs)
    ):
        return None
    if panel.get("disband") or "disband" in raw:
        return "disband"
    if panel.get("capture"):
        return "capture"
    if panel.get("settings") or "settings" in raw:
        # Dock "Settings" OCR leak while the city TRAIN pill (often brand-blue)
        # is already in UNIT_CROP. A real Settings overlay has no panel pill.
        overlay = (obs or {}).get("overlay") or {}
        frame = _frame(obs)
        for b in list(overlay.get("capture_blobs") or []) + list(overlay.get("do_it_blobs") or []):
            try:
                if _in_unit_panel(int(b["x"]), int(b["y"]), frame):
                    return None
            except (KeyError, TypeError, ValueError):
                continue
        return "settings"
    if "tech tree" in raw or "technology" in raw:
        return "tech"
    if panel.get("can_move") or panel.get("no_actions"):
        return "unit"
    return None


def _recruit_hard_block(reason: str | None) -> bool:
    return reason in {"settings", "disband", "tech", "unit", "capture", "game_stats"}


def _dismiss_recruit_block(reason: str | None) -> None:
    if not reason:
        return
    try:
        driver.back()
        _sleep(0.18)
    except Exception:
        pass


def _panel_train_blobs(obs: dict[str, Any] | None) -> list[dict[str, Any]]:
    overlay = (obs or {}).get("overlay") or {}
    frame = _frame(obs)
    out: list[dict[str, Any]] = []
    for b in overlay.get("train_blobs") or []:
        try:
            if _in_unit_panel(int(b["x"]), int(b["y"]), frame):
                out.append(b)
        except (KeyError, TypeError, ValueError):
            continue
    return out


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
            # Centroid of leftover roof-blue sits on the building. Click the
            # hex we searched (foot), not that centroid.
            return {
                "n": hit.get("n"),
                "ok": True,
                "x": px,
                "y": py,
                "kind": "move_hex",
                "centroid": [hit["x"], hit["y"]],
            }, 0.0
    for px, py in points:
        tint = detect.move_tint_at(arr, px, py, radius=max(14, pitch // 2), min_n=5)
        if tint.get("ok"):
            return {"n": tint.get("n"), "ok": True, "x": px, "y": py, "kind": "move_tint"}, 0.0
    return None, None


def _find_attack_mark(
    marks: list[dict[str, Any]],
    dest: dict[str, Any] | None,
    frame: tuple[int, int],
    arr=None,
    extra_xy: list[tuple[int, int]] | None = None,
) -> tuple[dict[str, Any] | None, float | None]:
    """Red hex on the city tile — foot first, not the garrison HP bar.

    Clustered centroids sit on the HP bar / roof; remap them to the walk
    foot before returning so /attack actually lands.
    """
    points: list[tuple[int, int]] = []
    if dest:
        walk = combat.city_walk_center(dest, frame)
        if walk:
            points.append(walk)
        for px, py in combat.city_stand_points(dest, frame):
            if (px, py) not in points:
                points.append((px, py))
    for xy in extra_xy or []:
        if xy not in points:
            points.append(xy)
    best = None
    best_d = 1.15
    for px, py in points:
        mark, d = combat.nearest_mark_hex(marks, px, py, frame, 1.15)
        if mark is not None and d is not None and d <= best_d:
            best = mark
            best_d = d
    if best is not None:
        aimed = combat.city_attack_aim(best, dest, frame) if dest else None
        if aimed and (int(best["x"]), int(best["y"])) != (int(aimed[0]), int(aimed[1])):
            return {
                **best,
                "x": int(aimed[0]),
                "y": int(aimed[1]),
                "aimed": "city_foot",
                "centroid": [int(best["x"]), int(best["y"])],
            }, round(best_d, 3)
        return best, round(best_d, 3)
    if arr is None:
        return None, None
    pitch = combat.hex_pitch(*frame)
    for px, py in points:
        hit = detect.attack_on_hex(arr, px, py, pitch, min_n=5)
        if hit.get("ok"):
            return {
                "n": hit.get("n"),
                "ok": True,
                "x": px,
                "y": py,
                "kind": "attack_hex",
                "aimed": "city_foot",
                "centroid": [hit["x"], hit["y"]],
            }, 0.0
    for px, py in points:
        tint = detect.attack_tint_at(arr, px, py, radius=max(14, pitch // 2), min_n=5)
        if tint.get("ok"):
            return {**tint, "aimed": "city_foot"}, 0.0
    return None, None


def _attack_aim_points(
    dest: dict[str, Any] | None,
    frame: tuple[int, int],
    mark: dict[str, Any] | None,
) -> list[tuple[int, int]]:
    """City-foot pixels for the red hex. HP-bar clicks do not attack."""
    pts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def _add(xy: tuple[int, int] | None) -> None:
        if not xy:
            return
        key = (int(xy[0]), int(xy[1]))
        if key in seen:
            return
        seen.add(key)
        pts.append(key)

    _add(combat.city_attack_aim(mark, dest, frame) if dest else None)
    walk = combat.city_walk_center(dest, frame) if dest else None
    _add(walk)
    if walk:
        _add((walk[0], walk[1] + 6))
        _add((walk[0], walk[1] + 12))
    if mark is not None:
        try:
            mx, my = int(mark["x"]), int(mark["y"])
        except (KeyError, TypeError, ValueError):
            mx = my = None
        if mx is not None and (not walk or my >= walk[1] - 4):
            _add((mx, my))
    return pts


def _hp_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if before.get("id") and not after.get("id"):
        return True
    if (
        before.get("hp") is not None
        and after.get("hp") is not None
        and before.get("hp") != after.get("hp")
    ):
        return True
    if (
        before.get("kind") != "city"
        and isinstance(before.get("n"), int)
        and isinstance(after.get("n"), int)
        and after["n"] < before["n"]
    ):
        return True
    return False


def _confirm_hp_drop(
    before: dict[str, Any],
    dest: dict[str, Any] | None,
    ident: str | None,
    x: int,
    y: int,
    t0: float,
    first_sleep: float = 0.4,
    polls: int = 2,
    interval: float = 0.3,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Wait for the strike animation, then verify garrison HP actually dropped.

    Live T41: /attack clicked a red cluster on the HP bar, slept 0.25s, and
    returned hp_dropped:false while the Vengir garrison was still full.
    """
    hp_before = _hp_snapshot(before, ident, x, y)
    _sleep(first_sleep)
    after = observe(mode="marks")
    if dest:
        register_cities([dest])
    hp_after = _hp_snapshot(after, ident, x, y)
    if _hp_changed(hp_before, hp_after):
        return after, hp_after, True
    still = _enemy_garrison(after, dest, _frame(after)) if dest else None
    if still is None and hp_before.get("id"):
        return after, hp_after, True
    for _ in range(polls):
        if time.time() - t0 >= ATTACK_BUDGET_S:
            break
        _sleep(interval)
        after = observe(mode="marks")
        if dest:
            register_cities([dest])
        hp_after = _hp_snapshot(after, ident, x, y)
        if _hp_changed(hp_before, hp_after):
            return after, hp_after, True
        still = _enemy_garrison(after, dest, _frame(after)) if dest else None
        if still is None and hp_before.get("id"):
            return after, hp_after, True
    return after, hp_after, False


def _relive_select(
    selected: dict[str, Any] | None,
    from_id: str | None,
    from_x: int | None,
    from_y: int | None,
    space: str,
) -> dict[str, Any] | None:
    """Settings/dock after select → live HP bar, same as /move-to."""
    if not selected:
        return selected
    panel = selected.get("unit") or {}
    if (selected.get("ok") and not panel.get("settings")) or not from_id:
        return selected
    sticky = selected.get("hit") or _resolve(from_id)
    live = _relive_unit(sticky, from_id) if sticky else None
    if not live:
        return selected
    return select_unit(
        x=from_x, y=from_y, id=str(live.get("id") or from_id), space=space, light=True,
    )


def _occupied_fields(
    obs: dict[str, Any],
    dest: dict[str, Any] | None,
    frame: tuple[int, int],
    from_id: str | None,
    city_id: str | None,
    garrison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured next step: attack until the garrison is gone, then move-to."""
    garrison = garrison or (
        combat.garrison_unit(obs.get("units") or [], dest, frame, max_hex=0.9)
        if dest else None
    )
    attacker = combat.land_attacker_near(obs.get("units") or [], dest, frame)
    aid = str((attacker or {}).get("id") or from_id or "") or None
    gid = str((garrison or {}).get("id") or "") or None
    cid = city_id or (str(dest.get("id")) if dest else None)
    body: dict[str, Any] = {}
    if aid:
        body["from_id"] = aid
    if cid:
        body["city_id"] = cid
    return {
        "reason": "city_occupied",
        "path_reason": "city_occupied",
        "next": "attack",
        "hint": (
            "POST /attack {from_id, city_id} until hp_dropped and garrison gone, "
            "then POST /move-to → stood_on_city, then POST /capture"
        ),
        "garrison": garrison,
        "garrison_id": gid,
        "attacker_id": aid,
        "next_call": {"method": "POST", "path": "/attack", "body": body},
        "attack_marks": list(obs.get("attack_marks") or []),
        "stood_on_city": False,
        "city_id": cid,
    }


def _enemy_garrison(
    obs: dict[str, Any],
    dest: dict[str, Any] | None,
    frame: tuple[int, int],
) -> dict[str, Any] | None:
    garrison = combat.garrison_unit(obs.get("units") or [], dest, frame, max_hex=0.9)
    if garrison and str(garrison.get("owner") or "") == "enemy":
        return garrison
    return None


def _is_city_dest(dest: dict[str, Any] | None, city_id: str | None, to_id: str | None) -> bool:
    if dest and (
        dest.get("kind") == "city"
        or dest.get("tile") is not None
        or dest.get("plate_y") is not None
    ):
        return True
    for ident in (city_id, to_id):
        s = str(ident or "")
        if s.startswith("c") and "_" in s[1:]:
            return bool(dest)
    return False


def _adjacent_city_foot(
    dest: dict[str, Any],
    frame: tuple[int, int],
    reach: dict[str, Any],
) -> tuple[dict[str, Any] | None, float | None]:
    """When clustered blues are empty but the unit is ~1 hex away, click the foot.

    Live T39: u3→c14_24 tile_dist≈0.94 / move_range=1 returned no_move_marks
    while the city hex was the destination.
    """
    walk = combat.city_walk_center(dest, frame)
    if walk is None:
        return None, None
    dist = reach.get("tile_dist")
    rng = float(reach.get("move_range") or 1)
    if dist is None or float(dist) > rng + 0.65:
        return None, None
    if float(dist) > 1.25:
        return None, None
    return {
        "x": int(walk[0]),
        "y": int(walk[1]),
        "n": 0,
        "ok": True,
        "kind": "city_foot",
    }, 0.0


def _unit_xy(obs: dict[str, Any] | None, ident: str | None) -> tuple[int, int] | None:
    if not ident:
        return None
    if obs:
        for u in obs.get("units") or []:
            if str(u.get("id") or "") != str(ident):
                continue
            if u.get("seen") is False:
                continue
            try:
                return int(u["x"]), int(u["y"])
            except (KeyError, TypeError, ValueError):
                continue
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
    if panel.get("settings"):
        return True
    if panel.get("train") and not panel.get("capture"):
        return True
    xy0 = _unit_xy(before, from_id)
    xy1 = _unit_xy(after, from_id)
    if xy0 and xy1 and xy0 == xy1:
        return True
    return False


def _opened_city_panel(obs: dict[str, Any] | None) -> bool:
    """TRAIN / Settings after a roof click — the unit did not walk."""
    if _panel_capture(obs):
        return False
    panel = (obs or {}).get("unit") or {}
    if panel.get("settings"):
        return True
    return bool(panel.get("train") and not panel.get("capture"))


def _unit_hex_on_city(
    obs: dict[str, Any] | None,
    dest: dict[str, Any] | None,
    frame: tuple[int, int],
    from_id: str | None,
) -> dict[str, Any] | None:
    """Same hex as the city walk/roof — not dest.tile (that numbering can match an adjacent HP bar)."""
    if not dest:
        return None
    walk = combat.city_walk_center(dest, frame) or combat.city_tile_center(dest, frame)
    if walk is None:
        return None
    city_hex = combat.tile_grid(walk[0], walk[1], frame)
    uid = str(from_id or "")
    for u in (obs or {}).get("units") or []:
        if u.get("seen") is False:
            continue
        owner = str(u.get("owner") or "")
        if owner and owner not in {"own", "unknown"}:
            continue
        if uid and str(u.get("id") or "") not in {"", uid}:
            continue
        try:
            ux, uy = int(u["x"]), int(u["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if combat.tile_grid(ux, uy, frame) != city_hex:
            continue
        return u
    return None


def _foot_aim_points(
    dest: dict[str, Any],
    frame: tuple[int, int],
    mark: dict[str, Any] | None,
) -> list[tuple[int, int]]:
    """City-foot pixels. Roof clicks select the city; nudge down onto the ring."""
    pts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def _add(xy: tuple[int, int] | None) -> None:
        if not xy:
            return
        key = (int(xy[0]), int(xy[1]))
        if key in seen:
            return
        seen.add(key)
        pts.append(key)

    walk = combat.city_walk_center(dest, frame)
    if mark is not None:
        _add(combat.city_move_aim(mark, dest, frame))
    _add(walk)
    if walk:
        _add((walk[0], walk[1] + 6))
        _add((walk[0], walk[1] + 12))
    if mark is not None:
        try:
            mx, my = int(mark["x"]), int(mark["y"])
        except (KeyError, TypeError, ValueError):
            mx = my = None
        if mx is not None and walk and my >= walk[1] - 4:
            _add((mx, my))
    return pts


def _confirm_stood_on_city(
    before: dict[str, Any],
    dest: dict[str, Any],
    frame: tuple[int, int],
    from_id: str | None,
    first_sleep: float = 0.45,
    polls: int = 3,
    interval: float = 0.4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Wait for the walk animation, then verify the unit is ON the city hex.

    Live T41: adjacent foot-click returned not_stood_on_city after 0.55s while
    the sprite was still next to Disrof. Do not re-select until coords stabilize
    (re-select cancels an in-progress walk).
    """
    _sleep(first_sleep)
    after = observe(mode="marks")
    if dest:
        register_cities([dest])
    frame = _frame(after) or frame
    on = _stood_on_city(after, dest, frame, allow_panel=True)
    if on.get("ok"):
        return after, on
    if _opened_city_panel(after):
        return after, on
    xy_prev = _unit_xy(after, from_id) or _unit_xy(before, from_id)
    stable = 0
    for _ in range(polls):
        _sleep(interval)
        after = observe(mode="marks")
        if dest:
            register_cities([dest])
        frame = _frame(after) or frame
        on = _stood_on_city(after, dest, frame, allow_panel=True)
        if on.get("ok"):
            return after, on
        if _opened_city_panel(after):
            return after, on
        xy = _unit_xy(after, from_id)
        if xy and xy_prev and xy == xy_prev:
            stable += 1
            if stable >= 2:
                break
        else:
            stable = 0
        xy_prev = xy or xy_prev
    return after, on


def _retry_city_foot(
    dest: dict[str, Any],
    from_id: str | None,
    from_x: int | None,
    from_y: int | None,
    space: str,
    selected: dict[str, Any] | None,
    city_id: str | None,
    to_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, float | None, dict[str, Any], dict[str, Any] | None]:
    """Re-select the walker and click the city foot again (full select so blues light)."""
    if from_id or from_x is not None:
        selected = select_unit(
            x=from_x, y=from_y, id=from_id, space=space, light=False,
        )
        selected = _relive_select(selected, from_id, from_x, from_y, space)
        retry_obs = remember() or observe(mode="marks")
    else:
        retry_obs = observe(mode="marks")
    panel = retry_obs.get("unit") or {}
    if not retry_obs.get("move_marks") or panel.get("settings"):
        _sleep(0.28)
        retry_obs = remember() or observe(mode="marks")
        panel = retry_obs.get("unit") or {}
    dest = _resolve(city_id or to_id, space) or dest
    if dest:
        register_cities([dest])
    retry_frame = _frame(retry_obs)
    retry_marks = list(retry_obs.get("move_marks") or [])
    if panel.get("settings"):
        retry_marks = []
    retry_arr = _shot_arr(retry_obs)
    mark2, hex_d2 = _find_walk_mark(retry_marks, dest, retry_frame, retry_arr)
    src = _resolve(from_id) if from_id else None
    fx = fy = None
    if src:
        fx, fy = int(src["x"]), int(src["y"])
    elif from_x is not None and from_y is not None:
        fx, fy = int(from_x), int(from_y)
    walk = combat.city_walk_center(dest, retry_frame) or combat.city_tile_center(dest, retry_frame)
    reach = combat.can_reach_tile(
        panel.get("unit"),
        (fx, fy) if fx is not None else None,
        walk,
        retry_frame,
    )
    if mark2 is None and not retry_marks and reach.get("ok"):
        mark2, hex_d2 = _adjacent_city_foot(dest, retry_frame, reach)
    aims = _foot_aim_points(dest, retry_frame, mark2)[:3]
    after = retry_obs
    on = _stood_on_city(after, dest, retry_frame, allow_panel=True)
    clicked = {"ok": True, "x": aims[0][0], "y": aims[0][1]} if aims else {"ok": True, "x": 0, "y": 0}
    mark = mark2
    hex_d = hex_d2
    if on.get("ok"):
        return after, on, mark, hex_d, clicked, selected
    for ax, ay in aims:
        clicked = _click(ax, ay, space="screen")
        mark = {
            **(mark2 or {"n": 0, "kind": "city_walk_retry"}),
            "x": ax,
            "y": ay,
            "aimed": "city_foot",
        }
        hex_d = hex_d2 if mark2 is not None else 0.0
        after, on = _confirm_stood_on_city(retry_obs, dest, retry_frame, from_id)
        if on.get("ok"):
            return after, on, mark, hex_d, clicked, selected
        if _opened_city_panel(after) and (from_id or from_x is not None):
            selected = select_unit(
                x=from_x, y=from_y, id=from_id, space=space, light=False,
            )
            retry_obs = remember() or after
    return after, on, mark, hex_d, clicked, selected


def _unit_tagged_on_city(obs: dict[str, Any], dest: dict[str, Any] | None) -> dict[str, Any] | None:
    cid = str((dest or {}).get("id") or (dest or {}).get("city_id") or "")
    if not cid:
        return None
    for u in obs.get("units") or []:
        if u.get("seen") is False:
            continue
        owner = str(u.get("owner") or "")
        if owner and owner not in {"own", "unknown"}:
            continue
        if str(u.get("on_city_id") or "") == cid:
            return u
        if str(u.get("near_city_id") or "") == cid:
            try:
                if float(u.get("near_city_hex") or 9) <= combat.CAPTURE_STANDING_HEX:
                    return u
            except (TypeError, ValueError):
                continue
    return None


def _stood_on_city(
    obs: dict[str, Any],
    dest: dict[str, Any] | None,
    frame: tuple[int, int],
    allow_panel: bool = False,
    max_hex: float | None = None,
) -> dict[str, Any]:
    """Pixel ON-tile. Panel Capture is game-truth for standing ON the city."""
    tight = 0.55 if max_hex is None else max_hex
    on = combat.unit_on_city(
        obs.get("units") or [],
        dest,
        frame,
        obs.get("unit") or {},
        max_hex=tight,
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
            max_hex=combat.STANDING_HEX if max_hex is None else max_hex,
            want_owner="own",
        )
        if wide.get("ok"):
            return wide
        hexed = _unit_hex_on_city(obs, dest, frame, None)
        if hexed is not None:
            center = combat.city_walk_center(dest, frame) or combat.city_tile_center(dest, frame)
            return {
                "ok": True,
                "reason": "tile_match",
                "unit": hexed,
                "hex_pitch": combat.hex_pitch(*frame),
                "tile_xy": [center[0], center[1]] if center else None,
                "hex_dist": 0.0,
            }
    if not _panel_capture(obs):
        return on
    # Capture button only appears ON the city. HP bars sit on the nameplate
    # (~1.0 from walk) so STANDING_HEX missed live Disrof; sticky coords can
    # still be the east port (~1.0) after the sprite stepped onto the hex.
    relaxed = combat.unit_on_city(
        obs.get("units") or [],
        dest,
        frame,
        obs.get("unit") or {},
        max_hex=combat.CAPTURE_STANDING_HEX,
        want_owner="own",
    )
    if relaxed.get("ok"):
        relaxed["reason"] = "panel_capture"
        return relaxed
    tagged = _unit_tagged_on_city(obs, dest)
    if tagged is not None:
        center = combat.city_walk_center(dest, frame) or combat.city_tile_center(dest, frame)
        return {
            "ok": True,
            "reason": "panel_capture",
            "unit": tagged,
            "hex_pitch": combat.hex_pitch(*frame),
            "tile_xy": [center[0], center[1]] if center else None,
            "hex_dist": tagged.get("near_city_hex"),
        }
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


def _prefer_live_hit(hit: dict[str, Any] | None) -> dict[str, Any] | None:
    """Do not click a sticky ghost (seen=false) when a live unit is on screen.

    Live T36: select of u25 hit sticky+settings/dock OCR, can_move false, and
    the mountain warrior never walked onto Disrof.
    """
    if not hit or hit.get("seen") is not False:
        return hit
    if hit.get("kind") == "city" or hit.get("plate_y") is not None:
        return hit
    mem = remember()
    units = list((mem or {}).get("units") or [])
    uid = str(hit.get("id") or "")
    tile = hit.get("tile")
    for u in units:
        if u.get("seen") is False:
            continue
        if uid and str(u.get("id") or "") == uid:
            return u
        if tile is not None and u.get("tile") == tile:
            return u
    try:
        hx, hy = int(hit["x"]), int(hit["y"])
    except (KeyError, TypeError, ValueError):
        return hit
    best = None
    best_d = 96
    for u in units:
        if u.get("seen") is False:
            continue
        if str(u.get("owner") or "") not in {"own", "unknown", ""}:
            continue
        try:
            d = abs(int(u["x"]) - hx) + abs(int(u["y"]) - hy)
        except (KeyError, TypeError, ValueError):
            continue
        if d < best_d:
            best_d = d
            best = u
    return best or hit


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
        hit = _prefer_live_hit(_resolve(ident, space))
        if not hit:
            return _fail("select_unit", f"no entity {ident}")
        if not _is_city_hit(hit) and hit.get("seen") is False:
            live = _relive_unit(hit, ident)
            if live:
                hit = live
        x, y = _entity_click_xy(hit)
        space = "screen"
    if x is None or y is None:
        return _fail("select_unit", "need x,y or id/city_id")
    points: list[tuple[int, int, str]]
    if hit and not _is_city_hit(hit):
        points = _unit_click_points(hit)
    else:
        points = [(int(x), int(y), "given")]
    clicked = None
    after: dict[str, Any] = {}
    where = "given"
    for i, (cx, cy, where) in enumerate(points):
        clicked = _click(cx, cy, space=space)
        _sleep(SELECT_LIGHT_SLEEP_S if light else 0.4)
        after = observe(mode="marks" if light else "full")
        panel = after.get("unit") or {}
        if _panel_has_unit(panel) or after.get("move_marks") or after.get("attack_marks"):
            break
        ghost = bool(hit and hit.get("seen") is False)
        if panel.get("settings"):
            _dismiss_settings(panel)
            continue
        if i == 0 and not ghost:
            break
        _dismiss_settings(panel)
    if clicked is None:
        return _fail("select_unit", "need x,y or id/city_id")
    panel = after.get("unit") or {}
    missed = bool(panel.get("settings")) and not after.get("move_marks")
    out = {
        "ok": not missed,
        "name": "select_unit",
        "id": ident,
        "city_id": city_id or (ident if ident and str(ident).startswith("c") else None),
        "hit": hit,
        "where": where,
        "x": clicked["x"],
        "y": clicked["y"],
        "unit": panel,
        "move_marks": after.get("move_marks"),
        "attack_marks": after.get("attack_marks"),
        "ready": after.get("ready"),
        "hud": after.get("hud"),
        "observe_mode": after.get("observe_mode"),
    }
    if missed:
        out["reason"] = "select_missed"
        out["hint"] = "click hit Settings/dock — unit id may be a sticky ghost"
    return out


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
        selected = _relive_select(selected, from_id, from_x, from_y, space)
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
    if dest is None:
        dest = _resolve(city_id or to_id, space)
    if dest is None and x is not None and y is not None:
        dest = _city_near_xy(int(x), int(y), after_select)
        if dest and not city_id:
            city_id = str(dest.get("id") or dest.get("city_id") or "") or None
    if dest:
        register_cities([dest])
    is_city = _is_city_dest(dest, city_id, to_id)
    if is_city:
        tile_xy = combat.city_tile_center(dest, frame)
        if tile_xy is None:
            try:
                tile_xy = (int(dest["x"]), int(dest["y"]))
            except (KeyError, TypeError, ValueError):
                return _fail(
                    "move_to",
                    f"no entity {city_id or to_id}" if (city_id or to_id) else "need x,y or city_id/to_id",
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
        if panel.get("settings"):
            marks = []
        if (not marks or panel.get("settings")) and from_id:
            # Light select (0.18s) often screenshots before the city-hex ring.
            _sleep(0.28)
            after_select = remember() or observe(mode="marks")
            if panel.get("settings") or not after_select.get("move_marks"):
                sticky = (selected or {}).get("hit") or _resolve(from_id)
                live = _relive_unit(sticky, from_id) if sticky else None
                if live:
                    selected = select_unit(
                        id=str(live.get("id") or from_id), space=space, light=True,
                    )
                    after_select = remember() or after_select
            dest = _resolve(city_id or to_id, space) or dest
            if dest:
                register_cities([dest])
            frame = _frame(after_select)
            marks = list(after_select.get("move_marks") or [])
            panel = after_select.get("unit") or {}
            if panel.get("settings"):
                marks = []
        use_pixels = not panel.get("settings")
        arr = _shot_arr(after_select) if use_pixels else None
        garrison = _enemy_garrison(after_select, dest, frame)
        step, _ = combat.nearest_mark_toward(
            marks,
            (fx, fy) if fx is not None else None,
            (x, y),
            frame,
            combat.move_range(panel.get("unit")),
        )
        suggested = [int(step["x"]), int(step["y"])] if step else [x, y]
        # Cannot stand ON the hex while a defender lives — even a false roof-blue.
        if garrison:
            occ = _occupied_fields(
                after_select, dest, frame, from_id, city_id, garrison=garrison,
            )
            return _fail(
                "move_to",
                occ["reason"],
                hint=occ["hint"],
                path_reason=occ["path_reason"],
                next=occ["next"],
                next_call=occ["next_call"],
                garrison=occ["garrison"],
                garrison_id=occ["garrison_id"],
                attacker_id=occ["attacker_id"],
                attack_marks=occ["attack_marks"],
                city_id=occ["city_id"],
                to_id=to_id or occ["city_id"],
                city_xy=[x, y],
                suggested_tile_xy=suggested,
                tile=dest.get("tile"),
                tile_dist=reach.get("tile_dist"),
                move_range=reach.get("move_range"),
                move_marks=marks,
                selected=selected,
                can_reach=reach,
                stood_on_city=False,
            )
        mark, hex_d = _find_walk_mark(marks, dest, frame, arr)
        aim = combat.city_move_aim(mark, dest, frame)
        if mark is None and (not marks) and reach.get("ok"):
            mark, hex_d = _adjacent_city_foot(dest, frame, reach)
            if mark is not None:
                aim = combat.city_move_aim(mark, dest, frame) or (int(mark["x"]), int(mark["y"]))
        if mark is None:
            path_reason = "no_mark_on_city_tile"
            hint = "no blue mark on the city tile — unit cannot stand ON this turn"
            reason = "no_mark_on_city_tile"
            if not reach.get("ok"):
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
        if aim is None:
            aim = (int(mark["x"]), int(mark["y"]))
        clicked = _click(int(aim[0]), int(aim[1]), space="screen")
        # Keep the mark payload but the click is the foot when the cluster sat on the roof.
        if aim and (int(mark["x"]), int(mark["y"])) != (int(aim[0]), int(aim[1])):
            mark = {**mark, "x": int(aim[0]), "y": int(aim[1]), "aimed": "city_foot"}
        after, on = _confirm_stood_on_city(after_select, dest, frame, from_id)
        # Still adjacent (or TRAIN panel): re-select and click the foot again.
        # Do this whenever wait failed — not only xy-unchanged misfire (T41 jitter).
        stand_retried = False
        if not on.get("ok"):
            stand_retried = True
            after, on, mark2, hex_d2, clicked2, selected2 = _retry_city_foot(
                dest, from_id, from_x, from_y, space, selected, city_id, to_id,
            )
            selected = selected2 or selected
            if clicked2:
                clicked = clicked2
            if mark2 is not None:
                mark = mark2
            if hex_d2 is not None:
                hex_d = hex_d2
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
            "stand_retried": stand_retried,
        }
        if not stood:
            out["reason"] = "not_stood_on_city"
            out["path_reason"] = "not_stood_on_city"
            out["hint"] = (
                "clicked the city foot, waited for the walk, re-selected if needed — "
                "unit is still adjacent; Capture needs ON tile"
            )
        return out
    if dest:
        x, y = int(dest["x"]), int(dest["y"])
        space = "screen"
    if x is None or y is None:
        ident = city_id or to_id
        reason = f"no entity {ident}" if ident else "need x,y or city_id/to_id"
        return _fail("move_to", reason, city_id=city_id, stood_on_city=False)
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
    """Bottom-left UNIT_CROP. GAME_STATS at 1280 sits just right of 0.48*w."""
    w, h = frame
    if coords.in_dock_zone(int(x), int(y)):
        return False
    return int(y) >= int(h * 0.78) and int(x) <= int(w * 0.48)


def _panel_capture_blobs(obs: dict[str, Any] | None) -> list[dict[str, Any]]:
    overlay = (obs or {}).get("overlay") or {}
    frame = _frame(obs)
    out: list[dict[str, Any]] = []
    for b in overlay.get("capture_blobs") or []:
        try:
            if _in_unit_panel(int(b["x"]), int(b["y"]), frame):
                out.append(b)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _panel_capture_ready(obs: dict[str, Any] | None) -> bool:
    """OCR Capture *or* a clustered pill in UNIT_CROP — not a map blue."""
    return _panel_capture(obs) or bool(_panel_capture_blobs(obs))


def _capture_soon(obs: dict[str, Any] | None) -> bool:
    unit = (obs or {}).get("unit") or {}
    if unit.get("capture_soon"):
        return True
    raw = str(unit.get("raw") or "").lower()
    return any(
        s in raw
        for s in (
            "ready to capture next",
            "will be ready to capture",
            "entering village",
            "entering city",
        )
    )


def _occupant_for_capture(
    obs: dict[str, Any],
    city: dict[str, Any] | None,
    frame: tuple[int, int],
    on: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if on and on.get("ok") and (on.get("unit") or {}).get("id"):
        return on.get("unit")
    occupant = combat.unit_on_city(
        obs.get("units") or [],
        city,
        frame,
        obs.get("unit") or {},
        max_hex=combat.STANDING_HEX,
        want_owner="own",
    )
    if occupant.get("ok"):
        return occupant.get("unit")
    if on and on.get("ok"):
        return _unit_tagged_on_city(obs, city)
    return None


def _unit_foot_first(hit: dict[str, Any]) -> list[tuple[int, int, str]]:
    """Prefer the sprite foot so we open Capture, not the city TRAIN roof."""
    rank = {"foot": 0, "below": 1, "bar_below": 2, "body": 3, "above": 4}
    return sorted(_unit_click_points(hit), key=lambda p: rank.get(p[2], 9))


def _open_capture_panel(obs: dict[str, Any], occupant: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    """Click the occupying unit until Capture / capture_soon shows in the panel."""
    if not occupant:
        return obs, False
    last = obs
    opened = False
    for i, (cx, cy, _where) in enumerate(_unit_foot_first(occupant)[:4]):
        _click(cx, cy, space="screen")
        _sleep(0.28 if i else SELECT_LIGHT_SLEEP_S)
        last = observe(mode="marks")
        opened = True
        panel = last.get("unit") or {}
        if panel.get("settings"):
            _dismiss_settings(panel)
            last = remember() or last
            continue
        if _panel_capture_ready(last) or _capture_soon(last):
            return last, True
        if panel.get("train"):
            continue
        if _panel_has_unit(panel):
            return last, True
    return last, opened


def capture_target(obs: dict[str, Any]) -> tuple[int, int] | None:
    """One Capture blob in UNIT_CROP / bottom-left panel. Never a map blue."""
    unit = obs.get("unit") or {}
    ready = obs.get("ready") or {}
    ocr = bool(unit.get("capture") or ready.get("capture"))
    overlay = obs.get("overlay") or {}
    frame = _frame(obs)
    panel = _panel_capture_blobs(obs)
    if panel:
        best = max(panel, key=lambda b: int(b.get("n") or 0))
        return int(best["x"]), int(best["y"])
    if not ocr:
        return None
    # OCR said Capture but the blue pill wasn't clustered — still click the panel.
    for b in overlay.get("do_it_blobs") or []:
        try:
            if _in_unit_panel(int(b["x"]), int(b["y"]), frame):
                return int(b["x"]), int(b["y"])
        except (KeyError, TypeError, ValueError):
            continue
    try:
        x0, y0, x1, y1 = coords.crop_box("UNIT_CROP")
    except Exception:
        return None
    return int(x0 + (x1 - x0) * 0.30), int(y0 + (y1 - y0) * 0.42)


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
    garrison = _enemy_garrison(before, hit, frame)
    if garrison:
        occ = _occupied_fields(before, hit, frame, None, city_id, garrison=garrison)
        return {
            "ok": False,
            "name": "capture",
            "reason": occ["reason"],
            "hint": occ["hint"],
            "next": occ["next"],
            "next_call": occ["next_call"],
            "garrison": occ["garrison"],
            "garrison_id": occ["garrison_id"],
            "attacker_id": occ["attacker_id"],
            "city_id": city_id,
            "unit": before.get("unit"),
            "captured": False,
            "stood_on_city": False,
        }
    # Same ON-tile radius as /move-to confirmation (STANDING_HEX=0.8). Tight
    # 0.55 missed live T46: move-to stood_on_city:true then capture
    # not_standing_on_city because the HP bar sat ~0.7 hex off the walk foot.
    # Do not pass allow_panel — leftover Capture OCR must not count a far unit.
    on = _stood_on_city(before, hit, frame, max_hex=combat.STANDING_HEX)
    panel_opened = False
    panel_capture = _panel_capture_ready(before)
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
            on = _stood_on_city(before, hit, frame, max_hex=combat.STANDING_HEX)
            if not on.get("ok"):
                on = occupant
            panel_capture = _panel_capture_ready(before)
            panel_opened = True
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
    if _capture_soon(before) and not panel_capture:
        return {
            "ok": False,
            "name": "capture",
            "reason": "not_ready_until_next_turn",
            "hint": "unit is on the tile; Capture appears next turn (entering city)",
            "next": "end-turn",
            "city_id": city_id,
            "on_city": on,
            "unit": before.get("unit"),
            "captured": False,
            "stood_on_city": True,
            "panel_opened": panel_opened,
        }
    if not panel_capture:
        occupant = _occupant_for_capture(before, hit, frame, on)
        before, opened = _open_capture_panel(before, occupant)
        panel_opened = panel_opened or opened
        frame = _frame(before)
        refreshed = _stood_on_city(before, hit, frame, max_hex=combat.STANDING_HEX)
        if refreshed.get("ok"):
            on = refreshed
        panel_capture = _panel_capture_ready(before)
    if _capture_soon(before) and not panel_capture:
        return {
            "ok": False,
            "name": "capture",
            "reason": "not_ready_until_next_turn",
            "hint": "unit is on the tile; Capture appears next turn (entering city)",
            "next": "end-turn",
            "city_id": city_id,
            "on_city": on,
            "unit": before.get("unit"),
            "captured": False,
            "stood_on_city": True,
            "panel_opened": panel_opened,
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
            "stood_on_city": True,
            "panel_opened": panel_opened,
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
            "stood_on_city": True,
            "panel_opened": panel_opened,
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
        "panel_opened": panel_opened,
        "stood_on_city": True,
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
    if garrison and str(garrison.get("owner") or "") == "own":
        garrison = None
    if garrison is None and x is not None and y is not None:
        garrison = combat.nearest_unit(
            obs.get("units") or [],
            int(x),
            int(y),
            max(48, combat.hex_pitch(*frame)),
        )
        if garrison and str(garrison.get("owner") or "") == "own":
            garrison = None
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
        selected = _relive_select(selected, from_id, from_x, from_y, space)
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

    def _refresh_aim(obs: dict[str, Any]):
        frame_now = _frame(obs)
        garrison_now = (
            combat.garrison_unit(obs.get("units") or [], dest, frame_now, max_hex=0.9)
            if dest else None
        )
        walk_now = combat.city_walk_center(dest, frame_now) if dest else None
        center_now = combat.city_tile_center(dest, frame_now) if dest else None
        ax = ay = None
        if walk_now:
            ax, ay = walk_now
        elif center_now:
            ax, ay = center_now
        elif garrison_now:
            ax, ay = int(garrison_now["x"]), int(garrison_now["y"])
        elif dest:
            ax, ay = int(dest["x"]), int(dest["y"])
        elif x is not None and y is not None:
            ax, ay = int(x), int(y)
        marks_now = list(
            obs.get("attack_marks")
            or (obs.get("overlay") or {}).get("attack_marks")
            or []
        )
        if not marks_now and selected:
            marks_now = list(selected.get("attack_marks") or [])
        extra_now: list[tuple[int, int]] = []
        if ax is not None:
            extra_now.append((int(ax), int(ay)))
        if garrison_now:
            extra_now.append((int(garrison_now["x"]), int(garrison_now["y"])))
        arr_now = _shot_arr(obs)
        mark_now, dist_now = _find_attack_mark(
            marks_now, dest, frame_now, arr_now, extra_xy=extra_now,
        )
        fx_now = fy_now = None
        if from_id:
            src_now = lookup(obs, from_id) or lookup_unit(from_id)
            if src_now:
                fx_now, fy_now = int(src_now["x"]), int(src_now["y"])
        elif from_x is not None:
            fx_now, fy_now = int(from_x), int(from_y)
        panel_now = obs.get("unit") or panel
        verdict_now = combat.can_strike(
            panel_now.get("unit"),
            (fx_now, fy_now) if fx_now is not None else None,
            (int(ax), int(ay)) if ax is not None else None,
            (dest or {}).get("kind"),
            frame_now,
            marks_now,
        )
        if mark_now is None:
            mark_now = verdict_now.get("red_mark")
            dist_now = None
        occ_now = _occupied_fields(
            obs, dest, frame_now, from_id, city_id or ident, garrison=garrison_now,
        ) if dest else {}
        return (
            frame_now, (ax, ay), marks_now, garrison_now, arr_now,
            mark_now, dist_now, verdict_now, occ_now, panel_now,
        )

    frame, (x, y), marks, garrison, arr, mark, d_hex, verdict, occ, panel = _refresh_aim(before)
    if dest and garrison and str(garrison.get("owner") or "") == "own":
        return {
            "ok": False,
            "name": "attack",
            "reason": "own_garrison",
            "hint": "city_id is labeled enemy but the tile unit is own — re-observe; do not strike Bardur",
            "from_id": from_id,
            "to_id": to_id or city_id,
            "city_id": dest.get("id") if dest else city_id,
            "dest": dest,
            "garrison": garrison,
            "garrison_id": str(garrison.get("id") or "") or None,
            "hp_dropped": False,
            "elapsed_s": _elapsed(),
            "unit": panel,
        }
    if x is None or y is None:
        return {
            "ok": False,
            "name": "attack",
            "reason": "need to_id/city_id or x,y",
            "elapsed_s": elapsed,
        }
    # Light select (0.18s) often screenshots before the garrison hex turns red.
    if mark is None and arr is not None and _elapsed() < ATTACK_BUDGET_S - 2:
        _sleep(0.28)
        before = observe(mode="marks")
        if dest:
            register_cities([dest])
        panel = before.get("unit") or panel
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
                "elapsed_s": _elapsed(),
                "unit": panel,
            }
        frame, (x, y), marks, garrison, arr, mark, d_hex, verdict, occ, panel = _refresh_aim(before)
    hp_before = _hp_snapshot(before, to_id or city_id, x, y)
    if verdict.get("reason") == "naval_no_land":
        land = occ.get("attacker_id")
        hint = verdict.get("hint") or "raft/ship cannot attack a land city — disembark first"
        if land and land != from_id:
            hint = (
                f"raft cannot hit the land garrison — POST /attack "
                f"{{from_id:{land}, city_id:{city_id or ident}}} until hp_dropped"
            )
        return {
            "ok": False,
            "name": "attack",
            "reason": "naval_no_land",
            "hint": hint,
            "next": "attack" if land and land != from_id else None,
            "next_call": occ.get("next_call") if land and land != from_id else None,
            "garrison": garrison,
            "garrison_id": occ.get("garrison_id"),
            "attacker_id": land,
            "from_id": from_id,
            "to_id": to_id or city_id,
            "city_id": dest.get("id") if dest else city_id,
            "can_strike": verdict,
            "elapsed_s": round(time.time() - t0, 3),
            "unit": panel,
        }
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
            "garrison_id": occ.get("garrison_id"),
            "attacker_id": occ.get("attacker_id") or from_id,
            "aim": [x, y],
            "suggested_tile_xy": [x, y],
            "selected": selected,
            "attack_marks": marks,
            "can_strike": verdict,
            "hp_before": hp_before,
            "elapsed_s": round(time.time() - t0, 3),
            "unit": panel,
        }
    aims = _attack_aim_points(dest, frame, mark)[:3]
    if not aims:
        aims = [(int(mark["x"]), int(mark["y"]))]
    clicked = {"ok": True, "x": aims[0][0], "y": aims[0][1]}
    after = before
    hp_after = hp_before
    hp_dropped = False
    strike_retried = False
    for i, (ax, ay) in enumerate(aims):
        if _elapsed() >= ATTACK_BUDGET_S:
            break
        if i > 0:
            strike_retried = True
            if from_id or from_x is not None:
                selected = select_unit(
                    x=from_x, y=from_y, id=from_id, space=space, light=True,
                )
                selected = _relive_select(selected, from_id, from_x, from_y, space)
                before = remember() or before
                panel = before.get("unit") or panel
                if panel.get("no_actions"):
                    break
        clicked = _click(int(ax), int(ay), space="screen")
        mark = {**(mark or {}), "x": int(ax), "y": int(ay), "aimed": "city_foot"}
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
                "garrison_dead": False,
                "reason": "timeout",
                "hint": "clicked red mark but skipped post-observe to stay under 8s",
                "elapsed_s": _elapsed(),
                "garrison": garrison,
                "aim": [x, y],
                "strike_retried": strike_retried,
            }
        after, hp_after, hp_dropped = _confirm_hp_drop(
            before, dest, to_id or city_id, x, y, t0,
        )
        if hp_dropped:
            break
    still = _enemy_garrison(after, dest, _frame(after)) if dest else None
    if still is None and hp_after.get("id") and str(hp_after.get("kind") or "") != "city":
        still = hp_after
    garrison_dead = not bool(still)
    nxt = "move-to" if garrison_dead else "attack"
    cid = dest.get("id") if dest else city_id
    next_call = {"method": "POST", "path": "/move-to", "body": {"from_id": from_id, "city_id": cid}} if garrison_dead else {
        "method": "POST", "path": "/attack", "body": {"from_id": from_id, "city_id": cid},
    }
    return {
        "ok": True,
        "name": "attack",
        "from_id": from_id,
        "to_id": to_id or city_id,
        "city_id": cid,
        "x": clicked["x"],
        "y": clicked["y"],
        "red_marks": marks,
        "clicked_mark": mark,
        "mark_hex_dist": d_hex,
        "garrison": still or garrison,
        "garrison_id": str((still or garrison or {}).get("id") or "") or None,
        "garrison_dead": garrison_dead,
        "next": nxt,
        "next_call": next_call,
        "aim": [x, y],
        "can_strike": verdict,
        "hp_before": hp_before,
        "hp_after": hp_after,
        "hp_dropped": hp_dropped,
        "strike_retried": strike_retried,
        "elapsed_s": round(time.time() - t0, 3),
        "selected": selected,
        "unit": after.get("unit"),
        "hud": after.get("hud"),
        "ready": after.get("ready"),
        "turn_diff": after.get("turn_diff"),
    }


def _safe_panel_xy(x: int, y: int, frame: tuple[int, int]) -> bool:
    """TRAIN / confirm live in UNIT_CROP. Never Tech dock or map TRAIN."""
    try:
        if coords.in_dock_zone(int(x), int(y)):
            return False
    except Exception:
        return False
    return _in_unit_panel(int(x), int(y), frame)


def _unit_crop_train_xy(frame: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """Left pill in UNIT_CROP (TRAIN / confirm). Never dock / map."""
    try:
        x0, y0, x1, y1 = coords.crop_box("UNIT_CROP")
    except Exception:
        return None
    x, y = int(x0 + (x1 - x0) * 0.30), int(y0 + (y1 - y0) * 0.42)
    fr = frame or coords.frame()
    if _safe_panel_xy(x, y, fr):
        return x, y
    return None


def recruit_target(obs: dict[str, Any], city_open: bool = False) -> tuple[int, int] | None:
    """UNIT_CROP TRAIN pill only. Never scaled TRAIN, Tech, or Disband."""
    frame = _frame(obs)
    overlay = obs.get("overlay") or {}
    panel = list(_panel_train_blobs(obs))
    if city_open:
        # Own-city TRAIN is often brand-blue (same cluster as Capture/DO IT).
        for b in list(overlay.get("capture_blobs") or []) + list(overlay.get("do_it_blobs") or []):
            try:
                if _safe_panel_xy(int(b["x"]), int(b["y"]), frame):
                    panel.append(b)
            except (KeyError, TypeError, ValueError):
                continue
    if panel:
        best = max(panel, key=lambda b: int(b.get("n") or 0))
        x, y = int(best["x"]), int(best["y"])
        if _safe_panel_xy(x, y, frame):
            return x, y
    unit = obs.get("unit") or {}
    ready = obs.get("ready") or {}
    raw = str(unit.get("raw") or "").lower()
    city_panel = city_open and any(
        s in raw for s in ("train", "traln", "population", "choose a unit", "choose unit")
    )
    if unit.get("train") or ready.get("train") or city_panel:
        # OCR saw TRAIN / city panel but the pill wasn't clustered — click the
        # panel, not the map.
        return _unit_crop_train_xy(frame)
    return None


def recruit(
    x: int | None = None,
    y: int | None = None,
    id: str | None = None,
    city_id: str | None = None,
    unit: str | None = None,
    space: str = "screen",
) -> dict[str, Any]:
    """Click own city so TRAIN appears, then the UNIT_CROP TRAIN / confirm pill.

    Roof/nameplate select the city. Never Disband, never Tech/dock/GAME_STATS.
    Marks-mode overlay; stop under ~8s like /attack.
    """
    t0 = time.time()

    def _elapsed() -> float:
        return round(time.time() - t0, 3)

    ident = city_id or id
    hit = _resolve(ident, space) if ident else None
    if ident and not hit:
        return {"ok": False, "name": "recruit", "reason": f"no entity {ident}", "city_id": ident}
    tried: list[dict[str, Any]] = []
    opened = None
    last_obs = remember()
    dismissed: list[str] = []
    points: list[tuple[int, int, str]] = []
    if hit:
        points = _recruit_click_points(hit, _frame(last_obs), last_obs)
    elif x is not None and y is not None:
        if not coords.in_dock_zone(int(x), int(y)):
            points = [(int(x), int(y), "xy")]
    target: tuple[int, int] | None = None
    confirmed = False

    def _timeout(reason: str = "timeout", **extra: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "name": "recruit",
            "reason": reason,
            "hint": "budget <8s — marks observe for TRAIN, not looping HUD OCR",
            "city_id": ident,
            "hit": hit,
            "tried": tried,
            "opened": opened,
            "dismissed": dismissed,
            "elapsed_s": _elapsed(),
            **extra,
        }

    def _overlay_bits(obs: dict[str, Any] | None) -> dict[str, Any]:
        overlay = (obs or {}).get("overlay") or {}
        return {
            "train_pixel": overlay.get("train_pixel"),
            "train_blobs": overlay.get("train_blobs"),
            "train_rgb": overlay.get("train_rgb"),
        }

    # A leftover unit/Settings panel hides TRAIN — BACK once, then click the city.
    block0 = _recruit_panel_block(last_obs)
    if block0 and _elapsed() < RECRUIT_BUDGET_S - 1.5:
        _dismiss_recruit_block(block0)
        dismissed.append(block0)
        last_obs = remember()

    if not points:
        if _elapsed() >= RECRUIT_BUDGET_S:
            return _timeout()
        last_obs = last_obs or observe(mode="marks")
        target = recruit_target(last_obs, city_open=True)
        if target is None:
            return {
                "ok": False,
                "name": "recruit",
                "reason": "need city_id (or x,y) — TRAIN not already visible",
                "city_id": ident,
                "elapsed_s": _elapsed(),
                "unit_panel": last_obs.get("unit"),
                "ready": last_obs.get("ready"),
                "overlay": _overlay_bits(last_obs),
            }
    for cx, cy, where in points:
        if _elapsed() >= RECRUIT_BUDGET_S:
            return _timeout()
        opened = {
            **_click(cx, cy, space="screen"),
            "where": where,
            "city_id": ident,
        }
        # City panel fades in; 0.28s on live 99876c5 still missed TRAIN.
        _sleep(0.38)
        if _elapsed() >= RECRUIT_BUDGET_S:
            return _timeout()
        last_obs = observe(mode="marks")
        block = _recruit_panel_block(last_obs)
        city_open = not _recruit_hard_block(block)
        target = recruit_target(last_obs, city_open=city_open)
        # One extra wait after nameplate/roof — TRAIN often appears a beat later.
        if (
            target is None
            and city_open
            and where in {"plate", "plate_above", "plate_left", "roof"}
            and _elapsed() < RECRUIT_BUDGET_S - 1.2
        ):
            _sleep(0.22)
            if _elapsed() < RECRUIT_BUDGET_S:
                last_obs = observe(mode="marks")
                block = _recruit_panel_block(last_obs)
                city_open = not _recruit_hard_block(block)
                target = recruit_target(last_obs, city_open=city_open)
        tried.append({
            "where": where,
            "x": cx,
            "y": cy,
            "train": target is not None,
            "block": block,
        })
        if target is not None:
            break
        if _recruit_hard_block(block) and _elapsed() < RECRUIT_BUDGET_S - 1.0:
            _dismiss_recruit_block(block)
            dismissed.append(block)
            last_obs = remember() or last_obs
    if target is None:
        return {
            "ok": False,
            "name": "recruit",
            "reason": "TRAIN not visible after city_id click — calibrate TRAIN or click the city",
            "city_id": ident,
            "hit": hit,
            "tried": tried,
            "opened": opened,
            "dismissed": dismissed,
            "elapsed_s": _elapsed(),
            "unit_panel": (last_obs or {}).get("unit"),
            "ready": (last_obs or {}).get("ready"),
            "overlay": _overlay_bits(last_obs),
        }
    if not _safe_panel_xy(target[0], target[1], _frame(last_obs)):
        return {
            "ok": False,
            "name": "recruit",
            "reason": "TRAIN click would miss UNIT_CROP (Tech/Disband/map)",
            "city_id": ident,
            "hit": hit,
            "tried": tried,
            "opened": opened,
            "elapsed_s": _elapsed(),
        }
    saw_train = bool(
        ((last_obs or {}).get("unit") or {}).get("train")
        or ((last_obs or {}).get("ready") or {}).get("train")
        or _panel_train_blobs(last_obs)
    )
    clicked = _click(target[0], target[1], space="screen")
    confirmed = True
    _sleep(0.28)
    after = last_obs or remember() or {}
    if _elapsed() < RECRUIT_BUDGET_S - 0.7:
        after = observe(mode="marks")
    if _game_stats_open(after):
        _dismiss_recruit_block("game_stats")
        dismissed.append("game_stats")
        return {
            "ok": False,
            "name": "recruit",
            "reason": "game_stats overlay — TRAIN stays in UNIT_CROP, never GAME_STATS",
            "city_id": ident,
            "hit": hit,
            "tried": tried,
            "opened": opened,
            "dismissed": dismissed,
            "elapsed_s": _elapsed(),
            "hud": after.get("hud"),
            "unit": after.get("unit"),
        }
    # Second pill only if TRAIN was replaced by a confirm (never Tech/Disband/Game Stats).
    panel_after = after.get("unit") or {}
    if (
        saw_train
        and not panel_after.get("train")
        and not _recruit_panel_block(after)
        and _elapsed() < RECRUIT_BUDGET_S - 0.4
    ):
        confirm_at = recruit_target(after, city_open=True)
        if (
            confirm_at
            and _safe_panel_xy(confirm_at[0], confirm_at[1], _frame(after))
            and (abs(confirm_at[0] - clicked["x"]) > 10 or abs(confirm_at[1] - clicked["y"]) > 10)
        ):
            clicked = _click(confirm_at[0], confirm_at[1], space="screen")
            confirmed = True
            _sleep(0.22)
            if _elapsed() < RECRUIT_BUDGET_S - 0.5:
                after = observe(mode="marks")
            if _game_stats_open(after):
                _dismiss_recruit_block("game_stats")
                dismissed.append("game_stats")
                return {
                    "ok": False,
                    "name": "recruit",
                    "reason": "game_stats overlay — TRAIN stays in UNIT_CROP, never GAME_STATS",
                    "city_id": ident,
                    "hit": hit,
                    "tried": tried,
                    "opened": opened,
                    "dismissed": dismissed,
                    "elapsed_s": _elapsed(),
                    "hud": after.get("hud"),
                    "unit": after.get("unit"),
                }
    return {
        "ok": True,
        "name": "recruit",
        "city_id": ident,
        "x": clicked["x"],
        "y": clicked["y"],
        "want_unit": unit,
        "tried": tried,
        "dismissed": dismissed,
        "confirmed": confirmed,
        "elapsed_s": _elapsed(),
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
