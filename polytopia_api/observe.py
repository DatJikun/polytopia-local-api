"""One screenshot → HUD + panel + map entities + overlay."""

from __future__ import annotations

from typing import Any

from PIL import Image

from . import combat
from . import coords
from . import detect
from . import driver
from . import entities
from . import snapshot

_LAST: dict[str, Any] | None = None
# city_id → last known city for this turn. Attack/move/capture must still
# resolve c22_17 after a later observe misses the frost plate.
_CITY_INDEX: dict[str, dict[str, Any]] = {}
# unit id → last known unit for this turn. /attack timeout must not forget u2.
_UNIT_INDEX: dict[str, dict[str, Any]] = {}


def remember() -> dict[str, Any] | None:
    return _LAST


def reset() -> None:
    global _LAST, _CITY_INDEX, _UNIT_INDEX
    _LAST = None
    _CITY_INDEX = {}
    _UNIT_INDEX = {}


def all_cities() -> list[dict[str, Any]]:
    """Unique sticky cities for this turn (id → last record)."""
    seen: dict[str, dict[str, Any]] = {}
    for c in _CITY_INDEX.values():
        cid = str(c.get("id") or c.get("city_id") or "")
        if cid and cid not in seen:
            seen[cid] = c
    return list(seen.values())


def lookup_city(ident: str) -> dict[str, Any] | None:
    ident = str(ident)
    hit = _CITY_INDEX.get(ident)
    if hit:
        return hit
    low = ident.lower()
    for c in _CITY_INDEX.values():
        name = c.get("name")
        if name and str(name).lower() == low:
            return c
        if str(c.get("id_alias") or "") == ident:
            return c
        if str(c.get("city_id") or "") == ident:
            return c
    return None


def register_cities(cities: list[dict[str, Any]] | None) -> None:
    for c in cities or []:
        cid = str(c.get("id") or c.get("city_id") or "")
        if not cid:
            continue
        stored = dict(c)
        stored["id"] = cid
        stored["city_id"] = cid
        _CITY_INDEX[cid] = stored
        alias = stored.get("id_alias")
        if alias:
            _CITY_INDEX[str(alias)] = stored


def lookup_unit(ident: str) -> dict[str, Any] | None:
    ident = str(ident)
    hit = _UNIT_INDEX.get(ident)
    if hit:
        return hit
    for u in _UNIT_INDEX.values():
        if str(u.get("id_alias") or "") == ident:
            return u
    return None


def register_units(units: list[dict[str, Any]] | None) -> None:
    for u in units or []:
        uid = str(u.get("id") or "")
        if not uid:
            continue
        stored = dict(u)
        _UNIT_INDEX[uid] = stored
        alias = stored.get("id_alias")
        if alias:
            _UNIT_INDEX[str(alias)] = stored


def lookup(obs: dict[str, Any] | None, ident: str) -> dict[str, Any] | None:
    ident = str(ident)
    low = ident.lower()
    # city_id must not match a unit / leftover mark. Second /move-to on
    # c14_24 live-returned need x,y when lookup wandered off cities.
    if ident.startswith("c") or _is_grid_city_id(ident):
        if obs:
            for key in ("cities", "cities_own", "cities_enemy"):
                for it in obs.get(key) or []:
                    if str(it.get("id")) == ident:
                        return it
                    if str(it.get("city_id") or "") == ident:
                        return it
                    if str(it.get("id_alias") or "") == ident:
                        return it
                    name = it.get("name")
                    if name and str(name).lower() == low:
                        return it
        return lookup_city(ident)
    keys = ("units", "cities", "cities_own", "cities_enemy", "villages", "move_marks", "fruit", "attack_marks")
    if ident.startswith("u"):
        keys = ("units",) + keys
    if obs:
        for key in keys:
            for it in obs.get(key) or []:
                if str(it.get("id")) == ident:
                    return it
                name = it.get("name")
                if name and str(name).lower() == low:
                    return it
                if ident.startswith("u") and str(it.get("id_alias") or "") == ident:
                    return it
    if ident.startswith("u"):
        return lookup_unit(ident)
    if ident[:1].isalpha():
        return lookup_city(ident) or lookup_unit(ident)
    return None


def _ids(prefix: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, it in enumerate(items):
        d = dict(it)
        d.setdefault("id", f"{prefix}{i}")
        out.append(d)
    return out


def _tile_key(d: dict[str, Any]) -> tuple[int, int] | None:
    t = d.get("tile")
    if isinstance(t, (list, tuple)) and len(t) >= 2:
        try:
            return int(t[0]), int(t[1])
        except (TypeError, ValueError):
            return None
    return None


def _is_grid_city_id(ident: Any) -> bool:
    s = str(ident or "")
    return len(s) > 2 and s[0] == "c" and "_" in s[1:]


def _stamp_city_id(c: dict[str, Any]) -> dict[str, Any]:
    cid = str(c.get("id") or c.get("city_id") or "")
    if cid:
        c["id"] = cid
        c["city_id"] = cid
    return c


def _sticky_city_fields(d: dict[str, Any], src: dict[str, Any]) -> None:
    if src.get("name") and not d.get("name"):
        d["name"] = src["name"]
    src_tribe = str(src.get("tribe") or "")
    dst_tribe = str(d.get("tribe") or "")
    # Keep frost-plate Disrof as enemy. A later grey-stone sample must not
    # invent cities_own from the same city_id — including when a friendly
    # unit stands on the port/mountain next to it.
    if src_tribe == "vengir" and str(src.get("owner") or "") == "enemy":
        if dst_tribe in {"oumaji", "bardur", "unknown", ""} or str(d.get("owner") or "") == "own":
            d["tribe"] = "vengir"
            d["owner"] = "enemy"
        ev = list(src.get("evidence") or [])
        if ev:
            cur = list(d.get("evidence") or [])
            for item in ev:
                if item not in cur:
                    cur.append(item)
            d["evidence"] = cur
        if src.get("tile") and not d.get("tile"):
            d["tile"] = list(src["tile"])
        if src.get("tile_xy") and not d.get("tile_xy"):
            d["tile_xy"] = list(src["tile_xy"])
        if src.get("plate_y") is not None and d.get("plate_y") is None:
            d["plate_y"] = src["plate_y"]
        if src.get("kind") and not d.get("kind"):
            d["kind"] = src["kind"]
    elif src_tribe == "vengir" and dst_tribe == "oumaji":
        d["tribe"] = "vengir"
        d["owner"] = "own" if str(src.get("owner") or "") == "own" else "enemy"
    elif str(src.get("owner") or "") == "own":
        # Live T46: after /recruit click, c7_16 flipped cities_own → cities_enemy.
        # A magenta speck / selection highlight must not steal Bardur mid-turn.
        d["owner"] = "own"
        if src_tribe:
            d["tribe"] = src_tribe
        ev = list(src.get("evidence") or [])
        if ev:
            cur = list(d.get("evidence") or [])
            for item in ev:
                if item not in cur:
                    cur.append(item)
            d["evidence"] = cur
    cid = str(d.get("id") or src.get("id") or "")
    if cid:
        d["id"] = cid
        d["city_id"] = cid


def stabilize(
    items: list[dict[str, Any]],
    prev: list[dict[str, Any]] | None,
    max_dist: int = 56,
    keep_missing: bool = False,
) -> list[dict[str, Any]]:
    """Keep ids stable. Cities match by tile (grid hash), not list index.

    ``keep_missing`` carries unmatched previous cities (seen=False) so a
    mid-turn observe that misses a frost plate does not forget ``c22_17``.
    """
    if not items:
        if keep_missing and prev:
            out = []
            for p in prev:
                if not p.get("id"):
                    continue
                d = dict(p)
                d["stable"] = True
                d["seen"] = False
                d["sticky"] = True
                out.append(d)
            return out
        return []
    used: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        d = dict(it)
        d["seen"] = True
        tile = _tile_key(d)
        tile_hit = None
        if tile is not None:
            for p in prev or []:
                pid = str(p.get("id") or "")
                if not pid or pid in used:
                    continue
                if _tile_key(p) == tile:
                    tile_hit = p
                    break
        if tile_hit and tile_hit.get("id"):
            if not d.get("id"):
                d["id"] = tile_hit["id"]
            _sticky_city_fields(d, tile_hit)
            d["stable"] = True
            used.add(str(d["id"]))
            used.add(str(tile_hit["id"]))
            out.append(d)
            continue
        best = None
        best_d = max_dist
        for p in prev or []:
            pid = str(p.get("id") or "")
            if not pid or pid in used:
                continue
            try:
                dist = (int(d["x"]) - int(p["x"])) ** 2 + (int(d["y"]) - int(p["y"])) ** 2
            except (KeyError, TypeError, ValueError):
                continue
            if dist < best_d * best_d:
                best_d = int(dist ** 0.5)
                best = p
        if best and best.get("id"):
            # Same hex already matched above. A neighbor at <56px with a
            # different owner is a different city — do not steal Bardur ids.
            same_tile = tile is not None and _tile_key(best) == tile
            if not same_tile:
                own_a, own_b = str(d.get("owner") or ""), str(best.get("owner") or "")
                tribe_a, tribe_b = str(d.get("tribe") or ""), str(best.get("tribe") or "")
                if (own_a and own_b and own_a != own_b) or {tribe_a, tribe_b} == {
                    "bardur",
                    "vengir",
                }:
                    best = None
        if best and best.get("id"):
            prev_id = str(best["id"])
            new_id = str(d.get("id") or "")
            # Grid-hash jitter (c22_17 vs c22_18) must keep the first id.
            if _is_grid_city_id(new_id) and _is_grid_city_id(prev_id) and new_id != prev_id:
                d["id_alias"] = new_id
                d["id"] = prev_id
                pt = _tile_key(best)
                if pt:
                    d["tile"] = [pt[0], pt[1]]
            elif not _is_grid_city_id(d.get("id")):
                d["id"] = best["id"]
            _sticky_city_fields(d, best)
            d["stable"] = True
            used.add(str(d["id"]))
            used.add(prev_id)
            if new_id:
                used.add(new_id)
        else:
            named = None
            want = str(d.get("name") or "").lower()
            if want:
                for p in prev or []:
                    pid = str(p.get("id") or "")
                    if not pid or pid in used:
                        continue
                    if str(p.get("name") or "").lower() == want:
                        named = p
                        break
            if named:
                if not d.get("id"):
                    d["id"] = named["id"]
                _sticky_city_fields(d, named)
                d["stable"] = True
                used.add(str(d["id"]))
                used.add(str(named["id"]))
            else:
                d["stable"] = False
        out.append(d)
    if keep_missing:
        for p in prev or []:
            pid = str(p.get("id") or "")
            if not pid or pid in used:
                continue
            d = dict(p)
            d["stable"] = True
            d["seen"] = False
            d["sticky"] = True
            used.add(pid)
            out.append(d)
    return out


def _next_unit_id(used: set[str]) -> str:
    i = 0
    while f"u{i}" in used:
        i += 1
    return f"u{i}"


def stabilize_units(
    items: list[dict[str, Any]],
    prev: list[dict[str, Any]] | None,
    max_dist: int = 56,
    keep_missing: bool = True,
) -> list[dict[str, Any]]:
    """Keep u2/u5/u11 for the turn even when an HP bar flickers after /attack.

    Match by tile first so a warrior that steps one hex (mountain → Disrof)
    keeps the same id instead of minting u12 while u5 stays stuck in the south.
    """
    used: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items or []:
        d = dict(it)
        d["seen"] = True
        tile = _tile_key(d)
        tile_hit = None
        if tile is not None:
            for p in prev or []:
                pid = str(p.get("id") or "")
                if not pid or pid in used:
                    continue
                if _tile_key(p) == tile:
                    tile_hit = p
                    break
        best = tile_hit
        best_d = 0.0 if tile_hit else max_dist
        if best is None:
            for p in prev or []:
                pid = str(p.get("id") or "")
                if not pid or pid in used:
                    continue
                try:
                    dist = ((int(d["x"]) - int(p["x"])) ** 2 + (int(d["y"]) - int(p["y"])) ** 2) ** 0.5
                except (KeyError, TypeError, ValueError):
                    continue
                if dist <= best_d:
                    best_d = dist
                    best = p
        if best and best.get("id"):
            d["id"] = best["id"]
            d["stable"] = True
            used.add(str(d["id"]))
        else:
            d["id"] = _next_unit_id(used | {str(p.get("id") or "") for p in (prev or [])})
            d["stable"] = False
            used.add(str(d["id"]))
        out.append(d)
    if keep_missing:
        for p in prev or []:
            pid = str(p.get("id") or "")
            if not pid or pid in used:
                continue
            d = dict(p)
            d["stable"] = True
            d["seen"] = False
            d["sticky"] = True
            used.add(pid)
            out.append(d)
    return out


def ready_flags(unit: dict[str, Any], overlay: dict[str, Any], confirm_ready: bool) -> dict[str, Any]:
    capture = bool(unit.get("capture"))
    train = bool(unit.get("train") or overlay.get("train_pixel"))
    harvest = bool(unit.get("harvest") or (confirm_ready and overlay.get("do_it_pixel")))
    return {
        "capture": capture,
        "train": train,
        "harvest": harvest,
        "move": bool(unit.get("can_move")),
        "attack": bool(overlay.get("attack_marks")),
        "confirm": bool(confirm_ready),
    }


def observe(shot: Any | None = None, mode: str = "full") -> dict[str, Any]:
    """mode=full: HUD + cities. mode=marks: panel + units + hexes (fast /attack)."""
    global _LAST
    path = shot or driver.screenshot()
    im = Image.open(path)
    coords.set_frame(*im.size)
    prev = _LAST
    light = mode == "marks" and prev is not None
    if light:
        hud = dict(prev.get("hud") or {})
    else:
        hud = driver.read_hud(im)
        snapshot.apply_turn_floor(hud)
    unit = driver.parse_unit_panel(
        driver.ocr_crop(im, coords.UNIT_CROP, driver.UNIT_PATH)
    )
    pix = detect.observe_pixels(im)
    overlay = pix["overlay"]
    arr = detect.as_rgb(im)
    if light:
        mapped_cities = list(prev.get("cities") or [])
        mapped_units = entities.attach_units_near_cities(
            arr, entities.find_units(arr), mapped_cities
        )
        villages = list(prev.get("villages") or [])
        fog = list(prev.get("fog_edge") or [])
        own_tribe = prev.get("own_tribe") or entities._own_tribe()
        villages_enabled = bool(prev.get("villages_enabled", False))
    else:
        mapped = entities.observe_map(arr)
        mapped_units = mapped["units"]
        mapped_cities = mapped["cities"]
        villages = mapped["villages"]
        fog = mapped.get("fog_edge") or []
        own_tribe = mapped["own_tribe"]
        villages_enabled = mapped.get("villages_enabled", False)
    prev_turn = ((prev or {}).get("hud") or {}).get("turn")
    new_turn = (hud or {}).get("turn")
    # Mid-turn HUD OCR flicker (T39→3) must not wipe sticky c14_24.
    # Only a trusted *increase* is a new turn.
    if (
        not light
        and prev_turn is not None
        and new_turn is not None
        and int(new_turn) > int(prev_turn)
        and snapshot.hud_turn_trusted(hud)
    ):
        _CITY_INDEX.clear()
        _UNIT_INDEX.clear()
        prev = None
    units = stabilize_units(mapped_units, (prev or {}).get("units"), keep_missing=True)
    register_units(units)
    # City ids are grid hashes (c{gx}_{gy}); never reindex as c0/c1.
    # Keep unmatched previous cities so attack/move/capture still resolve.
    # Recap unnamed Vengir so frost FPs do not grow 2→4→7 across the session.
    cities = [
        _stamp_city_id(c)
        for c in entities.session_cities(
            stabilize(
                list(mapped_cities),
                (prev or {}).get("cities"),
                keep_missing=True,
            )
        )
    ]
    register_cities(cities)
    own = [c for c in cities if c.get("owner") == "own"]
    enemy = [c for c in cities if c.get("owner") == "enemy"]
    attack_marks = _ids("a", pix.get("attack_marks") or overlay.get("attack_marks") or [])
    panel_type = unit.get("unit")
    strike = combat.unit_profile(panel_type)
    confirm_ready = bool(
        overlay["do_it_pixel"]
        or overlay["train_pixel"]
        or unit.get("capture")
        or unit.get("train")
        or unit.get("harvest")
        or (overlay.get("train_blobs") and unit.get("train"))
        or (overlay.get("capture_blobs") and unit.get("capture"))
        or (overlay.get("do_it_blobs") and unit.get("harvest"))
    )
    if light and prev.get("window"):
        info = prev["window"]
        win = {k: info.get(k) for k in ("found", "pid", "window_id", "width", "height", "match", "name")}
    else:
        info = driver.find_window()
        win = {
            k: info.get(k)
            for k in ("found", "pid", "window_id", "width", "height", "match", "name")
        }
    payload = {
        "hud": hud,
        "unit": unit,
        "units": units,
        "cities": cities,
        "cities_own": own,
        "cities_enemy": enemy,
        "villages": villages,
        "villages_enabled": villages_enabled,
        "fog_edge": fog,
        "own_tribe": own_tribe,
        "move_marks": _ids("m", pix["move_marks"]),
        "attack_marks": attack_marks,
        "fruit": _ids("f", pix["fruit"]),
        "selected_unit": strike,
        "overlay": overlay,
        "confirm_ready": confirm_ready,
        "ready": ready_flags(unit, overlay, confirm_ready),
        "hits_space": "screen",
        "window": win,
        "layout": coords.layout_info(),
        "screenshot": str(path),
        "observe_mode": "marks" if light else "full",
    }
    payload["observe_diff"] = snapshot.diff(prev, payload, reason="frame") if prev else None
    payload["turn_diff"] = snapshot.diff(snapshot.last_saved(), payload, reason="observe")
    _LAST = payload
    return payload
