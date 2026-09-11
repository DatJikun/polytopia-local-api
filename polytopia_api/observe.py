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


def remember() -> dict[str, Any] | None:
    return _LAST


def reset() -> None:
    global _LAST, _CITY_INDEX
    _LAST = None
    _CITY_INDEX = {}


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
    return None


def register_cities(cities: list[dict[str, Any]] | None) -> None:
    for c in cities or []:
        cid = str(c.get("id") or "")
        if not cid:
            continue
        stored = dict(c)
        _CITY_INDEX[cid] = stored
        alias = stored.get("id_alias")
        if alias:
            _CITY_INDEX[str(alias)] = stored


def lookup(obs: dict[str, Any] | None, ident: str) -> dict[str, Any] | None:
    ident = str(ident)
    keys = ("units", "cities", "cities_own", "cities_enemy", "villages", "move_marks", "fruit", "attack_marks")
    if ident.startswith("c"):
        keys = ("cities", "cities_own", "cities_enemy") + keys
    low = ident.lower()
    if obs:
        for key in keys:
            for it in obs.get(key) or []:
                if str(it.get("id")) == ident:
                    return it
                name = it.get("name")
                if name and str(name).lower() == low:
                    return it
                if ident.startswith("c") and str(it.get("id_alias") or "") == ident:
                    return it
    if ident.startswith("c") or ident[:1].isalpha():
        return lookup_city(ident)
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


def _sticky_city_fields(d: dict[str, Any], src: dict[str, Any]) -> None:
    if src.get("name") and not d.get("name"):
        d["name"] = src["name"]
    if str(src.get("tribe") or "") == "vengir" and str(d.get("tribe") or "") == "oumaji":
        d["tribe"] = "vengir"
        d["owner"] = "own" if str(src.get("owner") or "") == "own" else "enemy"


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


def observe(shot: Any | None = None) -> dict[str, Any]:
    global _LAST
    path = shot or driver.screenshot()
    im = Image.open(path)
    coords.set_frame(*im.size)
    hud = driver.read_hud(im)
    snapshot.apply_turn_floor(hud)
    unit = driver.parse_unit_panel(
        driver.ocr_crop(im, coords.UNIT_CROP, driver.UNIT_PATH)
    )
    pix = detect.observe_pixels(im)
    overlay = pix["overlay"]
    arr = detect.as_rgb(im)
    mapped = entities.observe_map(arr)
    prev = _LAST
    prev_turn = ((prev or {}).get("hud") or {}).get("turn")
    new_turn = (hud or {}).get("turn")
    if prev_turn is not None and new_turn is not None and prev_turn != new_turn:
        _CITY_INDEX.clear()
        prev = None
    units = stabilize(_ids("u", mapped["units"]), (prev or {}).get("units"))
    # City ids are grid hashes (c{gx}_{gy}); never reindex as c0/c1.
    # Keep unmatched previous cities so attack/move/capture still resolve.
    cities = stabilize(
        list(mapped["cities"]),
        (prev or {}).get("cities"),
        keep_missing=True,
    )
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
    info = driver.find_window()
    payload = {
        "hud": hud,
        "unit": unit,
        "units": units,
        "cities": cities,
        "cities_own": own,
        "cities_enemy": enemy,
        "villages": mapped["villages"],
        "villages_enabled": mapped.get("villages_enabled", False),
        "fog_edge": mapped.get("fog_edge") or [],
        "own_tribe": mapped["own_tribe"],
        "move_marks": _ids("m", pix["move_marks"]),
        "attack_marks": attack_marks,
        "fruit": _ids("f", pix["fruit"]),
        "selected_unit": strike,
        "overlay": overlay,
        "confirm_ready": confirm_ready,
        "ready": ready_flags(unit, overlay, confirm_ready),
        "hits_space": "screen",
        "window": {
            k: info.get(k)
            for k in ("found", "pid", "window_id", "width", "height", "match", "name")
        },
        "layout": coords.layout_info(),
        "screenshot": str(path),
    }
    payload["observe_diff"] = snapshot.diff(prev, payload, reason="frame") if prev else None
    payload["turn_diff"] = snapshot.diff(snapshot.last_saved(), payload, reason="observe")
    _LAST = payload
    return payload
