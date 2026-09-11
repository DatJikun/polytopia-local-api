"""Unit type, attack range, and whether a tile can be struck.

Pixel distance is a hex estimate. Red attack marks from the live frame
are ground truth; this layer only explains rafts / out-of-range.
"""

from __future__ import annotations

from typing import Any

# Melee 1, ranged 2–3. Naval cannot hit land cities.
RANGE: dict[str, int] = {
    "warrior": 1,
    "rider": 1,
    "defender": 1,
    "knight": 1,
    "giant": 1,
    "mind_bender": 1,
    "archer": 2,
    "catapult": 3,
    "raft": 1,
    "scout": 1,
    "rammer": 1,
    "bomber": 2,
    "boat": 1,
    "ship": 1,
}
NAVAL = {"raft", "scout", "rammer", "bomber", "boat", "ship"}
LAND_ONLY = {"warrior", "rider", "defender", "knight", "giant", "archer", "catapult", "mind_bender"}


def hex_pitch(frame_w: int, frame_h: int) -> int:
    return max(24, int(round(36 * min(frame_w / 1280, frame_h / 800))))


def tile_grid(x: int, y: int, frame: tuple[int, int]) -> list[int]:
    """Stable city id key. Same hex → same [gx, gy] across turns."""
    pitch = hex_pitch(*frame)
    gy_pitch = max(1, int(round(pitch * 0.75)))
    return [int(round(int(x) / pitch)), int(round(int(y) / gy_pitch))]


def city_id_for_tile(gx: int, gy: int) -> str:
    return f"c{int(gx)}_{int(gy)}"


def city_tile_center(city: dict[str, Any] | None, frame: tuple[int, int]) -> tuple[int, int] | None:
    """Standable hex — the building, not the nameplate and not the roof edge.

    Moonrise plates hang *under* the city. ``plate_y + pitch`` in the request
    is one hex offset from the plate onto the tile; on screen +y is down, so
    the hex center is above the plate by ~0.75 pitch (same distance).
    """
    if not city:
        return None
    if city.get("tile_xy"):
        try:
            return int(city["tile_xy"][0]), int(city["tile_xy"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    pitch = hex_pitch(*frame)
    x = int(city["x"])
    if city.get("plate_y") is not None:
        plate = int(city["plate_y"])
        # One hex from the plate toward the building (up the screen).
        y = plate - int(round(0.75 * pitch))
        if y < 0:
            y = int(city.get("y") or plate)
        return x, max(0, y)
    return x, int(city["y"])


def tile_dist(x0: int, y0: int, x1: int, y1: int, pitch: int) -> float:
    dx = (int(x1) - int(x0)) / max(1, pitch)
    dy = (int(y1) - int(y0)) / (max(1, pitch) * 0.75)
    return (dx * dx + dy * dy) ** 0.5


def unit_profile(unit_type: str | None) -> dict[str, Any]:
    kind = (unit_type or "").strip().lower() or None
    if kind == "boat":
        kind = "scout"
    rng = RANGE.get(kind) if kind else None
    naval = kind in NAVAL if kind else None
    return {
        "type": kind,
        "range": rng,
        "naval": bool(naval) if kind else None,
        "land_attack": False if kind in NAVAL else (True if kind in LAND_ONLY else None),
    }


def nearest_unit(
    units: list[dict[str, Any]],
    x: int,
    y: int,
    max_dist: int,
) -> dict[str, Any] | None:
    return nearest_mark(units, x, y, max_dist)


def garrison_unit(
    units: list[dict[str, Any]],
    city: dict[str, Any] | None,
    frame: tuple[int, int],
    max_hex: float = 0.55,
) -> dict[str, Any] | None:
    """Enemy (or any) unit standing on the city tile — HP bar, not the plate."""
    center = city_tile_center(city, frame)
    if not city or center is None:
        return None
    pitch = hex_pitch(*frame)
    hits: list[tuple[float, dict[str, Any]]] = []
    cx, cy = center
    for u in units or []:
        try:
            d = tile_dist(int(u["x"]), int(u["y"]), cx, cy, pitch)
        except (KeyError, TypeError, ValueError):
            continue
        if d <= max_hex:
            hits.append((d, u))
    if not hits:
        return None
    hits.sort(key=lambda t: (0 if str(t[1].get("owner") or "") == "enemy" else 1, t[0]))
    return hits[0][1]


def unit_on_city(
    units: list[dict[str, Any]],
    city: dict[str, Any] | None,
    frame: tuple[int, int],
    panel: dict[str, Any] | None = None,
    max_hex: float = 0.4,
) -> dict[str, Any]:
    """True when a unit stands ON the city tile (Capture), not merely adjacent."""
    panel = panel or {}
    if not city:
        return {"ok": False, "reason": "no_city"}
    pitch = hex_pitch(*frame)
    center = city_tile_center(city, frame)
    if center is None:
        return {"ok": False, "reason": "no_city", "hex_pitch": pitch}
    hit = None
    best_d = max_hex
    for u in units or []:
        try:
            d = tile_dist(int(u["x"]), int(u["y"]), center[0], center[1], pitch)
        except (KeyError, TypeError, ValueError):
            continue
        if d <= best_d:
            best_d = d
            hit = u
    if hit is None:
        return {
            "ok": False,
            "reason": "not_standing_on_city",
            "hex_pitch": pitch,
            "tile_xy": [center[0], center[1]],
            "hex_dist": None,
        }
    return {
        "ok": True,
        "reason": "unit_on_tile",
        "unit": hit,
        "hex_pitch": pitch,
        "tile_xy": [center[0], center[1]],
        "hex_dist": round(best_d, 3),
    }


def nearest_mark_hex(
    marks: list[dict[str, Any]],
    x: int,
    y: int,
    frame: tuple[int, int],
    max_hex: float,
) -> tuple[dict[str, Any] | None, float | None]:
    pitch = hex_pitch(*frame)
    best = None
    best_d = max_hex
    for m in marks or []:
        try:
            d = tile_dist(int(m["x"]), int(m["y"]), int(x), int(y), pitch)
        except (KeyError, TypeError, ValueError):
            continue
        if d <= best_d:
            best_d = d
            best = m
    return best, (None if best is None else round(best_d, 3))


def nearest_mark(
    marks: list[dict[str, Any]],
    x: int,
    y: int,
    max_dist: int,
) -> dict[str, Any] | None:
    best = None
    best_d = max_dist
    for m in marks or []:
        try:
            d = ((int(m["x"]) - x) ** 2 + (int(m["y"]) - y) ** 2) ** 0.5
        except (KeyError, TypeError, ValueError):
            continue
        if d <= best_d:
            best_d = d
            best = m
    return best


def can_strike(
    unit_type: str | None,
    from_xy: tuple[int, int] | None,
    to_xy: tuple[int, int] | None,
    to_kind: str | None,
    frame: tuple[int, int],
    attack_marks: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Red marks win. Type/range only vetoes obvious illegal hits (raft → city)."""
    prof = unit_profile(unit_type)
    pitch = hex_pitch(*frame)
    dist = None
    if from_xy and to_xy:
        dist = round(tile_dist(from_xy[0], from_xy[1], to_xy[0], to_xy[1], pitch), 2)
    mark = None
    if to_xy:
        mark, _ = nearest_mark_hex(attack_marks or [], to_xy[0], to_xy[1], frame, 1.15)
    land_city = (to_kind or "") in {"city", "village"}
    if prof["type"] in NAVAL and land_city:
        return {
            **prof,
            "ok": False,
            "reason": "naval_no_land",
            "hint": "raft/ship cannot attack a land city — disembark first",
            "tile_dist": dist,
            "hex_pitch": pitch,
            "red_mark": mark,
        }
    if mark is not None:
        return {**prof, "ok": True, "reason": "red_mark", "tile_dist": dist, "hex_pitch": pitch, "red_mark": mark}
    if dist is not None and isinstance(prof["range"], int) and dist > prof["range"] + 0.65:
        return {
            **prof,
            "ok": False,
            "reason": "out_of_range",
            "tile_dist": dist,
            "hex_pitch": pitch,
            "red_mark": None,
        }
    if not (attack_marks or []):
        return {
            **prof,
            "ok": False,
            "reason": "no_attack_marks",
            "hint": "select the attacker first; red hexes are the live attack set",
            "tile_dist": dist,
            "hex_pitch": pitch,
            "red_mark": None,
        }
    return {**prof, "ok": False, "reason": "no_mark_on_target", "tile_dist": dist, "hex_pitch": pitch, "red_mark": None}
