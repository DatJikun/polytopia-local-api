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
    mark = nearest_mark(attack_marks or [], to_xy[0], to_xy[1], max(pitch * 2, 40)) if to_xy else None
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
