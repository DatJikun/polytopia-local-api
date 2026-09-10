"""Map entities from a live-frame screenshot (screen pixels).

Units ≈ HP bars, cities ≈ white nameplates + tribe color above them,
villages ≈ small tan hut clusters that are not nameplates. Heuristic —
think_turn should treat low-n hits as hints, not truth.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import coords
from .detect import _cluster, _cluster_params, _map_bounds, is_water_rgb, merge_clusters, sample


def _own_tribe() -> str:
    return os.environ.get("POLYTOPIA_TRIBE", "bardur").strip().lower() or "bardur"


def classify_tribe(rgb: list[int] | tuple[int, int, int]) -> str:
    r, g, b = (int(x) for x in rgb[:3])
    # Oumaji sand / yellow tents
    if r >= 170 and g >= 130 and b <= 130 and (r + g) >= (2 * b + 70):
        return "oumaji"
    # Imperius / Kickoo blues
    if b >= r + 25 and b >= g + 10 and b >= 90:
        return "imperius"
    # Bardur dark wood / grey
    if max(r, g, b) <= 150 and abs(r - g) <= 30 and abs(g - b) <= 35:
        return "bardur"
    return "unknown"


def _owner_of(tribe: str) -> str:
    own = _own_tribe()
    if tribe == "unknown":
        return "unknown"
    return "own" if tribe == own else "enemy"


def find_units(arr: np.ndarray) -> list[dict[str, Any]]:
    """Lime HP bars (not yellow city tents). Click the body just below the bar."""
    y0, y1, x0, x1 = _map_bounds(arr)
    region = arr[y0:y1, x0:x1]
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    # Saturated lime; Oumaji yellow fails g >= r+40.
    green = (g >= 155) & (g >= r + 40) & (g >= b + 40) & (r <= 170) & (b <= 140)
    ys, xs = np.where(green)
    rad, mn = _cluster_params(arr, 14, 6)
    clusters = merge_clusters(_cluster(ys + y0, xs + x0, radius=rad, min_size=mn), dist=max(16, rad * 2))
    h, w = arr.shape[:2]
    drop = max(8, int(round(18 * h / coords.BASE_H)))
    max_n = max(40, int(round(220 * (h * w) / (coords.BASE_W * coords.BASE_H))))
    out: list[dict[str, Any]] = []
    for n, cx, cy in clusters:
        if n > max_n:
            continue
        uy = min(h - 1, cy + drop)
        rgb = sample(arr, cx, uy)
        if is_water_rgb(*rgb):
            continue
        pr, pg, pb = sample(arr, cx, cy)
        hp = "full"
        if pr >= 170 and pg <= 90:
            hp = "critical"
        elif pr >= 170 and pg >= 140 and pb <= 90:
            hp = "hurt"
        out.append(
            {
                "id": f"u{len(out)}",
                "kind": "unit",
                "x": cx,
                "y": uy,
                "bar_y": cy,
                "n": n,
                "hp": hp,
                "rgb": rgb,
            }
        )
    return out[:16]


def find_cities(arr: np.ndarray) -> list[dict[str, Any]]:
    """White-ish nameplates; tribe color sampled above the plate."""
    y0, y1, x0, x1 = _map_bounds(arr)
    region = arr[y0:y1, x0:x1]
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    white = (r >= 215) & (g >= 215) & (b >= 200) & ((np.maximum(r, g) - np.minimum(np.minimum(r, g), b)) <= 45)
    ys, xs = np.where(white)
    rad, mn = _cluster_params(arr, 22, 28)
    clusters = merge_clusters(_cluster(ys + y0, xs + x0, radius=rad, min_size=mn), dist=max(28, rad * 2))
    h, w = arr.shape[:2]
    up = max(12, int(round(28 * h / coords.BASE_H)))
    cities: list[dict[str, Any]] = []
    for n, cx, cy in clusters:
        # Nameplates are wider than tall.
        if n > 2200:
            continue
        rgb = sample(arr, cx, max(0, cy - up))
        tribe = classify_tribe(rgb)
        cities.append(
            {
                "id": f"c{len(cities)}",
                "kind": "city",
                "x": cx,
                "y": max(0, cy - up // 2),
                "plate_y": cy,
                "n": n,
                "tribe": tribe,
                "owner": _owner_of(tribe),
                "rgb": rgb,
            }
        )
    return cities[:12]


def find_villages(arr: np.ndarray, cities: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Small tan/brown hut clusters, away from city plates."""
    y0, y1, x0, x1 = _map_bounds(arr)
    region = arr[y0:y1, x0:x1]
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    tan = (
        (r >= 130)
        & (r <= 210)
        & (g >= 85)
        & (g <= 170)
        & (b >= 35)
        & (b <= 110)
        & (r >= g + 8)
        & (g >= b + 8)
    )
    ys, xs = np.where(tan)
    rad, mn = _cluster_params(arr, 16, 12)
    clusters = merge_clusters(_cluster(ys + y0, xs + x0, radius=rad, min_size=mn), dist=max(20, rad * 2))
    city_pts = [(c["x"], c["plate_y"] if "plate_y" in c else c["y"]) for c in (cities or [])]
    max_n = max(80, int(round(400 * (arr.shape[0] * arr.shape[1]) / (coords.BASE_W * coords.BASE_H))))
    villages: list[dict[str, Any]] = []
    for n, cx, cy in clusters:
        if n > max_n:
            continue
        rgb = sample(arr, cx, cy)
        if is_water_rgb(*rgb):
            continue
        if any((cx - x) ** 2 + (cy - y) ** 2 < 40 ** 2 for x, y in city_pts):
            continue
        villages.append(
            {
                "id": f"v{len(villages)}",
                "kind": "village",
                "x": cx,
                "y": cy,
                "n": n,
                "rgb": rgb,
            }
        )
    return villages[:10]


def observe_map(arr: np.ndarray) -> dict[str, Any]:
    units = find_units(arr)
    cities = find_cities(arr)
    villages = find_villages(arr, cities)
    own = [c for c in cities if c.get("owner") == "own"]
    enemy = [c for c in cities if c.get("owner") == "enemy"]
    return {
        "units": units,
        "cities": cities,
        "cities_own": own,
        "cities_enemy": enemy,
        "villages": villages,
        "own_tribe": _own_tribe(),
    }
