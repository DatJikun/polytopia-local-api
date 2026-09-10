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
    # Water/ice/sky must not become "Imperius".
    if is_water_rgb(r, g, b):
        return "unknown"
    # Oumaji sand / yellow tents
    if r >= 170 and g >= 130 and b <= 110 and (r + g) >= (2 * b + 80):
        return "oumaji"
    # Imperius royal blue — G well below B, not cyan
    if b >= 110 and b >= r + 40 and g <= min(b - 35, 125) and r <= 95:
        return "imperius"
    # Vengir wine / dark purple (Disrof etc.) — not grey Bardur, not water
    if (
        35 <= max(r, g, b) <= 125
        and b >= 45
        and g <= 75
        and r <= 115
        and b >= g + 10
        and r >= g
        and abs(r - b) <= 45
    ):
        return "vengir"
    # Bardur dark wood — not forest green, not near-black water shade
    if (
        55 <= max(r, g, b) <= 145
        and min(r, g, b) >= 50
        and abs(r - g) <= 25
        and abs(g - b) <= 30
        and abs(r - b) <= 30
        and g <= r + 15
    ):
        return "bardur"
    return "unknown"


def _owner_of(tribe: str) -> str:
    own = _own_tribe()
    if tribe == "unknown":
        return "unknown"
    return "own" if tribe == own else "enemy"


def villages_enabled() -> bool:
    return os.environ.get("POLYTOPIA_VILLAGES", "").strip().lower() in {"1", "true", "yes", "on"}


def sample_patch_tribe(arr: np.ndarray, x: int, y: int, radius: int = 10) -> tuple[str, list[int]]:
    """Majority tribe in a patch — one water pixel must not become Imperius."""
    h, w = arr.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = arr[y0:y1, x0:x1]
    if patch.size == 0:
        rgb = sample(arr, x, y)
        return classify_tribe(rgb), rgb
    votes: dict[str, int] = {}
    for py in range(patch.shape[0]):
        for px in range(patch.shape[1]):
            rgb = [int(v) for v in patch[py, px]]
            t = classify_tribe(rgb)
            if t != "unknown":
                votes[t] = votes.get(t, 0) + 1
    rgb = sample(arr, x, y)
    if not votes:
        return "unknown", rgb
    tribe = max(votes, key=votes.get)
    return tribe, rgb


def _bbox(xs: np.ndarray, ys: np.ndarray, cx: int, cy: int, radius: int) -> tuple[int, int]:
    if xs.size == 0:
        return 0, 0
    near = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    if not np.any(near):
        return 0, 0
    return int(xs[near].max() - xs[near].min()) + 1, int(ys[near].max() - ys[near].min()) + 1


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
        tribe, rgb = sample_patch_tribe(arr, cx, uy, radius=6)
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
                "tribe": tribe,
                "owner": _owner_of(tribe),
                "rgb": rgb,
            }
        )
    return out[:16]


def plate_name(arr: np.ndarray, cx: int, cy: int, bw: int, bh: int) -> str:
    """OCR the white nameplate. Empty if tesseract misses (tests / tiny plates)."""
    try:
        from pathlib import Path

        from PIL import Image

        from . import driver
    except Exception:
        return ""
    h, w = arr.shape[:2]
    x0 = max(0, int(cx - bw // 2 - 8))
    x1 = min(w, cx + bw // 2 + 8)
    y0 = max(0, int(cy - max(6, bh // 2) - 4))
    y1 = min(h, cy + max(6, bh // 2) + 4)
    if x1 - x0 < 12 or y1 - y0 < 6:
        return ""
    crop = Image.fromarray(arr[y0:y1, x0:x1].astype("uint8"), "RGB")
    try:
        text = driver.ocr_image(
            driver._prep_ocr(crop, scale=3, invert=True),
            Path("/tmp/polytopia-plate.png"),
            psm=7,
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    except Exception:
        return ""
    raw = "".join(ch for ch in (text or "") if ch.isalpha())
    if len(raw) < 3:
        return ""
    low = raw.lower()
    if low in {"score", "stars", "turn", "train", "capture", "polytopia", "next", "settings"}:
        return ""
    return raw[:18]


def find_cities(arr: np.ndarray) -> list[dict[str, Any]]:
    """White nameplates with a tribe-colored building above. Drops water/UI foam."""
    y0, y1, x0, x1 = _map_bounds(arr)
    region = arr[y0:y1, x0:x1]
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    white = (r >= 215) & (g >= 215) & (b >= 200) & ((np.maximum(r, g) - np.minimum(np.minimum(r, g), b)) <= 45)
    ys, xs = np.where(white)
    if xs.size == 0:
        return []
    axs, ays = xs + x0, ys + y0
    rad, mn = _cluster_params(arr, 22, 40)
    clusters = merge_clusters(_cluster(ays, axs, radius=rad, min_size=mn), dist=max(32, rad * 2))
    h, w = arr.shape[:2]
    s = min(w / coords.BASE_W, h / coords.BASE_H)
    up = max(12, int(round(28 * s)))
    min_w, max_w = int(24 * s), int(140 * s)
    min_h, max_h = max(4, int(5 * s)), int(24 * s)
    cities: list[dict[str, Any]] = []
    for n, cx, cy in clusters:
        if n > 2500:
            continue
        bw, bh = _bbox(axs, ays, cx, cy, max(24, rad * 2))
        if bh <= 0 or bw < min_w or bw > max_w or bh < min_h or bh > max_h:
            continue
        if bw / bh < 1.7:
            continue
        tribe, rgb = sample_patch_tribe(arr, cx, max(0, cy - up), radius=8)
        name = plate_name(arr, cx, cy, bw, bh)
        if tribe == "unknown" and not name:
            continue
        own = tribe == _own_tribe()
        if tribe == "unknown":
            own = False
        if not own and not name:
            # Sand / yellow UI next to a white streak is not an Oumaji city.
            if n < max(70, int(round(90 * s))) or bw < max(32, int(40 * s)):
                continue
        conf = 0.72 if own else 0.62
        if tribe == "imperius":
            conf = 0.55
        if name:
            conf = max(conf, 0.78)
        if tribe == "unknown":
            conf = 0.70 if name else 0.50
        if conf < 0.55:
            continue
        cities.append(
            {
                "id": f"c{len(cities)}",
                "kind": "city",
                "x": cx,
                "y": max(0, cy - up // 2),
                "plate_y": cy,
                "n": n,
                "w": bw,
                "h": bh,
                "name": name or None,
                "tribe": tribe,
                "owner": "own" if own else "enemy",
                "confidence": conf,
                "rgb": rgb,
            }
        )
    cities.sort(key=lambda c: (-float(c["confidence"]), -int(c["n"])))
    return cities[:8]


def find_villages(arr: np.ndarray, cities: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Off by default — tan dirt was flooding Domination with fake villages."""
    if not villages_enabled():
        return []
    y0, y1, x0, x1 = _map_bounds(arr)
    region = arr[y0:y1, x0:x1]
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    tan = (
        (r >= 140)
        & (r <= 200)
        & (g >= 90)
        & (g <= 155)
        & (b >= 40)
        & (b <= 95)
        & (r >= g + 15)
        & (g >= b + 12)
    )
    ys, xs = np.where(tan)
    if xs.size == 0:
        return []
    axs, ays = xs + x0, ys + y0
    rad, mn = _cluster_params(arr, 14, 28)
    clusters = merge_clusters(_cluster(ays, axs, radius=rad, min_size=mn), dist=max(18, rad * 2))
    city_pts = [(c["x"], c.get("plate_y", c["y"])) for c in (cities or [])]
    h, w = arr.shape[:2]
    s = min(w / coords.BASE_W, h / coords.BASE_H)
    area = (w * h) / (coords.BASE_W * coords.BASE_H)
    min_n = max(20, int(round(28 * area)))
    max_n = max(120, int(round(420 * area)))
    villages: list[dict[str, Any]] = []
    for n, cx, cy in clusters:
        if n < min_n or n > max_n:
            continue
        bw, bh = _bbox(axs, ays, cx, cy, max(16, rad * 2))
        if min(bw, bh) < max(6, int(8 * s)) or max(bw, bh) > int(40 * s):
            continue
        if bh <= 0 or not (0.5 <= bw / bh <= 2.0):
            continue
        rgb = sample(arr, cx, cy)
        if is_water_rgb(*rgb):
            continue
        if any((cx - x) ** 2 + (cy - y) ** 2 < 48 ** 2 for x, y in city_pts):
            continue
        villages.append(
            {
                "id": f"v{len(villages)}",
                "kind": "village",
                "x": cx,
                "y": cy,
                "n": n,
                "w": bw,
                "h": bh,
                "confidence": 0.68,
                "rgb": rgb,
            }
        )
    villages.sort(key=lambda v: -float(v["confidence"]))
    return villages[:4]


def find_fog_edge(arr: np.ndarray) -> list[dict[str, Any]]:
    """Dark unexplored tiles that touch lit map."""
    y0, y1, x0, x1 = _map_bounds(arr)
    region = arr[y0:y1, x0:x1]
    lum = region[:, :, 0].astype(np.int32) + region[:, :, 1] + region[:, :, 2]
    dark = lum <= 50
    lit = lum >= 140
    if not np.any(dark) or not np.any(lit):
        return []
    edge = np.zeros(dark.shape, dtype=bool)
    edge[1:, :] |= dark[1:, :] & lit[:-1, :]
    edge[:-1, :] |= dark[:-1, :] & lit[1:, :]
    edge[:, 1:] |= dark[:, 1:] & lit[:, :-1]
    edge[:, :-1] |= dark[:, :-1] & lit[:, 1:]
    ys, xs = np.where(edge)
    if xs.size == 0:
        return []
    rad, mn = _cluster_params(arr, 36, 12)
    clusters = merge_clusters(_cluster(ys + y0, xs + x0, radius=rad, min_size=mn), dist=max(40, rad * 2))
    out = []
    for n, cx, cy in clusters[:10]:
        out.append({"kind": "fog_edge", "x": cx, "y": cy, "n": n, "id": f"g{len(out)}"})
    return out


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
        "fog_edge": find_fog_edge(arr),
        "own_tribe": _own_tribe(),
        "villages_enabled": villages_enabled(),
    }
