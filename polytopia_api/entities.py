"""Map entities from a live-frame screenshot (screen pixels).

Units ≈ HP bars, cities ≈ frosted nameplates + tribe color above them,
villages ≈ small tan hut clusters that are not nameplates. Heuristic —
think_turn should treat low-n hits as hints, not truth.
"""

from __future__ import annotations

import os
import re
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
    # Vengir: wine walls AND magenta/purple roofs (Disrof). Grey stone is Bardur.
    purple = (r + b) / 2 - g
    if (
        g <= 95
        and max(r, b) >= 48
        and max(r, g, b) <= 175
        and min(r, b) >= 40
        and purple >= 12
        and abs(r - b) <= 55
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
    """Majority tribe in a patch — one water pixel must not become Imperius.

    Vengir cities are dark-grey stone + purple roofs. Majority vote would call
    them Bardur; any real purple roof wins.
    """
    h, w = arr.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius * 2), min(h, y + radius + 1)
    patch = arr[y0:y1, x0:x1]
    rgb = sample(arr, x, y)
    if patch.size == 0:
        return classify_tribe(rgb), rgb
    votes: dict[str, int] = {}
    for py in range(patch.shape[0]):
        for px in range(patch.shape[1]):
            pix = [int(v) for v in patch[py, px]]
            t = classify_tribe(pix)
            if t != "unknown":
                votes[t] = votes.get(t, 0) + 1
    if not votes:
        return "unknown", rgb
    vengir_n = votes.get("vengir", 0)
    total = sum(votes.values())
    if vengir_n >= 5 or (total and vengir_n >= 0.12 * total):
        return "vengir", rgb
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


def _clean_city_name(text: str) -> str:
    raw = "".join(ch for ch in (text or "") if ch.isalpha() or ch in "øæåØÆÅ")
    if len(raw) < 4:
        return ""
    low = raw.lower()
    if low in {"score", "stars", "turn", "train", "capture", "polytopia", "next", "settings"}:
        return ""
    if re.search(r"(.)\1{2,}", low):
        return ""
    if not any(c in "aeiouyøæå" for c in low):
        return ""
    return raw[:18]


def plate_name(arr: np.ndarray, cx: int, cy: int, bw: int, bh: int) -> str:
    """OCR the nameplate. Moonrise plates are light-grey frost with dark text.

    Forced invert (HUD-style) turns letters white and tesseract misses Disrof.
    """
    try:
        from pathlib import Path

        from PIL import Image

        from . import driver
    except Exception:
        return ""
    h, w = arr.shape[:2]
    x0 = max(0, int(cx - bw // 2 - 10))
    x1 = min(w, cx + bw // 2 + 10)
    y0 = max(0, int(cy - max(8, bh // 2) - 6))
    y1 = min(h, cy + max(8, bh // 2) + 6)
    if x1 - x0 < 12 or y1 - y0 < 6:
        return ""
    crop = Image.fromarray(arr[y0:y1, x0:x1].astype("uint8"), "RGB")
    best = ""
    try:
        for invert in (None, False, True):
            text = driver.ocr_image(
                driver._prep_ocr(crop, scale=3, invert=invert),
                Path("/tmp/polytopia-plate.png"),
                psm=7,
            )
            raw = _clean_city_name(text)
            if len(raw) > len(best):
                best = raw
            if len(raw) >= 4:
                break
    except Exception:
        return best
    return best


def _plate_mask(region: np.ndarray) -> np.ndarray:
    """Moonrise nameplates are frosted ~180 grey, not solid 215 white.

    Over dark Vengir terrain the banner sits around 150–210. Requiring 215+
    dropped Disrof. Snow/buildings are filtered later by aspect + contrast.
    """
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    r32, g32, b32 = r.astype(np.int32), g.astype(np.int32), b.astype(np.int32)
    chroma = np.maximum(r32, np.maximum(g32, b32)) - np.minimum(r32, np.minimum(g32, b32))
    frost = (r >= 148) & (g >= 148) & (b >= 140) & (r <= 242) & (chroma <= 32)
    hot = (r >= 210) & (g >= 210) & (b >= 195) & (chroma <= 45)
    return frost | hot


def _city_label_marks(arr: np.ndarray, cx: int, cy: int, bw: int, bh: int) -> bool:
    """Moonrise city labels carry a gold level-star (and often a blue pop icon)."""
    h, w = arr.shape[:2]
    x0 = max(0, cx - max(10, bw // 2 + 6))
    x1 = min(w, cx + max(10, bw // 2 + 6) + 1)
    y0 = max(0, cy - max(6, bh // 2 + 4))
    y1 = min(h, cy + max(6, bh // 2 + 4) + 1)
    crop = arr[y0:y1, x0:x1]
    if crop.size < 20:
        return False
    r, g, b = crop[:, :, 0].astype(np.int32), crop[:, :, 1].astype(np.int32), crop[:, :, 2].astype(np.int32)
    gold = (r >= 180) & (g >= 140) & (g <= 220) & (b <= 110) & (r >= b + 60)
    n = int(gold.sum())
    return 6 <= n <= 420


def _plate_contrast(arr: np.ndarray, cx: int, cy: int, bw: int, bh: int) -> bool:
    """Letters (or the star/icon) punch holes in the frost; snow does not.

    Solid white test banners have no ink; they still count if the crop is hot.
    """
    h, w = arr.shape[:2]
    x0 = max(0, cx - max(8, bw // 2))
    x1 = min(w, cx + max(8, bw // 2) + 1)
    y0 = max(0, cy - max(4, bh // 2))
    y1 = min(h, cy + max(4, bh // 2) + 1)
    crop = arr[y0:y1, x0:x1]
    if crop.size < 30:
        return False
    lum = crop.astype(np.int32).mean(axis=2)
    if float((lum >= 210).mean()) >= 0.55:
        return True
    return float(lum.std()) >= 14.0 and float(lum.min()) <= 120 and float(lum.max()) >= 155


def find_cities(arr: np.ndarray) -> list[dict[str, Any]]:
    """Frosted nameplates with a tribe-colored building above. Drops water/UI foam."""
    y0, y1, x0, x1 = _map_bounds(arr)
    region = arr[y0:y1, x0:x1]
    plate = _plate_mask(region)
    ys, xs = np.where(plate)
    if xs.size == 0:
        return []
    axs, ays = xs + x0, ys + y0
    rad, mn = _cluster_params(arr, 22, 32)
    clusters = merge_clusters(_cluster(ays, axs, radius=rad, min_size=mn), dist=max(32, rad * 2))
    h, w = arr.shape[:2]
    s = min(w / coords.BASE_W, h / coords.BASE_H)
    area = (w * h) / (coords.BASE_W * coords.BASE_H)
    up = max(22, int(round(50 * s)))
    min_w, max_w = int(22 * s), int(200 * s)
    min_h, max_h = max(4, int(5 * s)), max(16, int(36 * s))
    max_n = max(2200, int(round(4000 * area)))
    cities: list[dict[str, Any]] = []
    for n, cx, cy in clusters:
        if n > max_n:
            continue
        if cy < int(h * 0.06) or coords.in_dock_zone(cx, cy):
            continue
        bw, bh = _bbox(axs, ays, cx, cy, max(24, rad * 2))
        if bh <= 0:
            continue
        marks = _city_label_marks(arr, cx, cy, bw, bh)
        lim_h = max_h * 2 if marks else max_h
        lim_w = int(max_w * 1.35) if marks else max_w
        if bw < min_w or bw > lim_w or bh < min_h or bh > lim_h:
            continue
        if bw / bh < (1.15 if marks else 1.55):
            continue
        if not _plate_contrast(arr, cx, cy, bw, bh):
            continue
        tribe, rgb = sample_patch_tribe(arr, cx, max(0, cy - up), radius=16)
        if is_water_rgb(*rgb):
            continue
        name = plate_name(arr, cx, cy, bw, bh)
        if tribe == "unknown" and not name:
            continue
        own = tribe == _own_tribe()
        if tribe == "unknown":
            own = False
        if not own and not name and tribe == "oumaji":
            # Sand / yellow UI next to a frost streak is not an Oumaji city.
            # Vengir/Imperius keep the hit even when OCR misses Disrof.
            if n < max(70, int(round(90 * s))) or bw < max(32, int(40 * s)):
                continue
        conf = 0.72 if own else 0.62
        if tribe == "imperius":
            conf = 0.55
        if tribe == "vengir":
            conf = 0.70 if name else 0.64
        if name and (tribe != _own_tribe() or len(name) >= 5):
            conf = max(conf, 0.76)
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
    cities.sort(
        key=lambda c: (
            0 if c.get("owner") == "enemy" else 1,
            -float(c["confidence"]),
            -int(c["n"]),
        )
    )
    return _dedupe_cities(cities)[:12]


def _city_merge_limit(a: dict[str, Any], b: dict[str, Any], dist: int) -> int:
    tribes = {a.get("tribe"), b.get("tribe")}
    if tribes == {"vengir", "bardur"}:
        return max(dist, 96)
    return dist


def _prefer_city(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if a.get("tribe") == "vengir" and b.get("tribe") != "vengir":
        win = dict(a)
        if b.get("name") and not win.get("name"):
            win["name"] = b["name"]
        return win
    if b.get("tribe") == "vengir" and a.get("tribe") != "vengir":
        win = dict(b)
        if a.get("name") and not win.get("name"):
            win["name"] = a["name"]
        return win
    if a.get("name") and not b.get("name"):
        return a
    if b.get("name") and not a.get("name"):
        return b
    if float(a.get("confidence") or 0) >= float(b.get("confidence") or 0):
        return a
    return b


def _dedupe_cities(cities: list[dict[str, Any]], dist: int = 56) -> list[dict[str, Any]]:
    """One city can split into frost fragments; keep the better tribe/name.

    Vengir grey stone is often a second Bardur-looking hit next to the purple roof.
    """
    items = list(cities)
    for _ in range(6):
        out: list[dict[str, Any]] = []
        merged = False
        for c in items:
            hit = None
            best_d = 10**9
            for i, o in enumerate(out):
                d2 = (int(c["x"]) - int(o["x"])) ** 2 + (int(c["y"]) - int(o["y"])) ** 2
                lim = _city_merge_limit(c, o, dist)
                if d2 <= lim * lim and d2 < best_d:
                    best_d = d2
                    hit = i
            if hit is None:
                out.append(c)
                continue
            merged = True
            out[hit] = _prefer_city(c, out[hit])
        items = out
        if not merged:
            break
    return items


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
