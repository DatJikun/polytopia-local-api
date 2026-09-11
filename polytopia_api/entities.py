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

from . import combat
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
    # Dark-tile roofs (T44 Disrot / Rzgórst) sit below the old min=40 floor.
    purple = (r + b) / 2 - g
    if (
        g <= 105
        and max(r, b) >= 40
        and max(r, g, b) <= 190
        and min(r, b) >= 28
        and purple >= 10
        and abs(r - b) <= 60
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

    Vengir cities are dark-grey stone + purple roofs + gold window lights.
    Those lamps match Oumaji sand pixel-for-pixel. A few gold pixels on
    Bardur wood (nameplate star leak) must stay Bardur; gold *dominating*
    a dark patch is Disrof window lamps, not a desert city.
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
    oumaji_n = votes.get("oumaji", 0)
    bardur_n = votes.get("bardur", 0)
    total = sum(votes.values())
    mean_lum = float(patch.astype(np.int32).mean())
    # Purple roof is decisive (Disrof magenta). A handful of wine pixels on
    # Bardur wood must not steal cities_own (live: own dropped to 2 vs many).
    if vengir_n >= 3 or (total and vengir_n >= 0.08 * total):
        if bardur_n > vengir_n * 3 and bardur_n >= 0.45 * total and vengir_n < 12:
            return "bardur", rgb
        return "vengir", rgb
    if oumaji_n and vengir_n:
        return "vengir", rgb
    # Gold lamps match Oumaji sand. On dark stone they are Vengir windows —
    # unless Bardur wood is the majority (own cities + gold nameplate stars).
    if oumaji_n and bardur_n:
        if bardur_n > oumaji_n and (not total or bardur_n >= 0.45 * total):
            return "bardur", rgb
        return "vengir", rgb
    if oumaji_n and mean_lum < 135 and oumaji_n < 0.50 * total:
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


def _hp_green_mask(region: np.ndarray) -> np.ndarray:
    """Saturated lime HP bars. Dim mountain bars still count; grass does not."""
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    r32, g32, b32 = r.astype(np.int32), g.astype(np.int32), b.astype(np.int32)
    bright = (g >= 155) & (g >= r + 40) & (g >= b + 40) & (r <= 170) & (b <= 140)
    # Live 1280×800 mountain bars anti-alias toward ~140g.
    dim = (
        (g32 >= 138)
        & (g32 <= 200)
        & ((g32 - r32) >= 48)
        & ((g32 - b32) >= 48)
        & (r32 <= 155)
        & (b32 <= 125)
    )
    return bright | dim


def _is_leather_rgb(r: int, g: int, b: int) -> bool:
    """Bardur unit hide — warmer brown than mountain grey / snow / Vengir stone."""
    if r < 55 or r > 155 or g < 40 or b < 40:
        return False
    if r < g or r < b:
        return False
    if r - min(g, b) < 8:
        return False
    if max(r, g, b) - min(r, g, b) > 48:
        return False
    return not is_water_rgb(r, g, b)


def _unit_body_sample(arr: np.ndarray, cx: int, cy: int) -> tuple[str, list[int], int]:
    """Tribe under an HP bar. Snow/ice below the bar is a mountain, not water."""
    h, w = arr.shape[:2]
    drop = max(8, int(round(18 * h / coords.BASE_H)))
    offsets = [
        (0, drop),
        (0, drop + drop // 2),
        (0, drop * 2),
        (-8, drop),
        (8, drop),
        (0, max(6, drop // 2)),
    ]
    fallback_rgb = sample(arr, cx, min(h - 1, cy + drop))
    fallback_y = min(h - 1, cy + drop)
    for dx, dy in offsets:
        ux = min(max(0, cx + dx), w - 1)
        uy = min(max(0, cy + dy), h - 1)
        rgb = sample(arr, ux, uy)
        if is_water_rgb(*rgb):
            continue
        tribe, prgb = sample_patch_tribe(arr, ux, uy, radius=6)
        if tribe != "unknown":
            return tribe, prgb, uy
        if _is_leather_rgb(*rgb):
            return "bardur", rgb, uy
        fallback_rgb = prgb
        fallback_y = uy
    return "unknown", fallback_rgb, fallback_y


def _stamp_unit_tile(u: dict[str, Any], frame: tuple[int, int]) -> dict[str, Any]:
    try:
        gx, gy = combat.tile_grid(int(u["x"]), int(u["y"]), frame)
        u["tile"] = [gx, gy]
    except (KeyError, TypeError, ValueError):
        pass
    return u


def _unit_record(
    cx: int,
    uy: int,
    bar_y: int,
    n: int,
    tribe: str,
    rgb: list[int],
    hp: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "kind": "unit",
        "x": int(cx),
        "y": int(uy),
        "bar_y": int(bar_y),
        "n": int(n),
        "hp": hp,
        "tribe": tribe,
        "owner": _owner_of(tribe),
        "rgb": rgb,
    }
    if extra:
        rec.update(extra)
    return rec


def _hp_of_bar(arr: np.ndarray, cx: int, cy: int) -> str:
    pr, pg, pb = sample(arr, cx, cy)
    if pr >= 170 and pg <= 90:
        return "critical"
    if pr >= 170 and pg >= 140 and pb <= 90:
        return "hurt"
    return "full"


def find_units(arr: np.ndarray, cities: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Lime HP bars (not yellow city tents). Click the body just below the bar.

    Units on snowy mountains used to vanish: the pixel under the bar is ice,
    ``is_water_rgb`` skipped them, and /observe only listed southern warriors.
    Keep city-adjacent bars even when many larger HP bars exist elsewhere.
    """
    y0, y1, x0, x1 = _map_bounds(arr)
    region = arr[y0:y1, x0:x1]
    green = _hp_green_mask(region)
    ys, xs = np.where(green)
    rad, mn = _cluster_params(arr, 14, 6)
    mn = min(mn, 3)
    clusters = merge_clusters(_cluster(ys + y0, xs + x0, radius=rad, min_size=mn), dist=max(16, rad * 2))
    h, w = arr.shape[:2]
    frame = (w, h)
    # 1280×800 area-scale used to set max_n≈98 and drop a 16×5 mountain HP bar.
    max_n = max(160, int(round(360 * (h * w) / (coords.BASE_W * coords.BASE_H))))
    out: list[dict[str, Any]] = []
    for n, cx, cy in clusters:
        if n > max_n:
            continue
        tribe, rgb, uy = _unit_body_sample(arr, cx, cy)
        rec = _unit_record(cx, uy, cy, n, tribe, rgb, _hp_of_bar(arr, cx, cy))
        _stamp_unit_tile(rec, frame)
        rec["id"] = f"u{len(out)}"
        out.append(rec)
    out.sort(key=lambda u: (int(u.get("y") or 0), int(u.get("x") or 0)))
    for i, rec in enumerate(out):
        rec["id"] = f"u{i}"
    out = out[:36]
    if cities:
        return attach_units_near_cities(arr, out, cities)
    return out


def _lime_in_patch(arr: np.ndarray, x: int, y: int, radius: int) -> tuple[int, int, int]:
    h, w = arr.shape[:2]
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    crop = arr[y0:y1, x0:x1]
    if crop.size < 8:
        return 0, int(x), int(y)
    mask = _hp_green_mask(crop)
    n = int(mask.sum())
    if n <= 0:
        return 0, int(x), int(y)
    ys, xs = np.where(mask)
    return n, int(round(xs.mean() + x0)), int(round(ys.mean() + y0))


def _leather_in_patch(arr: np.ndarray, x: int, y: int, radius: int) -> tuple[int, int, int]:
    h, w = arr.shape[:2]
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    crop = arr[y0:y1, x0:x1]
    if crop.size < 8:
        return 0, int(x), int(y)
    r = crop[:, :, 0].astype(np.int32)
    g = crop[:, :, 1].astype(np.int32)
    b = crop[:, :, 2].astype(np.int32)
    leather = (
        (r >= 55) & (r <= 155) & (g >= 40) & (b >= 40)
        & (r >= g) & (r >= b)
        & ((r - np.minimum(g, b)) >= 8)
        & ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) <= 48)
    )
    n = int(leather.sum())
    if n <= 0:
        return 0, int(x), int(y)
    ys, xs = np.where(leather)
    return n, int(round(xs.mean() + x0)), int(round(ys.mean() + y0))


def _unit_near_existing(units: list[dict[str, Any]], x: int, y: int, pitch: int) -> bool:
    for u in units:
        try:
            if combat.tile_dist(int(u["x"]), int(u["y"]), x, y, pitch) <= 0.7:
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


def _tag_near_cities(units: list[dict[str, Any]], cities: list[dict[str, Any]], frame: tuple[int, int]) -> None:
    pitch = combat.hex_pitch(*frame)
    for u in units:
        best = None
        best_d = 1.85
        for c in cities or []:
            origin = combat.city_walk_center(c, frame) or combat.city_tile_center(c, frame)
            if origin is None:
                continue
            try:
                d = combat.tile_dist(int(u["x"]), int(u["y"]), origin[0], origin[1], pitch)
            except (KeyError, TypeError, ValueError):
                continue
            if d <= best_d:
                best_d = d
                best = c
        if best is None:
            continue
        cid = str(best.get("id") or best.get("city_id") or "")
        u["near_city_id"] = cid or None
        u["near_city_hex"] = round(best_d, 3)
        if best_d <= combat.ON_CITY_MARK_HEX:
            u["on_city_id"] = cid or None


def attach_units_near_cities(
    arr: np.ndarray,
    units: list[dict[str, Any]],
    cities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick up warriors whose HP bar hides under a nameplate.

    Mountain-south of Disrof (T36) and port-east (T37) both count. City
    building wood is not a unit.
    """
    if not cities:
        _tag_near_cities(units, cities, (arr.shape[1], arr.shape[0]))
        return units
    h, w = arr.shape[:2]
    frame = (w, h)
    pitch = combat.hex_pitch(*frame)
    rad = max(12, int(round(pitch * 0.7)))
    out = list(units)
    for c in cities:
        origin = combat.city_walk_center(c, frame) or combat.city_tile_center(c, frame)
        if origin is None:
            try:
                origin = (int(c["x"]), int(c["y"]))
            except (KeyError, TypeError, ValueError):
                continue
        cid = str(c.get("id") or c.get("city_id") or "")
        try:
            plate_y = int(c.get("plate_y") or origin[1])
        except (TypeError, ValueError):
            plate_y = int(origin[1])
        spots = [origin] + combat.hex_neighbors(origin[0], origin[1], pitch)
        try:
            spots.append((int(c["x"]), plate_y + int(round(0.75 * pitch))))
        except (KeyError, TypeError, ValueError):
            pass
        for i, (sx, sy) in enumerate(spots):
            if sx < 0 or sy < 0 or sx >= w or sy >= h:
                continue
            if _unit_near_existing(out, sx, sy, pitch):
                continue
            on_city = i == 0
            lime_n, lx, ly = _lime_in_patch(arr, sx, sy - max(4, pitch // 5), rad)
            leather_n, bx, by = (0, sx, sy)
            # City buildings are Bardur-wood / grey stone — never mint a unit
            # from the building column. East-port / south-mountain centroids
            # sit off that column (live T37 port-east of Disrof).
            if not on_city:
                leather_n, bx, by = _leather_in_patch(arr, sx, sy, rad)
                on_building = (
                    abs(bx - int(origin[0])) < max(10, pitch // 3)
                    and by <= plate_y + 4
                )
                if on_building:
                    leather_n = 0
            if lime_n < 3 and leather_n < 10:
                continue
            if lime_n >= 3:
                tribe, rgb, uy = _unit_body_sample(arr, lx, ly)
                rec = _unit_record(
                    lx,
                    uy,
                    ly,
                    lime_n,
                    tribe if tribe != "unknown" else (_own_tribe() if leather_n >= 8 else tribe),
                    rgb,
                    _hp_of_bar(arr, lx, ly),
                    extra={"from": "city_hex_hp", "near_city_id": cid or None},
                )
            else:
                rec = _unit_record(
                    bx,
                    by,
                    by,
                    leather_n,
                    _own_tribe(),
                    sample(arr, bx, by),
                    "full",
                    extra={"from": "city_hex_leather", "near_city_id": cid or None},
                )
            if rec.get("owner") == "unknown" and leather_n >= 10:
                rec["tribe"] = _own_tribe()
                rec["owner"] = "own"
            _stamp_unit_tile(rec, frame)
            if _unit_near_existing(out, int(rec["x"]), int(rec["y"]), pitch):
                continue
            rec["id"] = f"u{len(out)}"
            out.append(rec)
    _tag_near_cities(out, cities, frame)
    out.sort(
        key=lambda u: (
            0 if u.get("near_city_id") and float(u.get("near_city_hex") or 9) <= 1.5 else 1,
            int(u.get("y") or 0),
            int(u.get("x") or 0),
        )
    )
    for i, rec in enumerate(out):
        rec["id"] = f"u{i}"
    return out[:36]


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
    """OCR the nameplate. Moonrise: [blue icon] Name [gold star] on frost.

    Forced invert (HUD-style) turns letters white. Crop the *middle* of the
    plate so the icon/star do not become 'ounce' / 'EROS'.
    """
    try:
        from pathlib import Path

        from PIL import Image, ImageOps

        from . import driver
    except Exception:
        return ""
    h, w = arr.shape[:2]
    pad_x = max(8, bw // 10)
    x0 = max(0, int(cx - bw // 2 - pad_x))
    x1 = min(w, cx + bw // 2 + pad_x)
    y0 = max(0, int(cy - max(8, bh // 2) - 4))
    y1 = min(h, cy + max(8, bh // 2) + 4)
    if x1 - x0 < 12 or y1 - y0 < 6:
        return ""
    crop = Image.fromarray(arr[y0:y1, x0:x1].astype("uint8"), "RGB")
    cw, ch = crop.size
    inner = crop.crop((int(cw * 0.16), 0, max(int(cw * 0.16) + 1, int(cw * 0.78)), ch))
    if inner.size[0] < 10 or inner.size[1] < 5:
        inner = crop
    best = ""
    try:
        grey = ImageOps.autocontrast(inner.convert("L"))
        binary = grey.point(lambda p: 255 if p >= 145 else 0)
        variants = [
            (inner, None, 7),
            (inner, False, 7),
            (inner, False, 8),
            (binary, False, 8),
            (inner, True, 7),
        ]
        for src, invert, psm in variants:
            if src.mode == "L":
                prepared = src.resize(
                    (max(1, src.size[0] * 4), max(1, src.size[1] * 4)),
                    Image.Resampling.LANCZOS,
                )
            else:
                prepared = driver._prep_ocr(src, scale=4, invert=invert)
            text = driver.ocr_image(
                prepared,
                Path("/tmp/polytopia-plate.png"),
                psm=psm,
            )
            raw = _clean_city_name(text)
            if len(raw) > len(best):
                best = raw
            if len(raw) >= 5:
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


def _plate_is_frost(arr: np.ndarray, cx: int, cy: int, bw: int, bh: int) -> bool:
    """Moonrise frost banner (~180 grey), not a hot-white sand/UI streak."""
    h, w = arr.shape[:2]
    x0 = max(0, cx - max(8, bw // 2))
    x1 = min(w, cx + max(8, bw // 2) + 1)
    y0 = max(0, cy - max(4, bh // 2))
    y1 = min(h, cy + max(4, bh // 2) + 1)
    crop = arr[y0:y1, x0:x1]
    if crop.size < 20:
        return False
    r = crop[:, :, 0].astype(np.int32)
    g = crop[:, :, 1].astype(np.int32)
    b = crop[:, :, 2].astype(np.int32)
    chroma = np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))
    lum = (r + g + b) / 3.0
    frost = (lum >= 148) & (lum <= 215) & (chroma <= 36)
    return float(frost.mean()) >= 0.35


def _building_column(arr: np.ndarray, cx: int, cy: int, bw: int, up: int) -> tuple[int, int, int, int]:
    """Column above the frost plate — this city's building, not a neighbor.

    Plate clustering can sit a few px left of the roof; bw/3 missed Disrof magenta.
    A full plate-half (~60px) also ate the east-port warrior (live T37).
    """
    h, w = arr.shape[:2]
    half = max(22, min(32, max(bw // 3, bw // 4 + 8)))
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    y0 = max(0, cy - up * 2 - 16)
    y1 = max(y0 + 1, min(h, cy - 10))
    return x0, x1, y0, y1


def _column_tribe(arr: np.ndarray, cx: int, cy: int, bw: int, up: int) -> tuple[str, list[int]]:
    """Tribe of the building itself. Skip lime HP bars and water (garrison / port)."""
    x0, x1, y0, y1 = _building_column(arr, cx, cy, bw, up)
    crop = arr[y0:y1, x0:x1]
    rgb = sample(arr, cx, max(0, (y0 + y1) // 2))
    if crop.size == 0:
        return classify_tribe(rgb), rgb
    lime = _hp_green_mask(crop)
    votes: dict[str, int] = {}
    for py in range(crop.shape[0]):
        for px in range(crop.shape[1]):
            if lime[py, px]:
                continue
            pix = [int(v) for v in crop[py, px]]
            if is_water_rgb(*pix):
                continue
            t = classify_tribe(pix)
            if t != "unknown":
                votes[t] = votes.get(t, 0) + 1
    if not votes:
        return classify_tribe(rgb), rgb
    vengir_n = votes.get("vengir", 0)
    bardur_n = votes.get("bardur", 0)
    total = sum(votes.values())
    if bardur_n and bardur_n >= vengir_n * 3 and (not total or bardur_n >= 0.40 * total) and vengir_n < 12:
        return "bardur", rgb
    if vengir_n >= 8 or (total and vengir_n >= 0.08 * total):
        return "vengir", rgb
    return max(votes, key=votes.get), rgb


def _vengir_roof_pixels(arr: np.ndarray, cx: int, cy: int, bw: int, up: int) -> int:
    """Magenta/wine roof can sit well above the frost plate (outside radius=14)."""
    x0, x1, y0, y1 = _building_column(arr, cx, cy, bw, up)
    crop = arr[y0:y1, x0:x1]
    if crop.size < 20:
        return 0
    r = crop[:, :, 0].astype(np.int32)
    g = crop[:, :, 1].astype(np.int32)
    b = crop[:, :, 2].astype(np.int32)
    purple = (r + b) / 2 - g
    mag = (
        (g <= 105)
        & (np.maximum(r, b) >= 40)
        & (np.maximum(np.maximum(r, g), b) <= 190)
        & (np.minimum(r, b) >= 28)
        & (purple >= 10)
        & (np.abs(r - b) <= 60)
    )
    return int(mag.sum())


def _gold_window_pixels(arr: np.ndarray, cx: int, cy: int, bw: int, up: int) -> int:
    """Gold lamps on the building (above the plate), not the nameplate star."""
    x0, x1, y0, y1 = _building_column(arr, cx, cy, bw, up)
    crop = arr[y0:y1, x0:x1]
    if crop.size < 20:
        return 0
    r = crop[:, :, 0].astype(np.int32)
    g = crop[:, :, 1].astype(np.int32)
    b = crop[:, :, 2].astype(np.int32)
    gold = (r >= 180) & (g >= 140) & (g <= 220) & (b <= 110) & (r >= b + 60)
    return int(gold.sum())


def _plate_gold_split(arr: np.ndarray, cx: int, cy: int, bw: int, bh: int) -> tuple[int, int]:
    """Gold on the frost nameplate: (window lamps, level-star).

    Moonrise: [icon] name [gold star on the right]. Disrof lamps are two (or
    more) gold runs on the plate. A single compact blob is the level star —
    even when a split plate fragment puts that star in the left of its crop.
    """
    h, w = arr.shape[:2]
    pad = max(12, bw // 2 + 8)
    x0 = max(0, cx - pad)
    x1 = min(w, cx + pad + 1)
    y0 = max(0, cy - max(4, bh // 2))
    y1 = min(h, cy + max(4, bh // 2) + 1)
    crop = arr[y0:y1, x0:x1]
    if crop.size < 20:
        return 0, 0
    r = crop[:, :, 0].astype(np.int32)
    g = crop[:, :, 1].astype(np.int32)
    b = crop[:, :, 2].astype(np.int32)
    gold = (r >= 180) & (g >= 140) & (g <= 220) & (b <= 110) & (r >= b + 60)
    gold_n = int(gold.sum())
    if gold_n < 12:
        return 0, gold_n
    cols = np.any(gold, axis=0)
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(cols.tolist()):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, int(cols.shape[0])))
    # Two+ gold runs on the plate are Disrof lamps. A single blob is the
    # Moonrise level star unless it is much wider than a star.
    star_w = max(18, int(round(22 * w / coords.BASE_W)))
    if len(runs) >= 2:
        return gold_n, 0
    if len(runs) == 1 and (runs[0][1] - runs[0][0]) >= star_w + 10:
        return gold_n, 0
    # Star lives on the right. Gold only on the left of a split frost
    # fragment is Disrof windows (live 0bfb6f7), not Ufla's star.
    if len(runs) == 1:
        run_mid = (runs[0][0] + runs[0][1]) / 2.0
        if run_mid < crop.shape[1] * 0.45:
            return gold_n, 0
    return 0, gold_n


def _disrof_signal(
    roof_n: int,
    gold_n: int,
    frost_plate: bool = False,
    plate_window_n: int = 0,
) -> bool:
    """Live Disrof: magenta roof, building lamps, or gold windows on the frost plate.

    A garrison / adjacent unit eats some magenta; 8px was enough to drop T37.
    """
    if roof_n >= 8:
        return True
    if frost_plate and roof_n >= 4:
        return True
    if not frost_plate:
        return False
    return gold_n >= 8 or plate_window_n >= 8



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
        roof_n = _vengir_roof_pixels(arr, cx, cy, bw, up)
        gold_n = _gold_window_pixels(arr, cx, cy, bw, up)
        frost_plate = _plate_is_frost(arr, cx, cy, bw, bh)
        plate_win_n, _plate_star_n = _plate_gold_split(arr, cx, cy, bw, bh)
        strong_vengir = _disrof_signal(roof_n, gold_n, frost_plate, plate_win_n)
        # Adjacent unit on the plate can kill letter-contrast; Disrof still counts.
        if not _plate_contrast(arr, cx, cy, bw, bh) and not strong_vengir:
            continue
        sampled_tribe, rgb = sample_patch_tribe(
            arr, cx, max(0, cy - up - max(4, bh // 2)), radius=14
        )
        col_tribe, col_rgb = _column_tribe(arr, cx, cy, bw, up)
        tribe = sampled_tribe
        if is_water_rgb(*rgb):
            rgb = col_rgb
            if not strong_vengir and col_tribe != "vengir":
                continue
        bardurish = (
            sampled_tribe == "bardur"
            or col_tribe == "bardur"
            or classify_tribe(rgb) == "bardur"
            or (marks and frost_plate and sampled_tribe != "vengir")
        )
        # Magenta roof is enough. Gold lamps / plate windows only count on frost
        # (Oumaji sand matches gold pixel-for-pixel and is not Disrof).
        real_disrof = roof_n >= 8 or (
            frost_plate and (gold_n >= 8 or plate_win_n >= 8)
        )
        if real_disrof or (strong_vengir and not bardurish):
            tribe = "vengir"
        elif strong_vengir and bardurish:
            tribe = "bardur"
        elif tribe == "vengir":
            # Purple noise without a magenta roof / lamps: grey Bardur wood stays own.
            lum = (int(rgb[0]) + int(rgb[1]) + int(rgb[2])) / 3.0
            chroma = max(int(rgb[0]), int(rgb[1]), int(rgb[2])) - min(
                int(rgb[0]), int(rgb[1]), int(rgb[2])
            )
            if classify_tribe(rgb) == "bardur" or col_tribe == "bardur" or (
                40 <= lum <= 145 and chroma <= 40
            ):
                tribe = "bardur"
            elif marks and frost_plate:
                tribe = _own_tribe()
            else:
                continue
        elif tribe == "oumaji" and frost_plate:
            if classify_tribe(rgb) == "bardur" or col_tribe == "bardur":
                tribe = "bardur"
            else:
                continue
        # Skip plate OCR — names are optional; tesseract on frost fragments
        # invented ghosts like "Arch" and made /attack hang past 12s.
        name = ""
        if tribe == "unknown":
            continue
        own = tribe == _own_tribe()
        if not own and tribe == "oumaji":
            # Bright sand / yellow UI, not a Moonrise frost city.
            continue
        conf = 0.72 if own else 0.62
        if tribe == "imperius":
            # Live T39: a water/sky fragment tagged imperius next to Disrof.
            if not frost_plate and not name:
                continue
            conf = 0.62
        if tribe == "vengir":
            conf = 0.70 if name else 0.64
            if strong_vengir:
                conf = max(conf, 0.78)
        if name and (tribe != _own_tribe() or len(name) >= 5):
            conf = max(conf, 0.76)
        if tribe == "unknown":
            conf = 0.70 if name else 0.50
        if conf < 0.55:
            continue
        evidence: list[str] = []
        if frost_plate:
            evidence.append("frost_plate")
        if roof_n >= 8 or (strong_vengir and roof_n >= 4):
            evidence.append("magenta_roof")
        if gold_n >= 8 or plate_win_n >= 8:
            evidence.append("gold_lamps")
        if marks and "gold_lamps" not in evidence:
            evidence.append("gold_star")
        building_y = max(0, cy - up // 2)
        frame = (w, h)
        tile_xy = combat.city_tile_center(
            {"x": cx, "y": building_y, "plate_y": cy},
            frame,
        )
        if tile_xy is None:
            tile_xy = (cx, building_y)
        gx, gy = combat.tile_grid(tile_xy[0], tile_xy[1], frame)
        cid = combat.city_id_for_tile(gx, gy)
        cities.append(
            {
                "id": cid,
                "city_id": cid,
                "kind": "city",
                "x": cx,
                "y": building_y,
                "plate_y": cy,
                "tile": [gx, gy],
                "tile_xy": [tile_xy[0], tile_xy[1]],
                "n": n,
                "w": bw,
                "h": bh,
                "name": name or None,
                "tribe": tribe,
                "owner": "own" if own else "enemy",
                "confidence": conf,
                "evidence": evidence,
                "rgb": rgb,
                "roof_n": roof_n,
                "gold_n": gold_n,
            }
        )
    cities.sort(
        key=lambda c: (
            0 if c.get("owner") == "enemy" else 1,
            -float(c["confidence"]),
            -int(c["n"]),
        )
    )
    return _finalize_cities(cities)


def _city_merge_limit(a: dict[str, Any], b: dict[str, Any], dist: int) -> int:
    """Same hex → one city. Own Bardur must not be eaten by a nearby Vengir FP.

    One Disrof plate can split into a grey-stone (Bardur-looking) fragment and a
    gold/magenta hit. Those share a nameplate; distinct cities do not.
    """
    ta, tb = a.get("tile"), b.get("tile")
    if ta is not None and tb is not None and ta == tb:
        return 10**6
    if _plates_overlap(a, b):
        return 10**6
    oa, ob = a.get("owner"), b.get("owner")
    if oa and ob and oa != ob:
        # Same hex / overlapping nameplate already merged above (grey-stone
        # half of Disrof). A 64px radius ate nearby Bardur (live: cities_own=2).
        return 0
    tribes = {a.get("tribe"), b.get("tribe")}
    if tribes == {"vengir"}:
        # Same nameplate still merges via tile / plate overlap above.
        # Do not collapse Disrof + Rzgórst at 80px on a zoomed late-game map.
        return 0
    return dist


def _plates_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    try:
        ax, aw = int(a["x"]), int(a.get("w") or 40)
        bx, bw = int(b["x"]), int(b.get("w") or 40)
        ay = int(a.get("plate_y") or a["y"])
        by = int(b.get("plate_y") or b["y"])
    except (KeyError, TypeError, ValueError):
        return False
    if abs(ay - by) > 14:
        return False
    # Gold star punches a hole in the frost, so one nameplate can split into
    # two clusters ~100px apart. Merge those; distinct cities are farther.
    gap = abs(ax - bx) - (aw + bw) / 2
    oa, ob = a.get("owner"), b.get("owner")
    ta, tb = a.get("tile"), b.get("tile")
    tiles_differ = False
    if isinstance(ta, (list, tuple)) and isinstance(tb, (list, tuple)) and len(ta) >= 2 and len(tb) >= 2:
        try:
            tiles_differ = (int(ta[0]), int(ta[1])) != (int(tb[0]), int(tb[1]))
        except (TypeError, ValueError):
            tiles_differ = False
    if oa and ob and oa != ob and tiles_differ:
        # Neighbor Bardur 1 hex from Disrof (live: cities_own collapsed to 2).
        # Split halves of ONE banner share a tile and merge above.
        return False
    return gap <= 64


def _is_disrof_hit(c: dict[str, Any]) -> bool:
    ev = c.get("evidence") or []
    return c.get("tribe") == "vengir" and ("magenta_roof" in ev or "gold_lamps" in ev)


def _prefer_city(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    # Same nameplate: gold/magenta Disrof wins over the grey-stone half.
    # Distinct hexes with different owners never reach here unless plates overlap.
    def _score(c: dict[str, Any]) -> tuple:
        ev = c.get("evidence") or []
        return (
            1 if _is_disrof_hit(c) else 0,
            1 if "magenta_roof" in ev else 0,
            int(c.get("roof_n") or 0),
            int(c.get("gold_n") or 0),
            1 if "gold_lamps" in ev else 0,
            int(c.get("n") or 0),
        )

    if _score(a) != _score(b) and (_is_disrof_hit(a) or _is_disrof_hit(b)):
        win, lose = (a, b) if _score(a) > _score(b) else (b, a)
        out = dict(win)
        if lose.get("name") and not out.get("name"):
            out["name"] = lose["name"]
        return out
    own = _own_tribe()
    if a.get("owner") == "own" and b.get("owner") != "own" and a.get("tribe") == own:
        return a
    if b.get("owner") == "own" and a.get("owner") != "own" and b.get("tribe") == own:
        return b
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


def _keep_sticky_city(c: dict[str, Any]) -> bool:
    """Unnamed frost FPs must not accumulate across observes (2→4→7)."""
    if c.get("owner") == "own":
        return True
    if c.get("name"):
        return True
    if _is_disrof_hit(c):
        return True
    return False


def session_cities(cities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """After stabilize(keep_missing): drop unseen frost FPs, recap unnamed Vengir.

    Own-city growth (T44: 12 Bardur) must not wipe visible Vengir off cities_enemy.
    """
    kept: list[dict[str, Any]] = []
    for c in cities or []:
        if c.get("seen") is False and not _keep_sticky_city(c):
            continue
        kept.append(c)
    return _finalize_cities(kept)


# Live T44/T45: cities_own filled the old combined [:12] and cities_enemy went [].
# Never slice enemy to make room for Bardur.
MAX_CITIES_OWN = 24


def _finalize_cities(cities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every real enemy city. A 12-own map must not empty cities_enemy."""
    cities = _cap_vengir(_dedupe_cities(cities))
    own = [c for c in cities if c.get("owner") == "own"]
    enemy = [c for c in cities if c.get("owner") == "enemy"]
    other = [c for c in cities if c.get("owner") not in {"own", "enemy"}]
    return enemy + own[:MAX_CITIES_OWN] + other


def _cap_vengir(cities: list[dict[str, Any]], unnamed_limit: int = 2) -> list[dict[str, Any]]:
    """Keep every Disrof-class hit; only unnamed frost fragments are capped.

    OCR names are off (ghost Arch). Three visible Vengir cities must not collapse
    to two because they are unnamed. Real enemies are listed first so a later
    own-city cap cannot drop them.
    """
    named: list[dict[str, Any]] = []
    real: list[dict[str, Any]] = []
    frost_fp: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for c in cities:
        if c.get("tribe") != "vengir":
            others.append(c)
        elif c.get("name") or _is_disrof_hit(c):
            if c.get("name"):
                named.append(c)
            else:
                real.append(c)
        else:
            frost_fp.append(c)

    def _score(c: dict[str, Any]) -> tuple:
        ev = c.get("evidence") or []
        return (
            1 if "magenta_roof" in ev else 0,
            1 if "gold_lamps" in ev else 0,
            float(c.get("confidence") or 0),
            int(c.get("n") or 0),
        )

    real.sort(key=_score, reverse=True)
    frost_fp.sort(key=_score, reverse=True)
    return named + real + frost_fp[:unnamed_limit] + others


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
    cities = find_cities(arr)
    units = attach_units_near_cities(arr, find_units(arr), cities)
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
