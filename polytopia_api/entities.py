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


def _is_magenta_roof_rgb(r: int, g: int, b: int) -> bool:
    """True Disrof / dark-tile magenta — not Bardur wood with a wine shadow.

    classify_tribe(vengir) is looser (purple>=10) so wall pixels still vote.
    Roof evidence needs a real green-deficit: (88,70,82) is wood shade;
    (150,74,144) / (70,35,80) / (36,20,48) are roofs.
    Garrison-cover dark purple (48,36,58) must still count (T37 cover_roof).
    """
    purple = (r + b) / 2 - g
    return (
        g <= 95
        and max(r, b) >= 36
        and max(r, g, b) <= 190
        and min(r, b) >= 20
        and purple >= 16
        and (max(r, b) - g) >= 20
        and abs(r - b) <= 60
    )


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
    mag_n = 0
    for py in range(patch.shape[0]):
        for px in range(patch.shape[1]):
            pix = [int(v) for v in patch[py, px]]
            if _is_magenta_roof_rgb(*pix[:3]):
                mag_n += 1
    # Purple roof is decisive (Disrof magenta). Wine-shadow on Bardur wood
    # must not steal cities_own (live T46: own dropped to 2–3 vs many).
    if bardur_n and bardur_n >= vengir_n and (not total or bardur_n >= 0.35 * total):
        if mag_n >= 12 and mag_n > bardur_n:
            return "vengir", rgb
        return "bardur", rgb
    if vengir_n >= 8 or (total and vengir_n >= 0.12 * total):
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


def _bardur_wood_stats(
    arr: np.ndarray, cx: int, cy: int, bw: int, up: int
) -> tuple[int, int, int, float]:
    """Warm Bardur wood in the building column: (n, width, height, density).

    A longhouse is a tall wood bbox (live moonrise can be sparse). A
    unit-on-snow or mountain speck is short/sparse (c12_16). Tall-empty
    frost (live T46 c17_14: dens≈0.02) is not a longhouse either.
    """
    x0, x1, y0, y1 = _building_column(arr, cx, cy, bw, up)
    crop = arr[y0:y1, x0:x1]
    if crop.size == 0:
        return 0, 0, 0, 0.0
    xs: list[int] = []
    ys: list[int] = []
    n = 0
    for py in range(crop.shape[0]):
        for px in range(crop.shape[1]):
            r, g, b = (int(v) for v in crop[py, px][:3])
            if classify_tribe((r, g, b)) == "bardur" and r >= b - 4:
                n += 1
                xs.append(px)
                ys.append(py)
    if n == 0:
        return 0, 0, 0, 0.0
    ww = max(xs) - min(xs) + 1
    wh = max(ys) - min(ys) + 1
    dens = n / float(crop.shape[0] * crop.shape[1])
    return n, ww, wh, dens


def _warm_bardur_pixels(arr: np.ndarray, cx: int, cy: int, bw: int, up: int) -> int:
    """Bardur longhouse wood is warm grey. Cool mountain stone is not a city."""
    return _bardur_wood_stats(arr, cx, cy, bw, up)[0]


# Live T46 after #44: moonrise wine-grey is lighter than g<=72 / purple>=5.
# (82,74,80) / (88,80,86) vote Bardur and were 0 wine_n, so c14_23 vanished.
WINE_SALVAGE_MIN = 4


def _wine_wall_pixels(arr: np.ndarray, cx: int, cy: int, bw: int, up: int) -> int:
    """Loose wine/purple in the building column (dark Vengir that votes Bardur).

    Live c14_23 walls plurality-vote Bardur so sampled_tribe *and* col_tribe
    are bardur; classify_tribe(vengir) (purple>=10) misses wine-grey stone.
    #44's g<=72 missed lighter moonrise veins (g≈74–86, purple≈4–8). Real
    longhouses (90,85,80) have purple≈0. Thresholds stay below magenta roofs
    so wine-shadow on Bufla does not need this count.
    """
    x0, x1, y0, y1 = _building_column(arr, cx, cy, bw, up)
    crop = arr[y0:y1, x0:x1]
    if crop.size < 20:
        return 0
    r = crop[:, :, 0].astype(np.int32)
    g = crop[:, :, 1].astype(np.int32)
    b = crop[:, :, 2].astype(np.int32)
    purple = (r + b) / 2.0 - g
    wine = (
        (g <= 94)
        & (purple >= 3.0)
        & (np.minimum(r, b) >= 16)
        & (np.maximum(np.maximum(r, g), b) <= 175)
        & (np.abs(r.astype(np.int32) - b) <= 90)
    )
    # Live T46 after #47: c14_23 veins (96,92,90) are purple≈1 / g=92 so
    # wine_n stayed 0 and the 5th enemy vanished. Bardur wood (90,85,80)
    # is g=85 / purple=0 — keep that out.
    ash = (
        (g >= 88)
        & (g <= 96)
        & (r >= 88)
        & (b >= 84)
        & (purple >= 0.5)
        & (purple <= 8.0)
        & (np.abs(r.astype(np.int32) - b) <= 12)
        & (np.maximum(np.maximum(r, g), b) <= 110)
    )
    return int((wine | ash).sum())


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
    # Plurality wood wins. The old 3× rule let wine-shadow roofs eat Bardur.
    if bardur_n and bardur_n >= vengir_n and (not total or bardur_n >= 0.35 * total):
        return "bardur", rgb
    if vengir_n >= 8 or (total and vengir_n >= 0.12 * total):
        return "vengir", rgb
    return max(votes, key=votes.get), rgb


def _vengir_roof_pixels(arr: np.ndarray, cx: int, cy: int, bw: int, up: int) -> int:
    """Magenta/wine roof can sit well above the frost plate (outside radius=14).

    Loose classify_tribe purple (wine-shadow on Bardur wood) is not a roof.
    """
    x0, x1, y0, y1 = _building_column(arr, cx, cy, bw, up)
    crop = arr[y0:y1, x0:x1]
    if crop.size < 20:
        return 0
    r = crop[:, :, 0].astype(np.int32)
    g = crop[:, :, 1].astype(np.int32)
    b = crop[:, :, 2].astype(np.int32)
    purple = (r + b) / 2.0 - g
    mag = (
        (g <= 95)
        & (np.maximum(r, b) >= 36)
        & (np.maximum(np.maximum(r, g), b) <= 190)
        & (np.minimum(r, b) >= 20)
        & (purple >= 16)
        & ((np.maximum(r, b) - g) >= 20)
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
        # Gold-star Bardur banners (live cities_own 3–4 vs a full screen) still count.
        if not _plate_contrast(arr, cx, cy, bw, bh) and not strong_vengir and not marks:
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
        wood = (
            sampled_tribe == "bardur"
            or col_tribe == "bardur"
            or classify_tribe(rgb) == "bardur"
        )
        warm_n, wood_w, wood_h, wood_dens = _bardur_wood_stats(arr, cx, cy, bw, up)
        rec_shape = {
            "warm_n": warm_n,
            "wood_w": wood_w,
            "wood_h": wood_h,
            "wood_dens": wood_dens,
            "gold_n": gold_n,
            "n": n,
            "w": bw,
        }
        own_longhouse = wood and _is_own_longhouse_hit(rec_shape)
        # Magenta roof is enough. Huge building gold / plate-window *pairs* are
        # Disrof. Modest gold on Bardur wood next to a nameplate star is the
        # Ufla-class roof lamps (live T46: those must stay cities_own, not ghosts
        # and not stolen as Vengir). After #47 a few wine-shadow px (roof_n≥8)
        # plus recruit gold leak tagged Orkork magenta_roof+gold_lamps enemy.
        disrof_gold = frost_plate and (
            plate_win_n >= 8
            or gold_n >= 280
            or (gold_n >= 8 and not (wood and marks))
        )
        if own_longhouse:
            dens_lh = float(wood_dens or 0)
            if dens_lh >= 0.34:
                # Synthetic / solid-wood Disrof with plate windows stays enemy.
                real_disrof = roof_n >= 8 or gold_n >= 280 or plate_win_n >= 8
            else:
                # Sparse live longhouse: wine-shadow roof_n≈8 and recruit plate-gold
                # leak must not steal Orkork. Real Disrof roofs are hundreds of px.
                real_disrof = roof_n >= 40 or gold_n >= 280
        else:
            real_disrof = roof_n >= 8 or disrof_gold
        if own_longhouse and not real_disrof:
            tribe = "bardur"
        elif real_disrof or (strong_vengir and not wood):
            tribe = "vengir"
        elif strong_vengir and wood:
            tribe = "bardur"
        elif tribe == "vengir":
            # Purple noise without a magenta roof / lamps: grey Bardur wood stays own.
            # Mountain grey / frost+gold without wood used to become cities_own ghosts
            # (live T46: 5–7 Bardur vs 2 on screen — c15_15 / c15_8 / c21_23).
            if wood:
                tribe = "bardur"
            elif frost_plate and (marks or _plate_contrast(arr, cx, cy, bw, bh)):
                # Dark Vengir city: frost nameplate + gold star, no magenta this
                # frame (live T46: cities_enemy 4 vs Game Stats 5).
                tribe = "vengir"
            else:
                continue
        elif tribe == "oumaji" and frost_plate:
            if wood:
                tribe = "bardur"
            else:
                continue
        # Skip plate OCR — names are optional; tesseract on frost fragments
        # invented ghosts like "Arch" and made /attack hang past 12s.
        name = ""
        wine_n = _wine_wall_pixels(arr, cx, cy, bw, up) if frost_plate else 0
        dark_vengir = False
        # Live T46 after #45: light wine-grey c14_23 was wine_n==0 on every
        # hit, so the unknown+wine salvage never fired. City-sized frost+star
        # that is not a longhouse still counts as the 5th Vengir.
        if tribe == "unknown" and frost_plate and marks:
            warm_u, ww_u, wh_u, dens_u = _bardur_wood_stats(arr, cx, cy, bw, up)
            rec_unknown = {
                "warm_n": warm_u,
                "wood_w": ww_u,
                "wood_h": wh_u,
                "wood_dens": dens_u,
                "gold_n": gold_n,
                "n": n,
                "w": bw,
            }
            if wine_n >= WINE_SALVAGE_MIN or _own_miss_is_dark_vengir(
                rec_unknown,
                sampled_tribe=sampled_tribe,
                col_tribe=col_tribe,
                frost_plate=True,
                marks=True,
                wine_n=wine_n,
            ):
                tribe = "vengir"
                dark_vengir = True
        if tribe == "unknown":
            continue
        own = tribe == _own_tribe()
        grey_stone = False
        if own:
            # Ufla-class: Bardur wood under a frost/hot plate. Gold star preferred
            # but a lettered full plate still counts when the star clustered onto a
            # neighbor. Mid-dense low-gold (any width) is Vengir grey stone — live
            # T46 after #46 dropped ww≈47 c12_16 and cities_enemy stayed 4.
            lettered = frost_plate and _plate_contrast(arr, cx, cy, bw, bh)
            # #32's warm_n>=500 floor ate live Bufla/Orkork (cities_own 1→0).
            # Synthetic longhouses are ~1600+; live moonrise wood is far sparser.
            if _own_wood_is_grey_stone_mass(rec_shape):
                # Live c12_16: Vengir grey stone matches Bardur wood (warm≈936
                # dens≈0.28 gold_n≈5) and stole cities_own while cities_enemy
                # under-counted 5→4. Flip to enemy — do not drop the city.
                tribe = "vengir"
                own = False
                grey_stone = True
            elif _own_miss_is_dark_vengir(
                rec_shape,
                sampled_tribe=sampled_tribe,
                col_tribe=col_tribe,
                frost_plate=frost_plate,
                marks=bool(marks or lettered),
                wine_n=wine_n,
            ):
                # Live T46 after #43: wine-grey c14_23 voted Bardur+Bardur,
                # failed the longhouse gate, and vanished (enemy 5→4). Keep.
                tribe = "vengir"
                own = False
                dark_vengir = True
            elif (
                not wood
                or warm_n < 30
                or not _own_wood_looks_like_building(rec_shape)
            ):
                continue
            else:
                if not marks and not lettered:
                    continue
                # HUD stars + frost chrome classify as Bardur wood (live T46: c18_1
                # y≈45 / gy=1 while Orkork/Bufla/Grugri were the only cities on screen).
                if _plate_in_top_chrome(cy, h):
                    continue
        elif (
            tribe == "vengir"
            and frost_plate
            and marks
            and wine_n >= WINE_SALVAGE_MIN
            and gold_n < 24
        ):
            # wood=False frost+star path: unnamed frost FPs are capped at 2.
            # Tag wine-grey c14_23 as dark_vengir so it is a real 5th enemy.
            dark_vengir = True
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
        if own and warm_n >= 30:
            evidence.append("bardur_wood")
        if grey_stone:
            evidence.append("grey_stone")
        if dark_vengir:
            evidence.append("dark_vengir")
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
                "warm_n": warm_n,
                "wood_w": wood_w,
                "wood_h": wood_h,
                "wood_dens": wood_dens,
                "wine_n": wine_n,
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


# Live Orkork halves ~35px. Bufla+Orkork ~80px must stay two (#32 over-merged
# when coords.frame() was 1920 → pitch 54 → 1.2*pitch≈65).
OWN_SPLIT_MAX_PX = 48


def _own_close_plate_px() -> int:
    """Same Bardur plate can hash two tiles just outside cluster-merge.

    Live T46 Orkork: c15_24 @(541,657) + c16_24 @(574,667) ≈35px (cluster
    merge is 32px @ 1280 so both survive). Neighbor cities sit ~80px; a
    64px radius ate those. Hard-cap below 80 even if frame() is 1920.
    """
    try:
        fw, fh = coords.frame()
    except Exception:
        fw, fh = 1280, 800
    if fw <= 0 or fh <= 0:
        fw, fh = 1280, 800
    pitch = combat.hex_pitch(fw, fh)
    return max(36, min(OWN_SPLIT_MAX_PX, int(round(pitch * 1.2))))


def _with_city_extras(win: dict[str, Any], lose: dict[str, Any]) -> dict[str, Any]:
    """Keep the winning fragment; copy name / evidence from the other half."""
    out = dict(win)
    if lose.get("name") and not out.get("name"):
        out["name"] = lose["name"]
    ev = list(out.get("evidence") or [])
    for item in lose.get("evidence") or []:
        if item not in ev:
            ev.append(item)
    if ev:
        out["evidence"] = ev
    if int(lose.get("n") or 0) > int(out.get("n") or 0):
        out["n"] = lose["n"]
    if int(lose.get("w") or 0) > int(out.get("w") or 0):
        out["w"] = lose["w"]
    return out


def _city_merge_limit(a: dict[str, Any], b: dict[str, Any], dist: int) -> int:
    """Same hex → one city. Own Bardur must not be eaten by a nearby Vengir FP.

    One Disrof plate can split into a grey-stone (Bardur-looking) fragment and a
    gold/magenta hit. Those share a nameplate; distinct cities do not.
    """
    oa, ob = a.get("owner"), b.get("owner")
    ta, tb = a.get("tile"), b.get("tile")
    if ta is not None and tb is not None and ta == tb:
        return 10**6
    if oa == "own" and ob == "own" and _city_sep2(a, b) > OWN_SPLIT_MAX_PX * OWN_SPLIT_MAX_PX:
        # Live T46 after #32: Bufla+Orkork ~80px became one city then 0.
        # Same-tile splits still merge above; 1-hex neighbors never do.
        # Gold-star hole can hash gx±2 (~72px, live c15_12 vs Orkork c17_12)
        # — merge only when one hit is a fragment, not two full longhouses.
        if _own_plate_is_split_fragment(a, b) and _city_sep2(a, b) < 80 * 80:
            return 79
        return 0
    if _plates_overlap(a, b):
        return 10**6
    if oa and ob and oa != ob:
        # Same hex / overlapping nameplate already merged above (grey-stone
        # half of Disrof). A 64px radius ate nearby Bardur (live: cities_own=2).
        return 0
    if oa == "own" and ob == "own":
        # Same city can hash two tiles ~1 pitch apart (live Orkork c15_24 +
        # c16_24 at ~35px). True 1-hex neighbors sit ~80px.
        return _own_close_plate_px()
    tribes = {a.get("tribe"), b.get("tribe")}
    if tribes == {"vengir"}:
        # Same nameplate still merges via tile / plate overlap above (halves
        # ~35px). Kissing banners of two cities (~80px) must not collapse
        # Distop + grey-stone c12_16 (live T46: cities_enemy 4 vs Game Stats 5).
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
    if tiles_differ:
        # Gold-star / window holes split one plate ~50–68px. Two cities sit
        # ~80px (live T46: grey-stone c12_16 + Distop merged, enemy 4 vs 5).
        if _city_sep2(a, b) >= 72 * 72:
            return False
        if oa == "own" and ob == "own":
            # Neighbor Bardur 1 hex apart: banners kiss but they are two cities.
            return False
        if oa and ob and oa != ob:
            # Neighbor Bardur 1 hex from Disrof (live: cities_own collapsed to 2).
            return False
    return gap <= 64


def _is_disrof_hit(c: dict[str, Any]) -> bool:
    ev = c.get("evidence") or []
    return c.get("tribe") == "vengir" and (
        "magenta_roof" in ev
        or "gold_lamps" in ev
        or "grey_stone" in ev
        or "dark_vengir" in ev
    )


def _city_sep2(a: dict[str, Any], b: dict[str, Any]) -> int:
    """Squared pixel distance; own plates also compare plate_y (banner centroid)."""
    try:
        dx = int(a["x"]) - int(b["x"])
        dy = int(a["y"]) - int(b["y"])
    except (KeyError, TypeError, ValueError):
        return 10**9
    best = dx * dx + dy * dy
    pa, pb = a.get("plate_y"), b.get("plate_y")
    if pa is not None and pb is not None:
        try:
            dpy = int(pa) - int(pb)
            best = min(best, dx * dx + dpy * dpy)
        except (TypeError, ValueError):
            pass
    return best


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
    if a.get("owner") == "own" and b.get("owner") == "own":
        def _own_score(c: dict[str, Any]) -> tuple:
            ev = c.get("evidence") or []
            # Mid-dense grey-stone fragments can have higher warm_n than a
            # sparse longhouse (live Orkork gold-star half vs c16_24).
            return (
                0 if _own_wood_is_grey_stone_mass(c) else 1,
                1 if _is_real_own_hit({**c, "owner": "own"}) else 0,
                int(c.get("n") or 0),
                int(c.get("w") or 0),
                int(c.get("wood_h") or 0),
                1 if "gold_star" in ev or "gold_lamps" in ev else 0,
            )

        win, lose = (a, b) if _own_score(a) >= _own_score(b) else (b, a)
        return _with_city_extras(win, lose)
    if a.get("name") and not b.get("name"):
        return _with_city_extras(a, b)
    if b.get("name") and not a.get("name"):
        return _with_city_extras(b, a)
    if float(a.get("confidence") or 0) >= float(b.get("confidence") or 0):
        return _with_city_extras(a, b)
    return _with_city_extras(b, a)


def _hud_bottom(frame_h: int | None = None) -> int:
    """Bottom of the top chrome (stars / turn / BACK). HUD_CROP is 0.12 of height."""
    h = int(frame_h or 0)
    if h <= 0:
        try:
            _w, h = coords.frame()
        except Exception:
            h = 800
    return max(40, int(h * 0.12))


def _plate_in_top_chrome(plate_y: int, frame_h: int | None = None) -> bool:
    return int(plate_y) <= _hud_bottom(frame_h)


def _own_hit_in_top_chrome(c: dict[str, Any]) -> bool:
    """HUD / fog-edge frost is not a Bardur city (live T46: c18_1 @ y≈45).

    Dark HUD chrome matches Bardur wood; the gold star icon matches a
    nameplate mark. Real cities have their frost plate *below* the HUD.
    """
    try:
        _w, h = coords.frame()
    except Exception:
        h = 800
    hud_y1 = _hud_bottom(h)
    y = int(c.get("y") or 0)
    plate = c.get("plate_y")
    py = int(plate) if plate is not None else y
    if _plate_in_top_chrome(py, h):
        return True
    gy = 99
    tile = c.get("tile")
    if isinstance(tile, (list, tuple)) and len(tile) >= 2:
        try:
            gy = int(tile[1])
        except (TypeError, ValueError):
            gy = 99
    # Fog row from HUD frost (c18_1). Synthetic helpers use gy=0 at y=200 — keep those.
    if gy <= 1 and y <= hud_y1:
        return True
    return False


def _own_wood_footprint(c: dict[str, Any]) -> bool:
    """Bufla/Orkork longhouse wood bbox — independent of plate n/w.

    Sparse live wood is ww≳36 wh≳60 dens≈0.08. Mid-dense grey stone can be
    just as wide (live T46 c12_16 ww≈47 dens=0.277) — that is not a keep
    signal by itself. Frost/wood FPs after /recruit gold-leak can also
    inflate to ww≳36 (live c15_12).
    """
    return (
        int(c.get("wood_w") or 0) >= 36
        and int(c.get("wood_h") or 0) >= 60
        and int(c.get("warm_n") or 0) >= 140
        and float(c.get("wood_dens") or 0) >= 0.06
    )


def _own_wood_is_grey_stone_mass(c: dict[str, Any]) -> bool:
    """Vengir grey stone matches Bardur wood; live c12_16 is not a longhouse.

    Live Game Stats Bardur 2 / Vengir 5: c12_16 @(441,455) warm=936 wood_h=72
    dens=0.277 gold_n=5. After #46 a ww≳36 gate treated that as Bufla, so
    find_cities dropped it (not grey_stone, not city-sized sparse own) and
    cities_enemy stayed 4. Width does not matter — Bufla/Orkork are sparse
    (dens≈0.08–0.09) with roof lamps (gold_n 68–116). Synthetic solid
    longhouses are denser still (dens≳0.45) with gold_n=0. Dict fixtures
    omit gold_n — not this. Never a warm_n>=500 keep-floor.
    """
    if c.get("gold_n") is None or c.get("wood_dens") is None:
        return False
    dens = float(c.get("wood_dens") or 0)
    gold_n = int(c.get("gold_n") or 0)
    wh = int(c.get("wood_h") or 0)
    # Mid-density tall mass, almost no building lamps. Live c12_16
    # dens=0.277 ww≈32–47. Width is not a keep/drop: #46's ww<36 missed
    # the wide 5th Vengir; skinny leftovers beside Distop (ww≈9) are still
    # that city. Split halves of Bufla/Orkork (ww≈13, ~40px from the
    # longhouse) are dropped later. Sparse longhouses (dens≈0.08) stay own.
    # Scaled synthetic longhouses land dens≳0.40 / gold_n=0 — must stay own.
    return gold_n < 16 and 0.16 <= dens < 0.34 and wh >= 50


def _sparse_own_longhouse(c: dict[str, Any]) -> bool:
    """Live Bufla/Orkork wood footprint — keep even when roof lamps are occluded.

    #42 required gold_n>=24 at dens<0.34 and dropped Orkork/c16_24 (cities_own
    2→1). #43's ww>=28/wh>=50/warm>=140 also matched live frost/wood FPs
    (c15_12 / c15_15 / c15_24) so own flickered 2→3. Real longhouses are
    wider AND taller (Bufla/Orkork ww≳36 wh≳60 warm≳140 dens≳0.06, plate
    w≳70 or n≳900). Dict fixtures that omit plate n/w stay wood-only.
    Gold-lamp split plates (n/w below this) use `_gold_lamp_sparse_longhouse`.
    """
    if _own_wood_is_grey_stone_mass(c):
        return False
    if not _own_wood_footprint(c):
        return False
    # Already-vetted dict fixtures omit plate size.
    if c.get("w") is None and c.get("n") is None:
        return True
    return int(c.get("w") or 0) >= 70 or int(c.get("n") or 0) >= 900


def _gold_lamp_sparse_longhouse(c: dict[str, Any]) -> bool:
    """Bufla/Orkork roof lamps on sparse wood — gold-star hole can shrink the plate.

    #47 required dens<0.34 → city-sized plate (n≥900/w≥70) and live T46 clean
    observe dropped both longhouses (cities_own 2→0). Roof lamps (gold_n 68–116,
    warm≈260) keep even when n/w is a split fragment. Gold-leaked FPs after
    /recruit (c15_12 gold≈40 warm≈180 n≈620 w≈58) stay out. Grey stone is
    mid-dense / low-gold — not this.
    """
    if _own_wood_is_grey_stone_mass(c):
        return False
    gold = int(c.get("gold_n") or 0)
    dens = float(c.get("wood_dens") or 0)
    if gold < 24 or dens >= 0.16:
        return False
    # Plate centroid can sit a few px above the wood (live split); 60px is tight.
    if (
        int(c.get("wood_w") or 0) < 36
        or int(c.get("wood_h") or 0) < 54
        or int(c.get("warm_n") or 0) < 140
        or dens < 0.06
    ):
        return False
    warm = int(c.get("warm_n") or 0)
    # Live lamps 68–116 + wood ≈260. Leak fixtures are gold≈36–40 / warm≈180
    # on a fragment plate (n≈620 w≈58).
    if gold >= 48 and warm >= 220:
        return True
    return int(c.get("w") or 0) >= 62 or int(c.get("n") or 0) >= 720


def _is_own_longhouse_hit(c: dict[str, Any]) -> bool:
    """Sparse live longhouse or gold-lamp split plate — not grey stone."""
    return _sparse_own_longhouse(c) or _gold_lamp_sparse_longhouse(c)


def _dark_vengir_city_plate(c: dict[str, Any]) -> bool:
    """c14_23 salvage. Frost FPs are n≈500 w≈56; #47's 900/70 missed live plates."""
    return int(c.get("w") or 0) >= 60 or int(c.get("n") or 0) >= 640


def _own_miss_is_dark_vengir(
    rec: dict[str, Any],
    *,
    sampled_tribe: str,
    col_tribe: str,
    frost_plate: bool,
    marks: bool,
    wine_n: int = 0,
) -> bool:
    """Wine-wall Vengir that a Bardur wood vote would drop from both lists.

    Live after #43/#44: c14_23 (frost+star, gold_n<24) vanished because
    sampled_tribe *and* col_tribe were bardur (wine-grey stone). #44's
    wine_n>=8 + g<=72 still missed live veins (g≈74–86). Count lighter
    wine-grey. Real Orkork/Bufla (sparse gold-lamp longhouses, even when the
    gold-star hole splits plate n/w below 900/70) stay own. Mid-dense low-gold
    at any width is grey stone, not Bufla (live T46 after #46: c12_16 vanished
    and enemy stayed 4).
    """
    if not frost_plate or not marks:
        return False
    gold = int(rec.get("gold_n") or 0)
    if gold >= 24:
        return False
    dens = float(rec.get("wood_dens") or 0)
    if dens >= 0.34:
        return False
    # Sparse gold-lamp / city-sized longhouses stay own even when wine-shadow ticks.
    if _is_own_longhouse_hit(rec) and dens < 0.16:
        return False
    if sampled_tribe == "vengir" or col_tribe == "vengir":
        return True
    # Live T46 after #44: plurality Bardur, lighter wine-grey (g>72) still
    # Vengir walls. 8px was too strict — live c14_23 was 0 hits.
    if int(wine_n or 0) >= WINE_SALVAGE_MIN:
        return True
    # After #45/#47 wine_n==0 on the live frame. Frost+star that is not a
    # longhouse is still the missing 5th Vengir. Frost FPs are n≈500 w≈56;
    # live c14_23 can sit below #47's n≥900/w≥70.
    return _dark_vengir_city_plate(rec)


def _own_plate_is_split_fragment(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Gold-star hole of one Bardur plate, not two 1-hex neighbor longhouses.

    Live T46: Orkork c17_12 + phantom c15_12 ~72px, same plate_y, one wide
    plate and one fragment. Neighbor cities both have full plates (~80px).
    """
    try:
        ay = int(a.get("plate_y") if a.get("plate_y") is not None else a["y"])
        by = int(b.get("plate_y") if b.get("plate_y") is not None else b["y"])
    except (KeyError, TypeError, ValueError):
        return False
    if abs(ay - by) > 16:
        return False
    wa, wb = int(a.get("w") or 0), int(b.get("w") or 0)
    na, nb = int(a.get("n") or 0), int(b.get("n") or 0)
    wide = max(wa, wb) >= 70 and 0 < min(wa, wb) <= 62
    heavy = max(na, nb) >= 900 and 0 < min(na, nb) <= 750
    return wide or heavy


def _own_wood_looks_like_building(c: dict[str, Any]) -> bool:
    """Drop unit-on-snow / mountain specks; keep live longhouses with sparse wood.

    #32 required dens≥0.45×tall (and warm_n≥500). Live Bufla/Orkork failed that
    (cities_own 1→0). #33 short∧sparse (wh<34 && dens<0.38) still lets through
    tall-empty frost (live T46 c17_14: wh=48 dens=0.021 warm=67). Keep Bufla/
    Orkork (wh≈69–72 dens≈0.08–0.09 warm≈260, gold_n 68–116). Never restore
    warm_n>=500. Tall-dense-low-gold (c12_16 dens≈0.28 gold_n≈5) is grey stone.
    Frost/wood FPs (live c15_15 / c15_12 / c15_24) have a gold star or fruit
    on the plate but almost no building lamps, and are not dense synthetic
    longhouses (dens≳0.34, gold_n=0) or sparse live longhouses (Orkork
    c16_24: ww≳36 wh≳60 warm≳140 dens≳0.06, plate w≳70 even when gold_n<24).
    After /recruit or /move-to, leaked gold (gold_n≥24) on the same fragment
    (ww≈30 dens≈0.08, and live T46 after #46: c15_12 with ww inflated ≳36)
    used to pass this gate and sticky-grow own 2→3. A dense unit-sized blob
    (ww≈31 dens≈0.46 gold_n≥24) is not a synthetic longhouse. Gold-lamp
    longhouses keep without a city-sized plate; leaked FPs (gold≈40 warm≈180)
    do not. Split-plate longhouses merge earlier / stay sticky. Dict fixtures
    omit wood_* — treat those as already-vetted cities.
    """
    if c.get("wood_h") is None and c.get("wood_dens") is None:
        return True
    wh = int(c.get("wood_h") or 0)
    dens = float(c.get("wood_dens") or 0)
    warm = int(c.get("warm_n") or 0)
    # Tall frost streak: bbox can be 48px from scattered specks, almost empty.
    # Coupled to low warm_n so a dim longhouse at dens≈0.08 still counts.
    if dens < 0.05 and warm < 140:
        return False
    # Unit frost ~31×0.31; mountain speck ~23×0.17; scaled longhouse ~40×0.55.
    if wh < 34 and dens < 0.38:
        return False
    if wh < 26:
        return False
    if _own_wood_is_grey_stone_mass(c):
        return False
    if c.get("gold_n") is not None:
        gold = int(c.get("gold_n") or 0)
        ww = int(c.get("wood_w") or 0)
        nw = int(c.get("n") or 0)
        bw = int(c.get("w") or 0)
        # Sparse live longhouse keeps via city-sized plate *or* roof lamps on
        # the wood footprint. #47's dens<0.34 → n≥900/w≥70 dropped Bufla/Orkork
        # (gold-star hole splits the plate). Compact / synthetic longhouses keep
        # via dens≳0.34 even when gold_n=0 (scaled 1280 `_draw_bardur_city`
        # n≈700 w≈55). A gold-leaked unit blob (ww≈31 dens≈0.46 gold_n≥24) is
        # not that. After #46, gold≥24 plus a ww≳36 footprint alone let c15_12
        # become a 3rd own — require lamps+wood (gold≥48 warm≥220) or a near
        # plate, not fragment n≈620 w≈58.
        if dens < 0.34:
            if not _is_own_longhouse_hit(c):
                return False
        elif gold >= 24 and ww < 36:
            return False
        elif gold >= 24 and dens >= 0.34 and nw < 900 and bw < 70:
            return False
        # Narrow unit column (even a dense leather blob) is not a longhouse.
        if ww < 28 and gold < 24:
            return False
        if ww < 28 and dens < 0.34 and gold < 48:
            return False
    return True


def _is_real_own_hit(c: dict[str, Any]) -> bool:
    """Ufla-class Bardur: frost plate + gold star (or roof lamps). Bare frost is a ghost.

    Live T46 Game Stats: Bardur 2 cities (Bufla + Orkork). #32's warm_n>=500
    floor left cities_own 1→0. Close-plate merge is capped at 48px so ~80px
    neighbors stay two. Short/sparse, tall-empty, tall-dense-low-gold
    (c12_16 dens≈0.28 gold_n≈5 ww≈32–47 / c17_14), *and* frost/wood plates
    with a gold star but no city-sized longhouse (c15_15 / c15_12 / c15_24),
    including after /recruit+/move-to gold leak (gold_n≥24 on ww≈30 or a
    ww≳36 inflated fragment with gold≈40/warm≈180), still drop from own.
    Gold-lamp Bufla/Orkork stay even when the plate is below n≥900/w≥70.
    """
    if c.get("owner") != "own":
        return False
    if _own_hit_in_top_chrome(c):
        return False
    ev = c.get("evidence") or []
    if "grey_stone" in ev or "dark_vengir" in ev:
        return False
    if c.get("warm_n") is not None and int(c.get("warm_n") or 0) < 30:
        return False
    name = str(c.get("name") or "").strip()
    ev = c.get("evidence") or []
    if len(name) >= 4:
        return _own_wood_looks_like_building(c)
    # Gold star / roof lamps. A full lettered Bardur plate whose star clustered
    # onto a neighbor still counts (adjacent 1-hex cities). Tiny frost splits do not.
    if "gold_star" in ev or "gold_lamps" in ev:
        return _own_wood_looks_like_building(c)
    if (
        "frost_plate" in ev
        and "bardur_wood" in ev
        and int(c.get("n") or 0) >= 900
        and int(c.get("w") or 0) >= 52
    ):
        return _own_wood_looks_like_building(c)
    return False


def _keep_sticky_city(c: dict[str, Any]) -> bool:
    """Unnamed frost FPs must not accumulate across observes (2→4→7).

    Own ghosts stuck the same way (T46: 5–7 vs 2 Bardur on screen) — those
    still drop because they fail `_is_real_own_hit`. A real longhouse
    (Bufla / Orkork, including c16_24) stays sticky across missed frames
    so cities_own does not flicker 2→1. Do not cap misses: phantoms never
    pass the hit gate, and a 1-miss cap under-counted live Orkork.
    """
    if c.get("owner") == "own":
        return _is_real_own_hit(c)
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


def _drop_own_phantoms(cities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop unnamed / HUD-fog / unit-sized / tall-empty own cities (c15_24, c17_14).

    Tall-dense-low-gold c12_16 is Vengir grey stone — keep it as enemy so
    cities_enemy matches Game Stats (4→5) instead of deleting the city.
    Wine-wall frost+star (c14_23, dark_vengir) that a Bardur wood vote stole
    also flips to enemy instead of vanishing.
    """
    out: list[dict[str, Any]] = []
    for c in cities:
        if c.get("owner") == "own" and not _is_real_own_hit(c):
            if _own_wood_is_grey_stone_mass(c):
                d = dict(c)
                d["owner"] = "enemy"
                d["tribe"] = "vengir"
                ev = [x for x in (d.get("evidence") or []) if x != "bardur_wood"]
                if "grey_stone" not in ev:
                    ev.append("grey_stone")
                d["evidence"] = ev
                out.append(d)
            elif "dark_vengir" in (c.get("evidence") or []) or _own_miss_is_dark_vengir(
                c,
                sampled_tribe=str(c.get("tribe") or "bardur"),
                col_tribe="bardur",
                frost_plate="frost_plate" in (c.get("evidence") or []),
                marks=(
                    "gold_star" in (c.get("evidence") or [])
                    or "gold_lamps" in (c.get("evidence") or [])
                ),
                wine_n=int(c.get("wine_n") or 0),
            ):
                d = dict(c)
                d["owner"] = "enemy"
                d["tribe"] = "vengir"
                ev = [x for x in (d.get("evidence") or []) if x != "bardur_wood"]
                if "dark_vengir" not in ev:
                    ev.append("dark_vengir")
                d["evidence"] = ev
                out.append(d)
            continue
        out.append(c)
    return out


def _drop_split_plate_own(cities: list[dict[str, Any]], dist: int = 56) -> list[dict[str, Any]]:
    """Grey-stone half of a Disrof nameplate must not become cities_own.

    Live T37 cover_roof: the left frost fragment (w≈35, gold_star only) sat
    ~46px from Disrof and hashed a different tile, so cross-owner merge
    refused. A real Bardur neighbor is a full plate (w≳80).
    """
    disrofs = [c for c in cities if _is_disrof_hit(c) and c.get("owner") == "enemy"]
    if not disrofs:
        return cities
    out: list[dict[str, Any]] = []
    for c in cities:
        if (
            c.get("owner") == "own"
            and int(c.get("w") or 99) <= 44
            and int(c.get("roof_n") or 0) < 8
            and int(c.get("gold_n") or 0) < 8
        ):
            cx, cy = int(c["x"]), int(c["y"])
            split = False
            for d in disrofs:
                if (cx - int(d["x"])) ** 2 + (cy - int(d["y"])) ** 2 <= dist * dist:
                    split = True
                    break
            if split:
                continue
        out.append(c)
    return out


def _drop_grey_splits_near_own(cities: list[dict[str, Any]], dist: int = 56) -> list[dict[str, Any]]:
    """Gold-star halves of a Bardur longhouse are not a 6th Vengir.

    Mid-dense low-gold (any width) is grey stone so live ww≈47 c12_16 stays
    the 5th enemy. The same signature on a skinny fragment ~40px from Bufla
    or Orkork is the gold-star hole, not a city.
    """
    owns = [c for c in cities if c.get("owner") == "own" and _is_real_own_hit(c)]
    if not owns:
        return cities
    out: list[dict[str, Any]] = []
    for c in cities:
        ev = c.get("evidence") or []
        if (
            c.get("owner") == "enemy"
            and "grey_stone" in ev
            and "magenta_roof" not in ev
            and "gold_lamps" not in ev
        ):
            if any(_city_sep2(c, o) <= dist * dist for o in owns):
                continue
        out.append(c)
    return out


def _enemy_gold_leak_phantom(c: dict[str, Any]) -> bool:
    """Recruit/move gold leak on a frost fragment is not a 6th–8th Vengir.

    Live T46 after #47: cities_enemy bloated 4→7–8 with c15_14 / c16_12 / c8_23
    (gold_lamps, fragment plate). Real Disrof has magenta, huge gold, or a
    city-sized plate. Grey-stone c12_16 and dark_vengir c14_23 stay.
    """
    if c.get("owner") != "enemy" or c.get("tribe") != "vengir":
        return False
    if c.get("name"):
        return False
    ev = c.get("evidence") or []
    if "magenta_roof" in ev or "grey_stone" in ev or "dark_vengir" in ev:
        return False
    gold = int(c.get("gold_n") or 0)
    # Plate-window Disrof fragments often have gold_n=0 (lamps are on the frost).
    # Recruit leak paints building gold (live c15_12 gold≈36–40).
    if gold < 24 or gold >= 280:
        return False
    if int(c.get("roof_n") or 0) >= 40:
        return False
    if int(c.get("n") or 0) >= 900 or int(c.get("w") or 0) >= 70:
        return False
    return "gold_lamps" in ev or "gold_star" in ev


def _drop_enemy_gold_leak_phantoms(cities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in cities if not _enemy_gold_leak_phantom(c)]


def _drop_wine_shadow_enemy_near_own(
    cities: list[dict[str, Any]], dist: int = 56
) -> list[dict[str, Any]]:
    """Wine-shadow frost fragment next to Bufla/Orkork is not a 6th Vengir.

    Live after #47: Orkork leaked as magenta_roof+gold_lamps and a nearby
    split (c15_23) stayed enemy. Real Disrof / c14_23 / grey-stone are
    city-sized or tagged dark_vengir/grey_stone.
    """
    owns = [c for c in cities if c.get("owner") == "own" and _is_real_own_hit(c)]
    if not owns:
        return cities
    out: list[dict[str, Any]] = []
    for c in cities:
        if (
            c.get("owner") == "enemy"
            and int(c.get("n") or 0) <= 400
            and int(c.get("w") or 99) <= 56
            and int(c.get("gold_n") or 0) < 280
            and int(c.get("roof_n") or 0) < 40
            and "grey_stone" not in (c.get("evidence") or [])
            and "dark_vengir" not in (c.get("evidence") or [])
        ):
            if any(_city_sep2(c, o) <= dist * dist for o in owns):
                continue
        out.append(c)
    return out


def _finalize_cities(cities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every real enemy city. A 12-own map must not empty cities_enemy."""
    cities = _cap_vengir(
        _drop_wine_shadow_enemy_near_own(
            _drop_grey_splits_near_own(
                _drop_enemy_gold_leak_phantoms(
                    _dedupe_cities(
                        _drop_split_plate_own(_drop_own_phantoms(list(cities)))
                    )
                )
            )
        )
    )
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
                oa, ob = c.get("owner"), o.get("owner")
                if oa == "own" and ob == "own":
                    d2 = _city_sep2(c, o)
                else:
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
