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

    Vengir cities are dark-grey stone + purple roofs + gold window lights.
    Those lamps match Oumaji sand pixel-for-pixel; context wins: any purple
    roof, or gold-on-dark-stone, is Vengir — not a desert city.
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
    # Purple roof is decisive (Disrof magenta).
    if vengir_n >= 3 or (total and vengir_n >= 0.08 * total):
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


def _vengir_roof_pixels(arr: np.ndarray, cx: int, cy: int, bw: int, up: int) -> int:
    """Magenta/wine roof can sit well above the frost plate (outside radius=14)."""
    h, w = arr.shape[:2]
    x0 = max(0, cx - max(20, bw // 2 + 10))
    x1 = min(w, cx + max(20, bw // 2 + 10) + 1)
    y0 = max(0, cy - up * 2 - 16)
    y1 = max(y0 + 1, min(h, cy - 4))
    crop = arr[y0:y1, x0:x1]
    if crop.size < 20:
        return 0
    r = crop[:, :, 0].astype(np.int32)
    g = crop[:, :, 1].astype(np.int32)
    b = crop[:, :, 2].astype(np.int32)
    purple = (r + b) / 2 - g
    mag = (
        (g <= 100)
        & (np.maximum(r, b) >= 48)
        & (np.maximum(np.maximum(r, g), b) <= 185)
        & (np.minimum(r, b) >= 40)
        & (purple >= 12)
        & (np.abs(r - b) <= 55)
    )
    return int(mag.sum())


def _gold_window_pixels(arr: np.ndarray, cx: int, cy: int, bw: int, up: int) -> int:
    """Gold lamps on the building (above the plate), not the nameplate star."""
    h, w = arr.shape[:2]
    x0 = max(0, cx - max(16, bw // 2))
    x1 = min(w, cx + max(16, bw // 2) + 1)
    y0 = max(0, cy - up * 2 - 8)
    y1 = max(y0 + 1, min(h, cy - 6))
    crop = arr[y0:y1, x0:x1]
    if crop.size < 20:
        return 0
    r = crop[:, :, 0].astype(np.int32)
    g = crop[:, :, 1].astype(np.int32)
    b = crop[:, :, 2].astype(np.int32)
    gold = (r >= 180) & (g >= 140) & (g <= 220) & (b <= 110) & (r >= b + 60)
    return int(gold.sum())


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
        sampled_tribe, rgb = sample_patch_tribe(
            arr, cx, max(0, cy - up - max(4, bh // 2)), radius=14
        )
        tribe = sampled_tribe
        if is_water_rgb(*rgb):
            continue
        roof_n = _vengir_roof_pixels(arr, cx, cy, bw, up)
        gold_n = _gold_window_pixels(arr, cx, cy, bw, up)
        frost_plate = _plate_is_frost(arr, cx, cy, bw, bh)
        strong_vengir = roof_n >= 8 or gold_n >= 8
        if roof_n >= 8:
            tribe = "vengir"
        # Gold lamps on a frost nameplate are Disrof, not a desert city.
        # Bare frost + sand-colored pixels (no roof, no lamps) are FPs.
        if tribe == "oumaji" and frost_plate and strong_vengir:
            tribe = "vengir"
        if tribe == "vengir" and sampled_tribe not in {"vengir"} and not strong_vengir:
            continue
        # Skip plate OCR for known tribes — names are optional and tesseract
        # on every frost cluster made /attack hang past 12s.
        name = ""
        if tribe == "unknown":
            name = plate_name(arr, cx, cy, bw, bh)
        if tribe == "unknown" and not name:
            continue
        own = tribe == _own_tribe()
        if tribe == "unknown":
            own = False
        if not own and tribe == "oumaji" and not frost_plate:
            # Bright sand / yellow UI, not a Moonrise frost city.
            continue
        conf = 0.72 if own else 0.62
        if tribe == "imperius":
            conf = 0.55
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
        if roof_n >= 8:
            evidence.append("magenta_roof")
        if gold_n >= 8:
            evidence.append("gold_lamps")
        if marks:
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
        cities.append(
            {
                "id": combat.city_id_for_tile(gx, gy),
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
            }
        )
    cities.sort(
        key=lambda c: (
            0 if c.get("owner") == "enemy" else 1,
            -float(c["confidence"]),
            -int(c["n"]),
        )
    )
    return _cap_vengir(_dedupe_cities(cities))[:12]


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
        return 0
    tribes = {a.get("tribe"), b.get("tribe")}
    if tribes == {"vengir"}:
        return max(dist, 80)
    return dist


def _plates_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    try:
        ax, aw = int(a["x"]), int(a.get("w") or 40)
        bx, bw = int(b["x"]), int(b.get("w") or 40)
        ay = int(a.get("plate_y") or a["y"])
        by = int(b.get("plate_y") or b["y"])
    except (KeyError, TypeError, ValueError):
        return False
    if abs(ay - by) > 12:
        return False
    return abs(ax - bx) <= (aw + bw) / 2 + 8


def _prefer_city(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    # Same nameplate: gold/magenta Disrof wins over the grey-stone half.
    # Distinct hexes with different owners never reach here unless plates overlap.
    def _disrof(c: dict[str, Any]) -> bool:
        ev = c.get("evidence") or []
        return c.get("tribe") == "vengir" and ("magenta_roof" in ev or "gold_lamps" in ev)

    if _disrof(a) and not _disrof(b):
        win = dict(a)
        if b.get("name") and not win.get("name"):
            win["name"] = b["name"]
        return win
    if _disrof(b) and not _disrof(a):
        win = dict(b)
        if a.get("name") and not win.get("name"):
            win["name"] = a["name"]
        return win
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
    ev = c.get("evidence") or []
    if "magenta_roof" in ev or "gold_lamps" in ev:
        return True
    return False


def session_cities(cities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """After stabilize(keep_missing): drop unseen frost FPs, recap unnamed Vengir."""
    kept: list[dict[str, Any]] = []
    for c in cities or []:
        if c.get("seen") is False and not _keep_sticky_city(c):
            continue
        kept.append(c)
    return _cap_vengir(_dedupe_cities(kept))[:12]


def _cap_vengir(cities: list[dict[str, Any]], unnamed_limit: int = 2) -> list[dict[str, Any]]:
    """Keep real Disrof-class hits; drop a swarm of unnamed frost fragments."""
    named: list[dict[str, Any]] = []
    unnamed: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for c in cities:
        if c.get("tribe") != "vengir":
            others.append(c)
        elif c.get("name"):
            named.append(c)
        else:
            unnamed.append(c)
    def _score(c: dict[str, Any]) -> tuple:
        ev = c.get("evidence") or []
        return (
            1 if "magenta_roof" in ev else 0,
            1 if "gold_lamps" in ev else 0,
            float(c.get("confidence") or 0),
            int(c.get("n") or 0),
        )
    unnamed.sort(key=_score, reverse=True)
    return others + named + unnamed[:unnamed_limit]


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
