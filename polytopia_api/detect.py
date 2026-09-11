"""Vectorized pixel detection for Polytopia screenshots.

Colors are from the 1920x1200 Cloud Agent play sessions. Water is cyan;
move highlights are a more violet brand-blue. Never treat cyan as a move.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from . import coords

# Map area in design space — skip HUD and bottom chrome.
# Left 400px was cropping NW cities (Disrof) on 1280×800.
MAP_Y0, MAP_Y1 = 90, 1040
MAP_X0, MAP_X1 = 80, 1860

# Brand confirm blue (0, 153, 255)
DOIT_LO = np.array([0, 140, 240], dtype=np.int16)
DOIT_HI = np.array([25, 175, 255], dtype=np.int16)

# TRAIN modal blue (0, 122, 204)
TRAIN_LO = np.array([0, 100, 185], dtype=np.int16)
TRAIN_HI = np.array([25, 140, 230], dtype=np.int16)

# Unit move highlight (empirically 146, 204, 255) — NOT water
MOVE_LO = np.array([110, 185, 235], dtype=np.int16)
MOVE_HI = np.array([175, 220, 255], dtype=np.int16)

# Fruit cubes: royal blue, small clusters
FRUIT_LO = np.array([0, 70, 190], dtype=np.int16)
FRUIT_HI = np.array([110, 180, 255], dtype=np.int16)


def as_rgb(im: Image.Image) -> np.ndarray:
    arr = np.asarray(im.convert("RGB"))
    return arr.astype(np.int16)


def _cluster(ys: np.ndarray, xs: np.ndarray, radius: int = 28, min_size: int = 8) -> list[tuple[int, int, int]]:
    """Greedy clusters. Returns (count, cx, cy) largest-first."""
    if xs.size == 0:
        return []
    pts = np.stack([xs, ys], axis=1)
    used = np.zeros(len(pts), dtype=bool)
    out: list[tuple[int, int, int]] = []
    for i in range(len(pts)):
        if used[i]:
            continue
        dx = pts[:, 0] - pts[i, 0]
        dy = pts[:, 1] - pts[i, 1]
        near = (~used) & (dx * dx + dy * dy <= radius * radius)
        n = int(near.sum())
        if n < min_size:
            used[near] = True
            continue
        used[near] = True
        c = pts[near].mean(axis=0)
        out.append((n, int(c[0]), int(c[1])))
    out.sort(reverse=True)
    return out


def merge_clusters(clusters: list[tuple[int, int, int]], dist: int) -> list[tuple[int, int, int]]:
    """Collapse greedy fragments from one blob into a single hit."""
    out: list[tuple[int, int, int]] = []
    for n, cx, cy in clusters:
        hit = False
        for i, (n2, x2, y2) in enumerate(out):
            if (cx - x2) ** 2 + (cy - y2) ** 2 <= dist * dist:
                tot = n + n2
                out[i] = (
                    tot,
                    int(round((cx * n + x2 * n2) / tot)),
                    int(round((cy * n + y2 * n2) / tot)),
                )
                hit = True
                break
        if not hit:
            out.append((n, cx, cy))
    out.sort(reverse=True)
    return out


def _frame_xy(arr: np.ndarray, x: int, y: int) -> tuple[int, int]:
    h, w = arr.shape[:2]
    return (
        int(round(x * w / coords.BASE_W)),
        int(round(y * h / coords.BASE_H)),
    )


def _map_bounds(arr: np.ndarray) -> tuple[int, int, int, int]:
    x0, y0 = _frame_xy(arr, MAP_X0, MAP_Y0)
    x1, y1 = _frame_xy(arr, MAP_X1, MAP_Y1)
    h, w = arr.shape[:2]
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return y0, y1, x0, x1


def _mask_in(
    arr: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[np.ndarray, np.ndarray]:
    ax0, ay0 = _frame_xy(arr, x0, y0)
    ax1, ay1 = _frame_xy(arr, x1, y1)
    h, w = arr.shape[:2]
    ay0, ay1 = max(0, min(ay0, h)), max(0, min(ay1, h))
    ax0, ax1 = max(0, min(ax0, w)), max(0, min(ax1, w))
    if ay1 <= ay0 or ax1 <= ax0:
        return np.array([], dtype=int), np.array([], dtype=int)
    region = arr[ay0:ay1, ax0:ax1]
    m = np.all((region >= lo) & (region <= hi), axis=2)
    ys, xs = np.where(m)
    return ys + ay0, xs + ax0


def is_water_rgb(r: int, g: int, b: int) -> bool:
    """Cyan water / ice. Move highlights fail this (G stays ≤220 and B-G is larger)."""
    if b < 180 or g < 170:
        return False
    if g >= 225 and b >= 225:
        return True
    if r < 100 and g >= 190 and b >= 230:
        return True
    # cyan: G close to B (water 111,207,247 has |G-B|≈40)
    if b - r >= 70 and abs(g - b) <= 45 and g >= 185:
        return True
    return False


def is_move_rgb(r: int, g: int, b: int) -> bool:
    if not (110 <= r <= 175 and 185 <= g <= 220 and b >= 235):
        return False
    if b - g < 18:
        return False
    if g - r < 20:
        return False
    return not is_water_rgb(r, g, b)


def _cluster_params(arr: np.ndarray, radius: int, min_size: int) -> tuple[int, int]:
    h, w = arr.shape[:2]
    s = min(w / coords.BASE_W, h / coords.BASE_H)
    area = (w * h) / (coords.BASE_W * coords.BASE_H)
    return max(8, int(round(radius * s))), max(3, int(round(min_size * area)))


def _color_buttons(
    arr: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    kind: str,
    y0f: float,
    y1f: float,
    x0f: float,
    x1f: float,
    radius: int,
    min_size: int,
) -> list[dict[str, Any]]:
    """Large UI pills only. Map water / fruit used to yield dozens of blobs."""
    h, w = arr.shape[:2]
    y0, y1 = int(h * y0f), min(int(h * y1f), int(h * 0.84))
    x0, x1 = int(w * x0f), int(w * x1f)
    region = arr[y0:y1, x0:x1]
    if region.size == 0:
        return []
    m = np.all((region >= lo) & (region <= hi), axis=2)
    ys, xs = np.where(m)
    rad, mn = _cluster_params(arr, radius, min_size)
    clusters = merge_clusters(_cluster(ys + y0, xs + x0, radius=rad, min_size=mn), dist=max(24, rad))
    area = (h * w) / (coords.BASE_W * coords.BASE_H)
    min_n = max(70, mn)
    max_n = max(5000, int(round(12000 * area)))
    out: list[dict[str, Any]] = []
    for n, cx, cy in clusters:
        if n < min_n or n > max_n:
            continue
        if coords.in_dock_zone(cx, cy):
            continue
        out.append({"kind": kind, "x": cx, "y": cy, "n": n})
    out.sort(key=lambda b: -int(b["n"]))
    return out[:4]


def find_doit_buttons(arr: np.ndarray) -> list[dict[str, Any]]:
    """Capture / DO IT — live-frame confirm pill, not 2/3 of (1117, 681)."""
    return _color_buttons(arr, DOIT_LO, DOIT_HI, "do_it", 0.34, 0.78, 0.30, 0.90, 32, 90)


def find_capture_panel_buttons(arr: np.ndarray) -> list[dict[str, Any]]:
    """Capture lives in UNIT_CROP (bottom-left panel), not on the map."""
    h, w = arr.shape[:2]
    y0, y1 = int(h * 0.78), h
    x0, x1 = 0, int(w * 0.52)
    region = arr[y0:y1, x0:x1]
    if region.size == 0:
        return []
    m = np.all((region >= DOIT_LO) & (region <= DOIT_HI), axis=2)
    ys, xs = np.where(m)
    rad, mn = _cluster_params(arr, 28, 40)
    clusters = merge_clusters(_cluster(ys + y0, xs + x0, radius=rad, min_size=mn), dist=max(24, rad))
    out: list[dict[str, Any]] = []
    for n, cx, cy in clusters:
        if n < max(40, mn) or coords.in_dock_zone(cx, cy):
            continue
        out.append({"kind": "capture", "x": cx, "y": cy, "n": n})
    out.sort(key=lambda b: -int(b["n"]))
    return out[:2]


def find_train_buttons(arr: np.ndarray) -> list[dict[str, Any]]:
    """TRAIN is a live-frame UI blob — not 2/3 of (1110, 790), which is the map at 1280."""
    return _color_buttons(arr, TRAIN_LO, TRAIN_HI, "train", 0.38, 0.82, 0.28, 0.92, 28, 70)


def find_move_marks(arr: np.ndarray) -> list[dict[str, Any]]:
    y0, y1, x0, x1 = _map_bounds(arr)
    region = arr[y0:y1, x0:x1]
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    m = (
        (r >= 110) & (r <= 175)
        & (g >= 185) & (g <= 220)
        & (b >= 235)
        & ((b - g) >= 18)
        & ((g - r) >= 20)
        & ~((g >= 225) & (b >= 225))
        & ~((r < 100) & (g >= 190))
    )
    ys, xs = np.where(m)
    rad, mn = _cluster_params(arr, 24, 10)
    clusters = _cluster(ys + y0, xs + x0, radius=rad, min_size=mn)
    marks = []
    for n, cx, cy in clusters:
        pr, pg, pb = (int(x) for x in arr[cy, cx])
        if is_water_rgb(pr, pg, pb):
            continue
        marks.append({"kind": "move", "x": cx, "y": cy, "n": n, "rgb": [pr, pg, pb]})
    return marks


def is_attack_rgb(r: int, g: int, b: int) -> bool:
    """Red attack hex — not lime HP, not Oumaji yellow, not water."""
    if r < 165 or g > 115 or b > 120:
        return False
    if r < g + 50 or r < b + 40:
        return False
    return not is_water_rgb(r, g, b)


def find_attack_marks(arr: np.ndarray) -> list[dict[str, Any]]:
    """Red hexes after selecting a unit that can attack."""
    y0, y1, x0, x1 = _map_bounds(arr)
    region = arr[y0:y1, x0:x1]
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    m = (
        (r >= 165)
        & (g <= 115)
        & (b <= 120)
        & ((r - g) >= 50)
        & ((r - b) >= 40)
        & ~((g >= 155) & (g >= r + 40))
    )
    ys, xs = np.where(m)
    rad, mn = _cluster_params(arr, 22, 12)
    clusters = _cluster(ys + y0, xs + x0, radius=rad, min_size=mn)
    h, w = arr.shape[:2]
    max_n = max(50, int(round(280 * (h * w) / (coords.BASE_W * coords.BASE_H))))
    marks = []
    for n, cx, cy in clusters:
        if n > max_n:
            continue
        pr, pg, pb = (int(v) for v in arr[cy, cx])
        if not is_attack_rgb(pr, pg, pb):
            continue
        marks.append({"kind": "attack", "x": cx, "y": cy, "n": n, "rgb": [pr, pg, pb]})
    return marks


def find_fruit(arr: np.ndarray) -> list[dict[str, Any]]:
    y0, y1, x0, x1 = _map_bounds(arr)
    region = arr[y0:y1, x0:x1]
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    m = (
        (r <= 110)
        & (g >= 70) & (g <= 180)
        & (b >= 190)
        & ((b - g) >= 40)
        & (r < 120)
        & ~((g >= 190) & (b >= 230))
    )
    ys, xs = np.where(m)
    rad, mn = _cluster_params(arr, 22, 6)
    clusters = _cluster(ys + y0, xs + x0, radius=rad, min_size=mn)
    fruits = []
    max_n = max(40, int(round(140 * (arr.shape[0] * arr.shape[1]) / (coords.BASE_W * coords.BASE_H))))
    for n, cx, cy in clusters:
        if n > max_n:
            continue
        pr, pg, pb = (int(x) for x in arr[cy, cx])
        if is_water_rgb(pr, pg, pb):
            continue
        fruits.append({"kind": "fruit", "x": cx, "y": cy, "n": n, "rgb": [pr, pg, pb]})
    return fruits


def sample(arr: np.ndarray, x: int, y: int) -> list[int]:
    h, w = arr.shape[:2]
    x = min(max(0, x), w - 1)
    y = min(max(0, y), h - 1)
    return [int(v) for v in arr[y, x]]


def overlay_flags(arr: np.ndarray) -> dict[str, Any]:
    h, w = arr.shape[:2]
    coords.set_frame(w, h)
    doit_px = sample(arr, *coords.DO_IT)
    train_px = sample(arr, *coords.TRAIN)
    back_px = sample(arr, *coords.BACK)
    r, g, b = doit_px
    doit = r <= 25 and 140 <= g <= 175 and b >= 240
    r, g, b = train_px
    train = r <= 25 and 100 <= g <= 140 and 180 <= b <= 230
    br, bg, bb = back_px
    # In-game corner is black; overlays draw a white back circle
    back_lit = br + bg + bb >= 400
    blobs = find_doit_buttons(arr)
    capture_blobs = find_capture_panel_buttons(arr)
    train_blobs = find_train_buttons(arr)
    return {
        "do_it_pixel": doit,
        "train_pixel": train,
        "back_lit": back_lit,
        "do_it_rgb": doit_px,
        "train_rgb": train_px,
        "back_rgb": back_px,
        "do_it_blobs": blobs,
        "capture_blobs": capture_blobs,
        "train_blobs": train_blobs,
        "attack_marks": find_attack_marks(arr)[:16],
    }


def observe_pixels(im: Image.Image) -> dict[str, Any]:
    arr = as_rgb(im)
    flags = overlay_flags(arr)
    return {
        "overlay": flags,
        "move_marks": find_move_marks(arr)[:12],
        "attack_marks": flags.get("attack_marks") or find_attack_marks(arr)[:16],
        "fruit": find_fruit(arr)[:10],
    }
