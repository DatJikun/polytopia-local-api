"""Vectorized pixel detection for Polytopia screenshots.

Colors are from the 1920x1200 Cloud Agent play sessions. Water is cyan;
move highlights are a more violet brand-blue. Never treat cyan as a move.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

# Map area — skip HUD and bottom chrome
MAP_Y0, MAP_Y1 = 90, 1040
MAP_X0, MAP_X1 = 400, 1750

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


def _mask_in(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray, y0=MAP_Y0, y1=MAP_Y1, x0=MAP_X0, x1=MAP_X1) -> tuple[np.ndarray, np.ndarray]:
    region = arr[y0:y1, x0:x1]
    m = np.all((region >= lo) & (region <= hi), axis=2)
    ys, xs = np.where(m)
    return ys + y0, xs + x0


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


def find_doit_buttons(arr: np.ndarray) -> list[dict[str, Any]]:
    """Large brand-blue blobs in the confirm band."""
    ys, xs = _mask_in(arr, DOIT_LO, DOIT_HI, y0=560, y1=920, x0=800, x1=1450)
    clusters = _cluster(ys, xs, radius=40, min_size=80)
    out = []
    for n, cx, cy in clusters:
        out.append({"kind": "do_it", "x": cx, "y": cy, "n": n})
    return out


def find_train_buttons(arr: np.ndarray) -> list[dict[str, Any]]:
    ys, xs = _mask_in(arr, TRAIN_LO, TRAIN_HI, y0=700, y1=920, x0=900, x1=1350)
    clusters = _cluster(ys, xs, radius=35, min_size=60)
    return [{"kind": "train", "x": cx, "y": cy, "n": n} for n, cx, cy in clusters]


def find_move_marks(arr: np.ndarray) -> list[dict[str, Any]]:
    region = arr[MAP_Y0:MAP_Y1, MAP_X0:MAP_X1]
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
    clusters = _cluster(ys + MAP_Y0, xs + MAP_X0, radius=24, min_size=10)
    marks = []
    for n, cx, cy in clusters:
        pr, pg, pb = (int(x) for x in arr[cy, cx])
        if is_water_rgb(pr, pg, pb):
            continue
        marks.append({"kind": "move", "x": cx, "y": cy, "n": n, "rgb": [pr, pg, pb]})
    return marks


def find_fruit(arr: np.ndarray) -> list[dict[str, Any]]:
    region = arr[MAP_Y0:MAP_Y1, MAP_X0:MAP_X1]
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    m = (
        (r <= 110)
        & (g >= 70) & (g <= 180)
        & (b >= 190)
        & ((b - g) >= 40)
        & (r < 120)
        & ~((g >= 190) & (b >= 230))  # drop open water
    )
    ys, xs = np.where(m)
    clusters = _cluster(ys + MAP_Y0, xs + MAP_X0, radius=22, min_size=6)
    fruits = []
    for n, cx, cy in clusters:
        # Water lakes are huge; fruit cubes are small
        if n > 140:
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
    from . import coords

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
    return {
        "do_it_pixel": doit,
        "train_pixel": train,
        "back_lit": back_lit,
        "do_it_rgb": doit_px,
        "train_rgb": train_px,
        "back_rgb": back_px,
        "do_it_blobs": find_doit_buttons(arr),
        "train_blobs": find_train_buttons(arr),
    }


def observe_pixels(im: Image.Image) -> dict[str, Any]:
    arr = as_rgb(im)
    flags = overlay_flags(arr)
    return {
        "overlay": flags,
        "move_marks": find_move_marks(arr)[:12],
        "fruit": find_fruit(arr)[:10],
    }
