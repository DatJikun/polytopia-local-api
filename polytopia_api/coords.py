"""UI coordinates in 1920×1200 design space, scaled to the live frame.

Clicks and OCR crops were measured on a 1920×1200 Cloud Agent display.
1280×800 is the same 16:10 aspect (scale 2/3). Any other framebuffer is
mapped independently on X and Y.

Call ``set_frame(w, h)`` from the screenshot / window size before clicking.
``DO_IT`` / ``END_TURN`` / crops then return **screen** pixels.

Escape opens Settings — never send it. Close overlays with BACK.
"""

from __future__ import annotations

from typing import Any

BASE_W, BASE_H = 1920, 1200

_P: dict[str, tuple[int, int]] = {
    "DO_IT": (1117, 681),
    "TRAIN": (1110, 790),
    "RESEARCH": (1082, 715),
    "BACK": (38, 42),
    "SETTINGS": (880, 1124),
    "GAME_STATS": (1000, 1124),
    "TECH_TREE": (1092, 1124),
    "END_TURN": (1111, 1130),
    "WORKSHOP": (894, 655),
    "DESELECT": (700, 300),
}

_BOX: dict[str, tuple[int, int, int, int]] = {
    "HUD_CROP": (700, 0, 1250, 80),
    "UNIT_CROP": (0, 1000, 760, 1200),
}

_frame_w, _frame_h = BASE_W, BASE_H


def set_frame(w: int, h: int) -> None:
    global _frame_w, _frame_h
    _frame_w, _frame_h = max(1, int(w)), max(1, int(h))


def frame() -> tuple[int, int]:
    return _frame_w, _frame_h


def scale() -> tuple[float, float]:
    return _frame_w / BASE_W, _frame_h / BASE_H


def xy(x: int | float, y: int | float) -> tuple[int, int]:
    sx, sy = scale()
    return int(round(x * sx)), int(round(y * sy))


def box(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    a = xy(x0, y0)
    b = xy(x1, y1)
    return a[0], a[1], b[0], b[1]


def clamp_box(b: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in b)
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


class _Pt:
    def __init__(self, name: str) -> None:
        self.name = name

    def pair(self) -> tuple[int, int]:
        return xy(*_P[self.name])

    def __iter__(self):
        return iter(self.pair())

    def __getitem__(self, i: int) -> int:
        return self.pair()[i]

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"{self.name}{self.pair()}"


class _Box:
    def __init__(self, name: str) -> None:
        self.name = name

    def tuple(self) -> tuple[int, int, int, int]:
        return box(*_BOX[self.name])

    def __iter__(self):
        return iter(self.tuple())

    def __getitem__(self, i: int) -> int:
        return self.tuple()[i]

    def __len__(self) -> int:
        return 4

    def __repr__(self) -> str:
        return f"{self.name}{self.tuple()}"


DO_IT = _Pt("DO_IT")
TRAIN = _Pt("TRAIN")
RESEARCH = _Pt("RESEARCH")
BACK = _Pt("BACK")
SETTINGS = _Pt("SETTINGS")
GAME_STATS = _Pt("GAME_STATS")
TECH_TREE = _Pt("TECH_TREE")
END_TURN = _Pt("END_TURN")
WORKSHOP = _Pt("WORKSHOP")
DESELECT = _Pt("DESELECT")
HUD_CROP = _Box("HUD_CROP")
UNIT_CROP = _Box("UNIT_CROP")


def layout_info() -> dict[str, Any]:
    sx, sy = scale()
    return {
        "design": [BASE_W, BASE_H],
        "frame": [_frame_w, _frame_h],
        "scale": [round(sx, 4), round(sy, 4)],
        "screen": {name: list(xy(*pt)) for name, pt in _P.items()},
        "crops": {name: list(box(*b)) for name, b in _BOX.items()},
    }
