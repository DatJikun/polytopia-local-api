"""UI coordinates: empirical screen points beat naive 1920→frame scale.

Clicks on the Cloud Agent were measured in 1920×1200 *design* space. Unity's
dock does **not** land on 2/3 of those numbers at 1280×800 — End Turn is
(765, 746), not the scaled (741, 753). Named dock buttons therefore:

1. Exact-frame **empirical** override (builtin or calibrated), else
2. Scaled design-space fallback.

Map hits from ``/observe`` are always live-frame pixels. Escape opens
Settings — never send it. Close overlays with BACK.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
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

# Screen pixels, keyed by live frame. Never derived from 2/3 scale.
_BUILTIN_EMPIRICAL: dict[tuple[int, int], dict[str, tuple[int, int]]] = {
    (1280, 800): {
        "END_TURN": (765, 746),
    },
}

_frame_w, _frame_h = BASE_W, BASE_H
# runtime overrides: (w, h) -> {name: (x, y)}
_empirical: dict[tuple[int, int], dict[str, tuple[int, int]]] = {}
_loaded_file: Path | None = None


def layout_file() -> Path:
    env = os.environ.get("POLYTOPIA_LAYOUT", "").strip()
    if env:
        return Path(env).expanduser()
    xdg = Path.home() / ".config" / "polytopia-local-api" / "layout.json"
    cwd = Path.cwd() / "layout.json"
    if cwd.is_file():
        return cwd
    return xdg


def _merge_builtin() -> None:
    for frame, pts in _BUILTIN_EMPIRICAL.items():
        slot = _empirical.setdefault(frame, {})
        for name, xy_ in pts.items():
            slot.setdefault(name, xy_)


def load_empirical(path: Path | None = None) -> Path | None:
    """Load calibrated screen points. Missing file is fine."""
    global _loaded_file
    _merge_builtin()
    p = path or layout_file()
    if not p.is_file():
        _loaded_file = None
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _loaded_file = None
        return None
    frames = raw.get("frames") if isinstance(raw, dict) else None
    if not isinstance(frames, dict):
        frames = raw if isinstance(raw, dict) else {}
    for key, pts in frames.items():
        if key in {"design", "comment", "frames"}:
            continue
        if not isinstance(pts, dict):
            continue
        if "x" in str(key):
            try:
                a, b = str(key).lower().split("x", 1)
                frame = (int(a), int(b))
            except ValueError:
                continue
        else:
            continue
        slot = _empirical.setdefault(frame, {})
        for name, val in pts.items():
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                slot[str(name)] = (int(val[0]), int(val[1]))
    _loaded_file = p
    return p


def save_empirical(path: Path | None = None) -> Path:
    p = path or layout_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    frames: dict[str, dict[str, list[int]]] = {}
    for (w, h), pts in sorted(_empirical.items()):
        frames[f"{w}x{h}"] = {n: [int(x), int(y)] for n, (x, y) in sorted(pts.items())}
    payload = {
        "comment": "Screen-pixel empirical overrides. Not scaled from 1920×1200.",
        "design": [BASE_W, BASE_H],
        "frames": frames,
    }
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p


def reset(clear_file_overrides: bool = True) -> None:
    """Restore builtin empiria. Tests call this."""
    global _empirical, _loaded_file, _frame_w, _frame_h
    _empirical = {}
    _loaded_file = None
    _frame_w, _frame_h = BASE_W, BASE_H
    _merge_builtin()
    if not clear_file_overrides:
        load_empirical()


_merge_builtin()


def set_frame(w: int, h: int) -> None:
    global _frame_w, _frame_h
    _frame_w, _frame_h = max(1, int(w)), max(1, int(h))


def frame() -> tuple[int, int]:
    return _frame_w, _frame_h


def scale() -> tuple[float, float]:
    return _frame_w / BASE_W, _frame_h / BASE_H


def xy(x: int | float, y: int | float) -> tuple[int, int]:
    """Map a 1920×1200 *design* point onto the live frame (naive scale)."""
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


def _slot() -> dict[str, tuple[int, int]]:
    return _empirical.get((_frame_w, _frame_h), {})


def source_of(name: str) -> str:
    return "empirical" if name in _slot() else "scaled"


def point(name: str) -> tuple[int, int]:
    emp = _slot().get(name)
    if emp is not None:
        return int(emp[0]), int(emp[1])
    if name not in _P:
        raise KeyError(name)
    return xy(*_P[name])


def set_empirical(name: str, x: int, y: int, w: int | None = None, h: int | None = None) -> dict[str, Any]:
    """Record a live-frame click. Persists after save_empirical()."""
    fw = int(w or _frame_w)
    fh = int(h or _frame_h)
    slot = _empirical.setdefault((fw, fh), {})
    slot[str(name)] = (int(x), int(y))
    return {"name": name, "x": int(x), "y": int(y), "frame": [fw, fh], "source": "empirical"}


def crop_box(name: str) -> tuple[int, int, int, int]:
    """HUD / unit panel. Fractions of the live frame, not 2/3 of 1920."""
    w, h = _frame_w, _frame_h
    if name == "HUD_CROP":
        # Top-center strip; generous so digit OCR still sees ★ / turn.
        return int(w * 0.28), 0, int(w * 0.72), max(36, int(h * 0.09))
    if name == "UNIT_CROP":
        return 0, int(h * 0.80), int(w * 0.48), h
    return box(*_BOX[name])


class _Pt:
    def __init__(self, name: str) -> None:
        self.name = name

    def pair(self) -> tuple[int, int]:
        return point(self.name)

    def __iter__(self):
        return iter(self.pair())

    def __getitem__(self, i: int) -> int:
        return self.pair()[i]

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"{self.name}{self.pair()}/{source_of(self.name)}"


class _Box:
    def __init__(self, name: str) -> None:
        self.name = name

    def tuple(self) -> tuple[int, int, int, int]:
        return crop_box(self.name)

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
    sources = {name: source_of(name) for name in _P}
    screen = {name: list(point(name)) for name in _P}
    n_emp = sum(1 for s in sources.values() if s == "empirical")
    return {
        "design": [BASE_W, BASE_H],
        "frame": [_frame_w, _frame_h],
        "scale": [round(sx, 4), round(sy, 4)],
        "scale_note": "fallback only — dock buttons prefer empirical screen pixels",
        "empirical": n_emp > 0,
        "source": sources,
        "screen": screen,
        "crops": {name: list(crop_box(name)) for name in _BOX},
        "layout_file": str(_loaded_file) if _loaded_file else None,
    }
