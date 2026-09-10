"""Drive a locally running Polytopia window via screenshot + xdotool.

This is not Hello Games' protocol. It talks to the Unity window the same way
a player does: pixels in, mouse clicks out.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image

from . import coords

DISPLAY = os.environ.get("DISPLAY", ":1")
SCREEN_SIZE = os.environ.get("POLYTOPIA_SCREEN", "1920x1200")
SHOT_PATH = Path("/tmp/polytopia-api.png")
HUD_PATH = Path("/tmp/polytopia-hud.png")
UNIT_PATH = Path("/tmp/polytopia-unit.png")
WINDOW_NAME = "Polytopia"


class PolytopiaError(RuntimeError):
    pass


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["DISPLAY"] = DISPLAY
    return env


def _run(args: list[str], timeout: float = 8) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def find_window() -> dict[str, Any]:
    """Return pid + window id for the Unity Polytopia process."""
    ps = _run(["pgrep", "-af", "Polytopia.x86_64"])
    pid = None
    cmd = None
    for line in (ps.stdout or "").splitlines():
        if "Polytopia.x86_64" in line and "grep" not in line:
            parts = line.strip().split(None, 1)
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            cmd = parts[1] if len(parts) > 1 else ""
            break
    if pid is None:
        return {"found": False, "pid": None, "window_id": None, "name": None}

    search = _run(["xdotool", "search", "--pid", str(pid)])
    wids = [w for w in (search.stdout or "").split() if w.isdigit()]
    wid = wids[-1] if wids else None
    name = None
    if wid:
        n = _run(["xdotool", "getwindowname", wid])
        name = (n.stdout or "").strip() or None
    return {
        "found": bool(wid),
        "pid": pid,
        "window_id": int(wid) if wid else None,
        "name": name,
        "cmd": cmd,
        "display": DISPLAY,
    }


def activate() -> dict[str, Any]:
    info = find_window()
    if not info["found"]:
        raise PolytopiaError("Polytopia window not found")
    _run(["xdotool", "windowactivate", "--sync", str(info["window_id"])])
    time.sleep(0.05)
    return info


def screenshot(path: Path | None = None) -> Path:
    out = path or SHOT_PATH
    r = _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "x11grab",
            "-video_size",
            SCREEN_SIZE,
            "-i",
            f"{DISPLAY}.0",
            "-frames:v",
            "1",
            str(out),
        ],
        timeout=12,
    )
    if r.returncode != 0 or not out.exists():
        raise PolytopiaError(f"screenshot failed: {(r.stderr or r.stdout)[-400:]}")
    return out


def pixel(x: int, y: int, path: Path | None = None) -> tuple[int, int, int]:
    im = Image.open(path or screenshot())
    r, g, b = im.getpixel((int(x), int(y)))[:3]
    return int(r), int(g), int(b)


def click(x: int, y: int, button: int = 1, repeats: int = 1, pause: float = 0.12) -> dict[str, Any]:
    info = activate()
    wid = str(info["window_id"])
    _run(["xdotool", "mousemove", "--window", wid, str(int(x)), str(int(y))])
    time.sleep(pause)
    for _ in range(max(1, repeats)):
        _run(["xdotool", "click", str(button)])
        time.sleep(0.15)
    return {"ok": True, "x": int(x), "y": int(y), "window_id": info["window_id"]}


def press(key: str) -> dict[str, Any]:
    if key.lower() in {"escape", "esc"}:
        raise PolytopiaError("Escape opens Settings — use POST /back instead")
    info = activate()
    _run(["xdotool", "key", key])
    return {"ok": True, "key": key, "window_id": info["window_id"]}


def back() -> dict[str, Any]:
    return click(*coords.BACK)


def end_turn() -> dict[str, Any]:
    """Firm double click on End Turn (single clicks often miss)."""
    return click(*coords.END_TURN, repeats=2, pause=0.1)


def confirm() -> dict[str, Any]:
    """Click the blue DO IT / confirm button if it is present."""
    shot = screenshot()
    r, g, b = pixel(*coords.DO_IT, path=shot)
    is_blue = r <= 20 and 140 <= g <= 170 and b >= 240
    if not is_blue:
        r2, g2, b2 = pixel(*coords.TRAIN, path=shot)
        is_train = r2 <= 20 and 100 <= g2 <= 140 and 180 <= b2 <= 230
        if is_train:
            return {**click(*coords.TRAIN), "kind": "train", "rgb": [r2, g2, b2]}
        return {
            "ok": False,
            "kind": None,
            "rgb_doit": [r, g, b],
            "hint": "no blue confirm/train button at known coords",
        }
    return {**click(*coords.DO_IT), "kind": "do_it", "rgb": [r, g, b]}


def ocr_crop(im: Image.Image, box: tuple[int, int, int, int], path: Path) -> str:
    crop = im.crop(box)
    crop.save(path)
    r = _run(["tesseract", str(path), "stdout", "--psm", "6"], timeout=15)
    return (r.stdout or "").strip()


def _ocr_hud_text(im: Image.Image) -> str:
    return ocr_crop(im, coords.HUD_CROP, HUD_PATH)


def parse_hud(text: str) -> dict[str, Any]:
    """Parse HUD OCR. Tolerates Tum/Turn and 1,800 / 1800 score."""
    compact = text.replace("\n", " ")
    score = None
    stars = None
    turn = None
    income = None

    m_inc = re.search(r"\(\s*\+?\s*(-?\d+)\s*\)", compact)
    if m_inc:
        income = int(m_inc.group(1))

    # "1,760 *5 8" or "1800 *0 9"
    m = re.search(
        r"(\d{1,3}(?:,\d{3})+|\d{3,5})\s*[*★xX]?\s*(\d{1,3})\s+(\d{1,3})\b",
        compact,
    )
    if m:
        score = int(m.group(1).replace(",", ""))
        stars = int(m.group(2))
        turn = int(m.group(3))
    else:
        nums = [int(n.replace(",", "")) for n in re.findall(r"\d{1,3}(?:,\d{3})+|\d+", compact)]
        # drop the income we already captured
        if income is not None and income in nums:
            nums = [n for n in nums if n != income]
        if len(nums) >= 3:
            score, stars, turn = nums[0], nums[1], nums[2]
        elif len(nums) == 2:
            stars, turn = nums[0], nums[1]

    return {
        "raw": text,
        "score": score,
        "stars": stars,
        "income": income,
        "turn": turn,
    }


def parse_unit_panel(text: str) -> dict[str, Any]:
    low = text.lower()
    can_move = "blue mark" in low or "select a blue" in low
    no_actions = "no actions left" in low or ("next turn" in low and "no action" in low)
    harvest = "harvest" in low
    clear_forest = "clear forest" in low
    train = "train" in low
    village = "village" in low
    settings = "settings" in low
    unit = None
    for name in ("catapult", "archer", "warrior", "rider", "defender", "knight", "giant"):
        if name in low:
            unit = name
            break
    return {
        "raw": text,
        "unit": unit,
        "can_move": can_move,
        "no_actions": no_actions,
        "harvest": harvest,
        "clear_forest": clear_forest,
        "train": train,
        "village": village,
        "settings": settings,
    }


def hud() -> dict[str, Any]:
    shot = screenshot()
    im = Image.open(shot)
    parsed = parse_hud(_ocr_hud_text(im))
    info = find_window()
    parsed["window"] = {k: info[k] for k in ("found", "pid", "window_id")}
    parsed["screenshot"] = str(shot)
    return parsed
