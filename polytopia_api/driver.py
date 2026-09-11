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

from PIL import Image, ImageEnhance, ImageOps

from . import coords

DISPLAY = os.environ.get("DISPLAY", ":1")
SHOT_PATH = Path("/tmp/polytopia-api.png")
HUD_PATH = Path("/tmp/polytopia-hud.png")
UNIT_PATH = Path("/tmp/polytopia-unit.png")
WINDOW_NAME = "Polytopia"
GAME_BIN = "Polytopia.x86_64"
_BROWSER_MARK = ("chrome", "chromium", "firefox", "google-chrome", "msedge")


class PolytopiaError(RuntimeError):
    pass


def argv0(cmd: str) -> str:
    cmd = (cmd or "").strip()
    return cmd.split(None, 1)[0] if cmd else ""


def is_game_bin(cmd: str) -> bool:
    """True if the *executable path* ends with /Polytopia.x86_64.

    Steam wrappers often *contain* that name as an argument; they start with
    steam.sh / reaper and flags (`` -``). Game argv0 may include spaces
    (``The Battle of Polytopia/Polytopia.x86_64``).
    """
    cmd = (cmd or "").strip()
    m = re.match(r"^(.*?/Polytopia\.x86_64)(?:\s|$)", cmd)
    if not m:
        return False
    exe = m.group(1)
    if re.search(r"\s-", exe):
        return False
    return exe.rsplit("/", 1)[-1] == GAME_BIN


def pick_game_process(lines: list[str]) -> tuple[int, str] | None:
    """Last matching game binary. Wrapper lines that merely *contain* the name are skipped."""
    hits: list[tuple[int, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        low = cmd.lower()
        if "grep" in low.split()[:1] or low.startswith("grep "):
            continue
        if is_game_bin(cmd):
            hits.append((pid, cmd))
    return hits[-1] if hits else None


def is_browser_window(name: str | None, wm_class: str | None) -> bool:
    blob = f"{name or ''} {wm_class or ''}".lower()
    if "polytopia.local" in blob:
        return True
    return any(m in blob for m in _BROWSER_MARK)


def is_game_window(name: str | None, wm_class: str | None) -> bool:
    if is_browser_window(name, wm_class):
        return False
    n = (name or "").strip()
    if n == WINDOW_NAME:
        return True
    classes = re.findall(r'"([^"]+)"', wm_class or "")
    if not classes and wm_class:
        classes = [wm_class.strip()]
    return any(c.strip() == WINDOW_NAME for c in classes)


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


def _wids_from(args: list[str]) -> list[str]:
    r = _run(args)
    return [w for w in (r.stdout or "").split() if w.isdigit()]


def _wm_class(wid: str) -> str:
    r = _run(["xprop", "-id", str(wid), "WM_CLASS"])
    if r.returncode == 0 and (r.stdout or "").strip():
        return r.stdout.strip()
    r2 = _run(["xdotool", "getwindowclassname", str(wid)])
    return (r2.stdout or "").strip()


def _geometry(wid: str) -> dict[str, int | None]:
    geo = _run(["xdotool", "getwindowgeometry", "--shell", wid])
    vals: dict[str, int] = {}
    for line in (geo.stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                vals[k] = int(v)
            except ValueError:
                continue
    return {
        "x": vals.get("X"),
        "y": vals.get("Y"),
        "width": vals.get("WIDTH"),
        "height": vals.get("HEIGHT"),
    }


def _inspect_window(wid: str) -> dict[str, Any]:
    n = _run(["xdotool", "getwindowname", wid])
    name = (n.stdout or "").strip() or None
    cls = _wm_class(wid)
    pid = None
    pr = _run(["xdotool", "getwindowpid", wid])
    try:
        pid = int((pr.stdout or "").strip())
    except ValueError:
        pass
    geo = _geometry(wid)
    return {
        "window_id": int(wid),
        "name": name,
        "wm_class": cls,
        "pid": pid,
        **geo,
    }


def _empty_window() -> dict[str, Any]:
    return {
        "found": False,
        "pid": None,
        "window_id": None,
        "name": None,
        "wm_class": None,
        "x": None,
        "y": None,
        "width": None,
        "height": None,
        "cmd": None,
        "display": DISPLAY,
        "match": None,
    }


def find_window() -> dict[str, Any]:
    """Unity game window only — never Steam wrapper PID, never Chrome polytopia.local."""
    ps = _run(["pgrep", "-af", GAME_BIN])
    picked = pick_game_process((ps.stdout or "").splitlines())
    pid, cmd = picked if picked else (None, None)

    wids: list[str] = []
    if pid is not None:
        wids.extend(_wids_from(["xdotool", "search", "--pid", str(pid)]))
    named = _wids_from(["xdotool", "search", "--name", f"^{WINDOW_NAME}$"])
    classed = _wids_from(["xdotool", "search", "--class", f"^{WINDOW_NAME}$"])
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for src, group in (("pid", wids), ("name", named), ("class", classed)):
        for w in group:
            if w in seen:
                continue
            seen.add(w)
            ordered.append((src, w))

    best: dict[str, Any] | None = None
    best_src = None
    best_area = -1
    for src, wid in ordered:
        info = _inspect_window(wid)
        if not is_game_window(info.get("name"), info.get("wm_class")):
            continue
        area = int(info.get("width") or 0) * int(info.get("height") or 0)
        if area >= best_area:
            best_area = area
            best = info
            best_src = src
    if not best:
        empty = _empty_window()
        empty["cmd"] = cmd
        empty["pid"] = pid
        return empty
    return {
        "found": True,
        "pid": best.get("pid") or pid,
        "window_id": best["window_id"],
        "name": best.get("name"),
        "wm_class": best.get("wm_class"),
        "cmd": cmd,
        "display": DISPLAY,
        "match": best_src,
        "x": best.get("x"),
        "y": best.get("y"),
        "width": best.get("width"),
        "height": best.get("height"),
    }


def display_size() -> tuple[int, int]:
    """Framebuffer size. POLYTOPIA_SCREEN=1280x800 overrides."""
    env = os.environ.get("POLYTOPIA_SCREEN", "").strip().lower()
    if "x" in env:
        a, b = env.split("x", 1)
        try:
            return int(a), int(b)
        except ValueError:
            pass
    r = _run(["xdotool", "getdisplaygeometry"])
    parts = (r.stdout or "").split()
    if len(parts) >= 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return coords.BASE_W, coords.BASE_H


def sync_layout(info: dict[str, Any] | None = None, image: Image.Image | None = None) -> dict[str, Any]:
    """Point coords.py at the live window / screenshot / display."""
    w = h = None
    if image is not None:
        w, h = image.size
    elif info and info.get("width") and info.get("height"):
        w, h = int(info["width"]), int(info["height"])
    else:
        w, h = display_size()
    coords.set_frame(w, h)
    return coords.layout_info()


def grab_region() -> tuple[str, int, int]:
    """ffmpeg x11grab source: size + offset of the game window, else full display."""
    info = find_window()
    if info.get("found") and info.get("width") and info.get("height"):
        w, h = int(info["width"]), int(info["height"])
        ox = int(info.get("x") or 0)
        oy = int(info.get("y") or 0)
        coords.set_frame(w, h)
        return f"{w}x{h}", ox, oy
    w, h = display_size()
    coords.set_frame(w, h)
    return f"{w}x{h}", 0, 0


def activate() -> dict[str, Any]:
    info = find_window()
    if not info["found"]:
        raise PolytopiaError("Polytopia window not found")
    _run(["xdotool", "windowactivate", "--sync", str(info["window_id"])])
    time.sleep(0.05)
    return info


def screenshot(path: Path | None = None) -> Path:
    out = path or SHOT_PATH
    size, ox, oy = grab_region()
    src = f"{DISPLAY}.0+{ox},{oy}"
    r = _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "x11grab",
            "-video_size",
            size,
            "-i",
            src,
            "-frames:v",
            "1",
            str(out),
        ],
        timeout=12,
    )
    if r.returncode != 0 or not out.exists():
        raise PolytopiaError(f"screenshot failed: {(r.stderr or r.stdout)[-400:]}")
    try:
        im = Image.open(out)
        coords.set_frame(*im.size)
    except Exception:
        pass
    return out


def pixel(x: int, y: int, path: Path | None = None) -> tuple[int, int, int]:
    im = Image.open(path or screenshot())
    r, g, b = im.getpixel((int(x), int(y)))[:3]
    return int(r), int(g), int(b)


def click(
    x: int,
    y: int,
    button: int = 1,
    repeats: int = 1,
    pause: float = 0.12,
    space: str = "screen",
    allow_dock: bool = False,
) -> dict[str, Any]:
    """Click. ``space=design`` maps 1920×1200 playbook coords onto the live frame."""
    info = activate()
    sync_layout(info)
    if space == "design":
        x, y = coords.xy(x, y)
    if not allow_dock and coords.in_dock_zone(int(x), int(y)):
        raise PolytopiaError(
            f"dock click denied at {[int(x), int(y)]} {coords.dock_zone()} — "
            "use POST /end-turn or POST /click {name:TECH_TREE}. "
            "computerUse must not hit End Turn."
        )
    wid = str(info["window_id"])
    _run(["xdotool", "mousemove", "--window", wid, str(int(x)), str(int(y))])
    time.sleep(pause)
    for _ in range(max(1, repeats)):
        _run(["xdotool", "click", str(button)])
        time.sleep(0.15)
    return {
        "ok": True,
        "x": int(x),
        "y": int(y),
        "space": space,
        "window_id": info["window_id"],
        "layout": coords.layout_info(),
    }


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
    return click(*coords.END_TURN, repeats=2, pause=0.1, allow_dock=True)


def click_named(name: str, repeats: int = 1) -> dict[str, Any]:
    """Click a layout button by name. Refuses Tech/Settings if they sit on End Turn."""
    key = str(name or "").strip().upper().replace("-", "_")
    aliases = {
        "TECH": "TECH_TREE",
        "TREE": "TECH_TREE",
        "STATS": "GAME_STATS",
        "GAME": "GAME_STATS",
        "END": "END_TURN",
        "NEXT": "END_TURN",
    }
    key = aliases.get(key, key)
    info = activate()
    sync_layout(info)
    if key not in coords._P:
        raise PolytopiaError(f"unknown button {name}")
    pt = coords.point(key)
    if coords.too_close_to_end_turn(key, pt):
        end = coords.point("END_TURN")
        raise PolytopiaError(
            f"{key} {list(pt)} is too close to END_TURN {list(end)} "
            f"(<{coords.DOCK_MIN_SEP}px) — calibrate TECH_TREE instead of clicking"
        )
    result = click(pt[0], pt[1], space="screen", repeats=repeats, allow_dock=True)
    result["name"] = key
    result["source"] = coords.source_of(key)
    return result


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


def _prep_ocr(crop: Image.Image, scale: int = 3, invert: bool | None = None) -> Image.Image:
    """Dark HUD → white paper, upscaled. Helps tesseract on tiny digits."""
    g = crop.convert("L")
    g = ImageOps.autocontrast(g)
    pixels = list(g.getdata())
    mean = sum(pixels) / max(1, len(pixels))
    do_inv = invert if invert is not None else mean < 140
    if do_inv:
        g = ImageOps.invert(g)
    g = ImageEnhance.Contrast(g).enhance(1.8)
    w, h = g.size
    return g.resize((max(1, w * scale), max(1, h * scale)), Image.Resampling.LANCZOS)


def ocr_image(
    im: Image.Image,
    path: Path,
    psm: int = 6,
    whitelist: str | None = None,
) -> str:
    im.save(path)
    args = ["tesseract", str(path), "stdout", "--psm", str(psm), "--oem", "3"]
    if whitelist:
        args += ["-c", f"tessedit_char_whitelist={whitelist}"]
    r = _run(args, timeout=15)
    return (r.stdout or "").strip()


def ocr_crop(im: Image.Image, box: tuple[int, int, int, int], path: Path) -> str:
    coords.set_frame(*im.size)
    crop_box = coords.clamp_box(tuple(int(v) for v in box), *im.size)
    crop = _prep_ocr(im.crop(crop_box))
    return ocr_image(crop, path, psm=6)


def _ocr_fix_digits(text: str) -> str:
    """Common tesseract confusions in the HUD number strip."""
    out = []
    for tok in (text or "").replace("\n", " ").split():
        if re.fullmatch(r"[0-9OIlSZsso.,*★xX+-]+", tok):
            t = (
                tok.replace("O", "0")
                .replace("o", "0")
                .replace("I", "1")
                .replace("l", "1")
                .replace("S", "5")
                .replace("s", "5")
                .replace("Z", "2")
            )
            out.append(t)
        else:
            out.append(tok)
    return " ".join(out)


def _filled(parsed: dict[str, Any]) -> int:
    return sum(parsed.get(k) is not None for k in ("score", "stars", "turn"))


def _first_plausible(nums: list[int], lo: int, hi: int) -> int | None:
    for n in nums:
        if lo <= n <= hi:
            return n
    return None


_HUD_CACHE: dict[str, Any] | None = None


def _hud_star_x(crop: Image.Image) -> int | None:
    """Gold star icon in the HUD — splits score | ★ | turn."""
    arr = __import__("numpy").asarray(crop.convert("RGB")).astype("int16")
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    star = (r >= 190) & (g >= 140) & (b <= 110) & (r + g >= 360) & (r >= b + 80)
    ys, xs = __import__("numpy").where(star)
    if xs.size < 8:
        return None
    return int(xs.mean())


def read_hud(im: Image.Image) -> dict[str, Any]:
    """OCR the top strip: score left of the star, ★ then turn to the right."""
    global _HUD_CACHE
    coords.set_frame(*im.size)
    box = coords.clamp_box(tuple(int(v) for v in coords.HUD_CROP), *im.size)
    crop = im.crop(box)
    candidates: list[tuple[int, dict[str, Any], str, str]] = []
    digits: dict[str, str] = {}
    star_x = _hud_star_x(crop)

    for invert in (True, False):
        prepared = _prep_ocr(crop, scale=3, invert=invert)
        raw_full = ocr_image(prepared, HUD_PATH, psm=6)
        raw_line = ocr_image(prepared, Path("/tmp/polytopia-hud-line.png"), psm=7)
        blob = _ocr_fix_digits("\n".join(x for x in (raw_full, raw_line) if x))
        parsed = parse_hud(blob)
        candidates.append((_filled(parsed), parsed, raw_full, raw_line))

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, parsed, raw_full, raw_line = candidates[0]

    w, h = crop.size
    if star_x is not None:
        sx = max(8, min(w - 8, star_x))
        slices = {
            "score": (0, 0, max(1, sx - 4), h),
            "stars": (sx, 0, min(w, sx + max(28, w // 6)), h),
            "turn": (min(w - 1, sx + max(24, w // 7)), 0, w, h),
        }
    else:
        slices = {
            "score": (0, 0, max(1, int(w * 0.45)), h),
            "stars": (int(w * 0.28), 0, max(int(w * 0.28) + 1, int(w * 0.72)), h),
            "turn": (int(w * 0.58), 0, w, h),
        }
    for key, sl in slices.items():
        sub = _prep_ocr(crop.crop(sl), scale=4, invert=True)
        txt = ocr_image(
            sub,
            Path(f"/tmp/polytopia-hud-{key}.png"),
            psm=8,
            whitelist="0123456789,",
        )
        txt = _ocr_fix_digits(txt)
        digits[key] = txt
        nums = [int(n.replace(",", "")) for n in re.findall(r"\d{1,3}(?:,\d{3})+|\d+", txt)]
        if key == "score":
            crop_val = _first_plausible(nums, 80, 80000) or _first_plausible(nums, 0, 80000)
            if crop_val is not None:
                parsed["score"] = crop_val
        elif key == "stars":
            crop_val = _first_plausible(nums, 0, 80)
            if crop_val is not None:
                parsed["stars"] = crop_val
        elif key == "turn" and parsed.get("turn") is None:
            parsed["turn"] = _first_plausible(nums, 0, 200)

    missing = [k for k in ("score", "stars", "turn") if parsed.get(k) is None]
    parsed["missing"] = missing
    parsed["stale"] = False
    parsed["stale_fields"] = []
    parsed["turn_ocr"] = parsed.get("turn")
    prev_turn = (_HUD_CACHE or {}).get("turn")
    # Never impersonate a turn number from cache (T24 ended up as T4.json).
    if missing and _HUD_CACHE:
        for k in missing:
            if k == "turn":
                continue
            if _HUD_CACHE.get(k) is not None:
                parsed[k] = _HUD_CACHE[k]
                parsed["stale"] = True
                parsed["stale_fields"].append(k)
    if isinstance(parsed.get("turn"), int) and isinstance(prev_turn, int) and parsed["turn"] < prev_turn:
        parsed["turn_ocr"] = parsed["turn"]
        parsed["turn"] = None
        parsed["stale"] = True
        if "turn" not in parsed["stale_fields"]:
            parsed["stale_fields"].append("turn")
        if "turn" not in parsed["missing"]:
            parsed["missing"].append("turn")
    from . import snapshot as _snap

    _snap.apply_turn_floor(parsed)
    if parsed.get("turn_untrusted_reason"):
        parsed["turn_trusted"] = False
    else:
        parsed["turn_trusted"] = parsed.get("turn") is not None and "turn" not in (
            parsed.get("stale_fields") or []
        )
    floor = parsed.get("turn_floor")
    if parsed.get("turn_trusted"):
        cache_turn = parsed.get("turn")
    elif isinstance(floor, int):
        cache_turn = floor
    else:
        cache_turn = prev_turn
    if sum(v is not None for v in (parsed.get("score"), parsed.get("stars"), cache_turn)) >= 2:
        _HUD_CACHE = {
            "score": parsed.get("score"),
            "stars": parsed.get("stars"),
            "turn": cache_turn,
            "income": parsed.get("income"),
        }

    parsed["raw"] = raw_full
    parsed["raw_line"] = raw_line
    parsed["digits"] = digits
    parsed["hud_crop"] = list(box)
    parsed["star_x"] = star_x
    return parsed


def parse_hud(text: str) -> dict[str, Any]:
    """Parse HUD OCR. Tolerates Tum/Turn, O/0, 1,800 / 1800 score."""
    compact = _ocr_fix_digits((text or "").replace("\n", " "))
    score = None
    stars = None
    turn = None
    income = None

    m_inc = re.search(r"\(\s*\+?\s*(-?\d+)\s*\)", compact)
    if m_inc:
        income = int(m_inc.group(1))

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
        if income is not None:
            nums = [n for n in nums if n != income]
        big = [n for n in nums if n >= 80]
        if big:
            score = max(big)
            rest = [n for n in nums if n != score]
        else:
            rest = list(nums)
        if len(rest) >= 2:
            stars, turn = rest[0], rest[1]
        elif len(rest) == 1:
            # One leftover: turn sits on the right of the strip more often than stars.
            if rest[0] <= 200:
                turn = rest[0]

    low = (text or "").lower()
    game_stats = "game mode" in low or "game stats" in low
    return {
        "raw": text,
        "score": score,
        "stars": stars,
        "income": income,
        "turn": turn,
        "game_stats": game_stats,
    }


def parse_unit_panel(text: str) -> dict[str, Any]:
    low = text.lower()
    can_move = "blue mark" in low or "select a blue" in low
    no_actions = "no actions left" in low or ("next turn" in low and "no action" in low)
    harvest = "harvest" in low
    clear_forest = "clear forest" in low
    # Tesseract often reads the pill as Traln / TRAN; "choose a unit" is the TRAIN panel.
    train = (
        "train" in low
        or "traln" in low
        or "choose a unit" in low
        or "choose unit" in low
    )
    disband = "disband" in low
    village = "village" in low
    settings = "settings" in low
    game_stats = "game mode" in low or "game stats" in low
    capture_soon = any(
        s in low
        for s in (
            "ready to capture next",
            "will be ready to capture",
            "entering village",
            "entering city",
        )
    )
    capture = ("capture" in low or "conquer" in low) and not capture_soon
    unit = None
    for name in (
        "catapult",
        "archer",
        "warrior",
        "rider",
        "defender",
        "knight",
        "giant",
        "bomber",
        "raft",
        "scout",
        "rammer",
        "mind bender",
        "boat",
    ):
        if name in low:
            unit = name.replace(" ", "_")
            break
    return {
        "raw": text,
        "unit": unit,
        "can_move": can_move,
        "no_actions": no_actions,
        "harvest": harvest,
        "clear_forest": clear_forest,
        "train": train and not disband,
        "disband": disband,
        "village": village,
        "settings": settings or game_stats,
        "game_stats": game_stats,
        "capture": capture,
        "capture_soon": capture_soon,
    }


def hud() -> dict[str, Any]:
    shot = screenshot()
    im = Image.open(shot)
    coords.set_frame(*im.size)
    parsed = read_hud(im)
    info = find_window()
    parsed["window"] = {k: info[k] for k in ("found", "pid", "window_id", "width", "height")}
    parsed["layout"] = coords.layout_info()
    parsed["screenshot"] = str(shot)
    return parsed
