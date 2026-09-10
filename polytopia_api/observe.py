"""One screenshot → HUD + panel + map entities + overlay."""

from __future__ import annotations

from typing import Any

from PIL import Image

from . import coords
from . import detect
from . import driver
from . import entities

_LAST: dict[str, Any] | None = None


def remember() -> dict[str, Any] | None:
    return _LAST


def lookup(obs: dict[str, Any], ident: str) -> dict[str, Any] | None:
    ident = str(ident)
    for key in ("units", "cities", "cities_own", "cities_enemy", "villages", "move_marks", "fruit"):
        for it in obs.get(key) or []:
            if str(it.get("id")) == ident:
                return it
    return None


def _ids(prefix: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, it in enumerate(items):
        d = dict(it)
        d.setdefault("id", f"{prefix}{i}")
        out.append(d)
    return out


def ready_flags(unit: dict[str, Any], overlay: dict[str, Any], confirm_ready: bool) -> dict[str, Any]:
    capture = bool(
        unit.get("capture")
        or (
            unit.get("village")
            and not unit.get("capture_soon")
            and (overlay.get("do_it_pixel") or overlay.get("do_it_blobs"))
        )
    )
    train = bool(unit.get("train") or overlay.get("train_pixel") or overlay.get("train_blobs"))
    harvest = bool(unit.get("harvest") or (confirm_ready and overlay.get("do_it_pixel")))
    return {
        "capture": capture,
        "train": train,
        "harvest": harvest,
        "move": bool(unit.get("can_move")),
        "confirm": bool(confirm_ready),
    }


def observe(shot: Any | None = None) -> dict[str, Any]:
    global _LAST
    path = shot or driver.screenshot()
    im = Image.open(path)
    coords.set_frame(*im.size)
    hud = driver.read_hud(im)
    unit = driver.parse_unit_panel(
        driver.ocr_crop(im, coords.UNIT_CROP, driver.UNIT_PATH)
    )
    pix = detect.observe_pixels(im)
    overlay = pix["overlay"]
    arr = detect.as_rgb(im)
    mapped = entities.observe_map(arr)
    confirm_ready = bool(
        overlay["do_it_pixel"]
        or overlay["train_pixel"]
        or overlay["do_it_blobs"]
        or overlay["train_blobs"]
        or unit.get("capture")
        or unit.get("train")
    )
    info = driver.find_window()
    payload = {
        "hud": hud,
        "unit": unit,
        "units": mapped["units"],
        "cities": mapped["cities"],
        "cities_own": mapped["cities_own"],
        "cities_enemy": mapped["cities_enemy"],
        "villages": mapped["villages"],
        "own_tribe": mapped["own_tribe"],
        "move_marks": _ids("m", pix["move_marks"]),
        "fruit": _ids("f", pix["fruit"]),
        "overlay": overlay,
        "confirm_ready": confirm_ready,
        "ready": ready_flags(unit, overlay, confirm_ready),
        "hits_space": "screen",
        "window": {
            k: info.get(k)
            for k in ("found", "pid", "window_id", "width", "height")
        },
        "layout": coords.layout_info(),
        "screenshot": str(path),
    }
    _LAST = payload
    return payload
