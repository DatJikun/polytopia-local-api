"""One screenshot → HUD + unit panel + pixel detections."""

from __future__ import annotations

from typing import Any

from PIL import Image

from . import coords
from . import detect
from . import driver


def observe(shot: Any | None = None) -> dict[str, Any]:
    path = shot or driver.screenshot()
    im = Image.open(path)
    coords.set_frame(*im.size)
    hud = driver.parse_hud(driver._ocr_hud_text(im))
    unit = driver.parse_unit_panel(
        driver.ocr_crop(im, coords.UNIT_CROP, driver.UNIT_PATH)
    )
    pix = detect.observe_pixels(im)
    overlay = pix["overlay"]
    confirm_ready = bool(
        overlay["do_it_pixel"]
        or overlay["train_pixel"]
        or overlay["do_it_blobs"]
        or overlay["train_blobs"]
    )
    info = driver.find_window()
    return {
        "hud": hud,
        "unit": unit,
        "move_marks": pix["move_marks"],
        "fruit": pix["fruit"],
        "overlay": overlay,
        "confirm_ready": confirm_ready,
        "hits_space": "screen",
        "window": {
            k: info.get(k)
            for k in ("found", "pid", "window_id", "width", "height")
        },
        "layout": coords.layout_info(),
        "screenshot": str(path),
    }
