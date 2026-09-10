"""One screenshot → HUD + panel + map entities + overlay."""

from __future__ import annotations

from typing import Any

from PIL import Image

from . import combat
from . import coords
from . import detect
from . import driver
from . import entities
from . import snapshot

_LAST: dict[str, Any] | None = None


def remember() -> dict[str, Any] | None:
    return _LAST


def lookup(obs: dict[str, Any], ident: str) -> dict[str, Any] | None:
    ident = str(ident)
    keys = ("units", "cities", "cities_own", "cities_enemy", "villages", "move_marks", "fruit", "attack_marks")
    if ident.startswith("c"):
        keys = ("cities", "cities_own", "cities_enemy") + keys
    low = ident.lower()
    for key in keys:
        for it in obs.get(key) or []:
            if str(it.get("id")) == ident:
                return it
            name = it.get("name")
            if name and str(name).lower() == low:
                return it
    return None


def _ids(prefix: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, it in enumerate(items):
        d = dict(it)
        d.setdefault("id", f"{prefix}{i}")
        out.append(d)
    return out


def stabilize(
    items: list[dict[str, Any]],
    prev: list[dict[str, Any]] | None,
    max_dist: int = 56,
) -> list[dict[str, Any]]:
    """Keep city/unit ids stable across frames when the blob barely moved."""
    if not items:
        return []
    used: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        d = dict(it)
        best = None
        best_d = max_dist
        for p in prev or []:
            pid = str(p.get("id") or "")
            if not pid or pid in used:
                continue
            try:
                dist = (int(d["x"]) - int(p["x"])) ** 2 + (int(d["y"]) - int(p["y"])) ** 2
            except (KeyError, TypeError, ValueError):
                continue
            if dist < best_d * best_d:
                best_d = int(dist ** 0.5)
                best = p
        if best and best.get("id"):
            d["id"] = best["id"]
            if best.get("name") and not d.get("name"):
                d["name"] = best["name"]
            d["stable"] = True
            used.add(str(best["id"]))
        else:
            named = None
            want = str(d.get("name") or "").lower()
            if want:
                for p in prev or []:
                    pid = str(p.get("id") or "")
                    if not pid or pid in used:
                        continue
                    if str(p.get("name") or "").lower() == want:
                        named = p
                        break
            if named:
                d["id"] = named["id"]
                d["stable"] = True
                used.add(str(named["id"]))
            else:
                d["stable"] = False
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
        "attack": bool(overlay.get("attack_marks")),
        "confirm": bool(confirm_ready),
    }


def observe(shot: Any | None = None) -> dict[str, Any]:
    global _LAST
    path = shot or driver.screenshot()
    im = Image.open(path)
    coords.set_frame(*im.size)
    hud = driver.read_hud(im)
    snapshot.apply_turn_floor(hud)
    unit = driver.parse_unit_panel(
        driver.ocr_crop(im, coords.UNIT_CROP, driver.UNIT_PATH)
    )
    pix = detect.observe_pixels(im)
    overlay = pix["overlay"]
    arr = detect.as_rgb(im)
    mapped = entities.observe_map(arr)
    prev = _LAST
    units = stabilize(_ids("u", mapped["units"]), (prev or {}).get("units"))
    cities = stabilize(_ids("c", mapped["cities"]), (prev or {}).get("cities"))
    own = [c for c in cities if c.get("owner") == "own"]
    enemy = [c for c in cities if c.get("owner") == "enemy"]
    attack_marks = _ids("a", pix.get("attack_marks") or overlay.get("attack_marks") or [])
    panel_type = unit.get("unit")
    strike = combat.unit_profile(panel_type)
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
        "units": units,
        "cities": cities,
        "cities_own": own,
        "cities_enemy": enemy,
        "villages": mapped["villages"],
        "villages_enabled": mapped.get("villages_enabled", False),
        "fog_edge": mapped.get("fog_edge") or [],
        "own_tribe": mapped["own_tribe"],
        "move_marks": _ids("m", pix["move_marks"]),
        "attack_marks": attack_marks,
        "fruit": _ids("f", pix["fruit"]),
        "selected_unit": strike,
        "overlay": overlay,
        "confirm_ready": confirm_ready,
        "ready": ready_flags(unit, overlay, confirm_ready),
        "hits_space": "screen",
        "window": {
            k: info.get(k)
            for k in ("found", "pid", "window_id", "width", "height", "match", "name")
        },
        "layout": coords.layout_info(),
        "screenshot": str(path),
    }
    payload["observe_diff"] = snapshot.diff(prev, payload, reason="frame") if prev else None
    payload["turn_diff"] = snapshot.diff(snapshot.last_saved(), payload, reason="observe")
    _LAST = payload
    return payload
