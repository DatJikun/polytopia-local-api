from pathlib import Path

import os

from PIL import Image, ImageDraw

from polytopia_api import coords
from polytopia_api.commands import _city_click_points, capture_target, recruit_target
from polytopia_api.detect import as_rgb, find_move_marks, find_train_buttons, is_move_rgb, is_snow_rgb, is_water_rgb
from polytopia_api.driver import _hud_star_x, parse_hud, parse_unit_panel
from polytopia_api.entities import classify_tribe, find_cities, find_units, find_villages
from polytopia_api.mcp import TOOLS, handle
from polytopia_api.observe import stabilize
from polytopia_api.play import plan
from polytopia_api.snapshot import diff as snapshot_diff


def test_parse_hud():
    p = parse_hud("Score Stars (+7) Tum\n1,750 *5 8")
    assert p == {
        **{k: p[k] for k in p},
        "score": 1750,
        "stars": 5,
        "income": 7,
        "turn": 8,
    }
    p = parse_hud("Score Stars (+7) Turn\n1,800 *0 9")
    assert p["stars"] == 0 and p["turn"] == 9 and p["score"] == 1800
    p = parse_hud("1,8O0 *7 l7")
    assert p["score"] == 1800 and p["stars"] == 7 and p["turn"] == 17


def test_parse_unit():
    u = parse_unit_panel("Bardur Warrior\nSelect a blue mark to move.")
    assert u["can_move"] and u["unit"] == "warrior" and not u["no_actions"]
    u = parse_unit_panel("Bardur Catapult\nNo actions left. Press 'Next Turn' to move this unit again.")
    assert u["no_actions"] and u["unit"] == "catapult"
    u = parse_unit_panel("Harvest Fruit\nExtract this resource to upgrade your city.")
    assert u["harvest"]
    u = parse_unit_panel("Clear Forest")
    assert u["clear_forest"]
    u = parse_unit_panel("Capture\nThis city is ready to capture.")
    assert u["capture"] and not u["capture_soon"]
    u = parse_unit_panel("Entering Village! Will be ready to capture next turn.")
    assert u["capture_soon"] and not u["capture"] and u["village"]
    u = parse_unit_panel("Train\nChoose a unit.")
    assert u["train"]
    u = parse_unit_panel("Traln\nChoose a unit.")
    assert u["train"] and not u.get("disband")
    u = parse_unit_panel("Bardur Warrior\nDisband")
    assert u["disband"] and not u["train"] and u["unit"] == "warrior"
    u = parse_unit_panel("Bardur Raft\nSelect a blue mark to move.")
    assert u["unit"] == "raft" and u["can_move"]


def test_water_vs_move():
    assert is_water_rgb(111, 207, 247)
    assert is_water_rgb(58, 199, 249)
    assert is_water_rgb(143, 255, 255)
    assert is_move_rgb(146, 204, 255)
    assert not is_move_rgb(111, 207, 247)
    assert not is_move_rgb(58, 199, 249)
    # Mountain snow / ice sparkle must not count as ocean (live mountain unit drop).
    assert is_snow_rgb(240, 245, 250)
    assert is_snow_rgb(255, 255, 255)
    assert not is_water_rgb(240, 245, 250)
    assert not is_water_rgb(220, 230, 235)


def test_find_move_marks_ignores_water():
    im = Image.new("RGB", (1920, 1200), (20, 20, 20))
    px = im.load()
    for y in range(700, 820):
        for x in range(900, 1100):
            px[x, y] = (58, 199, 249)
    for y in range(590, 610):
        for x in range(1230, 1255):
            px[x, y] = (146, 204, 255)
    marks = find_move_marks(__import__("numpy").asarray(im).astype("int16"))
    assert marks, marks
    assert all(abs(m["x"] - 1242) < 20 and abs(m["y"] - 600) < 20 for m in marks)


def test_empirical_end_turn_1280():
    coords.reset()
    coords.set_frame(1920, 1200)
    assert tuple(coords.END_TURN) == (1111, 1130)
    assert coords.source_of("END_TURN") == "scaled"

    coords.set_frame(1280, 800)
    # Empiria, not 2/3 of 1111,1130 (that would be 741,753 and misses the dock).
    assert tuple(coords.END_TURN) == (765, 746)
    assert coords.source_of("END_TURN") == "empirical"
    sx, sy = 1280 / 1920, 800 / 1200
    dx, dy = tuple(coords.DO_IT)
    assert coords.source_of("DO_IT") == "scaled"
    assert dx == round(1117 * sx) and dy == round(681 * sy)
    info = coords.layout_info()
    assert info["frame"] == [1280, 800]
    assert info["empirical"] is True
    assert info["source"]["END_TURN"] == "empirical"
    dock = coords.dock_from_end((765, 746))
    assert tuple(coords.SETTINGS) == dock["SETTINGS"]
    assert tuple(coords.TECH_TREE) == dock["TECH_TREE"]
    assert abs(coords.SETTINGS[1] - 746) <= 8
    tech, end = tuple(coords.TECH_TREE), tuple(coords.END_TURN)
    dist = ((tech[0] - end[0]) ** 2 + (tech[1] - end[1]) ** 2) ** 0.5
    assert dist >= coords.DOCK_MIN_SEP, (tech, end, dist)
    assert not coords.too_close_to_end_turn("TECH_TREE", tech)
    # The T18→T22 bug: 1920 Tech–End gap is ~19px → ~13px after docking to 1280 End Turn.
    assert coords.too_close_to_end_turn("TECH_TREE", (end[0] - 13, end[1]))
    info = coords.layout_info()
    assert info["tech_too_close"] is False
    assert info["tech_end_dist"] >= coords.DOCK_MIN_SEP
    # Not naive origin-scale (that misses the 1280 dock).
    assert tuple(coords.SETTINGS) != coords.xy(880, 1124)
    hud = tuple(coords.HUD_CROP)
    assert hud[1] == 0 and hud[3] <= 110
    coords.set_empirical("DO_IT", 740, 450)
    assert tuple(coords.DO_IT) == (740, 450)
    assert coords.source_of("DO_IT") == "empirical"
    coords.set_frame(1920, 1200)
    tech1920, end1920 = tuple(coords.TECH_TREE), tuple(coords.END_TURN)
    dist1920 = ((tech1920[0] - end1920[0]) ** 2 + (tech1920[1] - end1920[1]) ** 2) ** 0.5
    assert dist1920 >= coords.DOCK_MIN_SEP, (tech1920, end1920, dist1920)
    coords.reset()


def test_find_move_marks_1280():
    im = Image.new("RGB", (1280, 800), (20, 20, 20))
    px = im.load()
    for y in range(int(700 * 800 / 1200), int(820 * 800 / 1200)):
        for x in range(int(900 * 1280 / 1920), int(1100 * 1280 / 1920)):
            px[x, y] = (58, 199, 249)
    cx, cy = round(1242 * 1280 / 1920), round(600 * 800 / 1200)
    for y in range(cy - 8, cy + 8):
        for x in range(cx - 10, cx + 10):
            px[x, y] = (146, 204, 255)
    marks = find_move_marks(__import__("numpy").asarray(im).astype("int16"))
    assert marks, marks
    assert all(abs(m["x"] - cx) < 25 and abs(m["y"] - cy) < 25 for m in marks)


def test_entities_on_synthetic_map():
    im = Image.new("RGB", (1920, 1200), (30, 40, 28))
    d = ImageDraw.Draw(im)
    # own (Bardur) city: grey building + white nameplate + gold level-star
    d.rectangle((880, 360, 940, 420), fill=(90, 85, 80))
    d.rectangle((860, 430, 960, 448), fill=(240, 240, 235))
    d.ellipse((942, 432, 956, 446), fill=(224, 188, 63))
    # enemy (Oumaji) city
    d.rectangle((1180, 360, 1240, 420), fill=(210, 180, 60))
    d.rectangle((1160, 430, 1260, 448), fill=(245, 245, 240))
    # enemy (Vengir) city — Disrof-like wine building
    d.rectangle((480, 360, 540, 420), fill=(70, 35, 80))
    d.rectangle((460, 430, 560, 448), fill=(245, 245, 240))
    # enemy (Vengir) frost plate + magenta roof (Moonrise Disrof: ~180 grey, not 215 white)
    d.rectangle((200, 355, 270, 418), fill=(90, 85, 80))
    d.rectangle((214, 348, 256, 378), fill=(150, 74, 144))
    d.rectangle((186, 428, 308, 450), fill=(180, 180, 178))
    d.rectangle((200, 434, 206, 444), fill=(40, 40, 40))
    # unit HP bar
    d.rectangle((700, 500, 728, 505), fill=(90, 210, 50))
    # village hut
    d.rectangle((548, 628, 564, 644), fill=(168, 118, 62))
    arr = __import__("numpy").asarray(im).astype("int16")
    units = find_units(arr)
    cities = find_cities(arr)
    os.environ.pop("POLYTOPIA_VILLAGES", None)
    assert find_villages(arr, cities) == []
    os.environ["POLYTOPIA_VILLAGES"] = "1"
    villages = find_villages(arr, cities)
    os.environ.pop("POLYTOPIA_VILLAGES", None)
    assert units, units
    assert any(abs(u["x"] - 714) < 25 for u in units)
    assert all("tribe" in u and "owner" in u for u in units)
    assert len(cities) >= 2, cities
    owners = {c["owner"] for c in cities}
    assert "own" in owners and "enemy" in owners, cities
    tribes = {c["tribe"] for c in cities}
    assert "bardur" in tribes, cities
    assert "vengir" in tribes, cities
    yellow = [c for c in cities if 1160 <= int(c["x"]) <= 1260]
    assert not yellow, yellow  # bright oumaji-looking ghost, no purple/gold-on-dark
    assert all(isinstance(c.get("tile"), list) and len(c["tile"]) == 2 for c in cities), cities
    assert all(str(c["id"]).startswith("c") and "_" in str(c["id"])[1:] for c in cities), cities
    frost = [c for c in cities if 170 <= int(c["x"]) <= 310]
    assert frost and all(c["tribe"] == "vengir" and c["owner"] == "enemy" for c in frost), cities
    assert villages, villages
    assert classify_tribe((210, 180, 60)) == "oumaji"
    assert classify_tribe((90, 85, 80)) == "bardur"
    assert classify_tribe((70, 35, 80)) == "vengir"
    assert classify_tribe((150, 74, 144)) == "vengir"  # magenta roof
    assert classify_tribe((30, 40, 28)) == "unknown"  # grass ≠ Bardur city
    assert classify_tribe((58, 199, 249)) == "unknown"  # water ≠ Imperius
    assert classify_tribe((143, 255, 255)) == "unknown"

    # Water foam + cyan is not a city; a dirt field is not a village.
    noisy = Image.new("RGB", (1920, 1200), (30, 40, 28))
    nd = ImageDraw.Draw(noisy)
    nd.rectangle((400, 200, 900, 500), fill=(58, 199, 249))
    nd.rectangle((520, 310, 610, 322), fill=(240, 245, 250))
    nd.rectangle((700, 700, 980, 980), fill=(160, 120, 70))
    narr = __import__("numpy").asarray(noisy).astype("int16")
    assert not find_cities(narr), find_cities(narr)
    os.environ["POLYTOPIA_VILLAGES"] = "1"
    assert not find_villages(narr, []), find_villages(narr, [])
    os.environ.pop("POLYTOPIA_VILLAGES", None)


def test_capture_and_recruit_targets():
    coords.reset()
    coords.set_frame(1920, 1200)
    assert capture_target({"overlay": {}, "ready": {}, "unit": {}}) is None
    obs = {
        "overlay": {"do_it_blobs": [{"x": 10, "y": 20, "n": 40}], "train_blobs": []},
        "ready": {"capture": True},
        "unit": {"capture": True},
    }
    ocr_hit = capture_target(obs)
    assert ocr_hit is not None and ocr_hit != (10, 20), ocr_hit
    assert ocr_hit[1] >= int(1200 * 0.78), ocr_hit  # UNIT_CROP, not map DO IT
    panel = {
        "layout": {"frame": [1920, 1200]},
        "overlay": {
            "capture_blobs": [
                {"x": 220, "y": 1100, "n": 200},
                {"x": 400, "y": 400, "n": 800},
            ]
        },
        "ready": {"capture": True},
        "unit": {"capture": True},
    }
    assert capture_target(panel) == (220, 1100)
    blob_no_ocr = {
        "layout": {"frame": [1920, 1200]},
        "overlay": {"capture_blobs": [{"x": 220, "y": 1100, "n": 200}]},
        "ready": {},
        "unit": {},
    }
    assert capture_target(blob_no_ocr) == (220, 1100)
    noisy = {
        "overlay": {"do_it_blobs": [{"x": 400, "y": 400, "n": 800}], "capture_blobs": [{"x": 400, "y": 400, "n": 800}]},
        "ready": {},
        "unit": {},
    }
    assert capture_target(noisy) is None
    obs = {
        "overlay": {"train_blobs": [{"x": 3, "y": 4}], "do_it_blobs": []},
        "ready": {"train": True},
        "unit": {"train": True},
    }
    # Map-colored TRAIN at (3,4) must not be clicked — UNIT_CROP only.
    panel_hit = recruit_target(obs)
    assert panel_hit is not None and panel_hit[1] >= int(1200 * 0.78) and panel_hit[0] <= int(1920 * 0.52), panel_hit
    coords.set_frame(1280, 800)
    ocr_only = {
        "overlay": {"train_blobs": [], "train_pixel": False},
        "ready": {"train": True},
        "unit": {"train": True},
        "layout": {"frame": [1280, 800]},
    }
    # OCR Train with no blob → UNIT_CROP panel, never scaled map TRAIN.
    panel_xy = recruit_target(ocr_only)
    assert panel_xy is not None and panel_xy[1] >= int(800 * 0.78) and panel_xy[0] <= int(1280 * 0.52)
    plate = _city_click_points({"x": 10, "y": 40, "plate_y": 80, "kind": "city"})
    assert plate[0] == (10, 80, "plate")
    assert plate[1] == (10, 40, "building")


def test_mcp_lists_local_tools():
    names = [t["name"] for t in TOOLS]
    for n in (
        "local_observe",
        "local_select_unit",
        "local_move_to",
        "local_attack",
        "local_capture",
        "local_recruit",
        "local_end_turn",
        "local_click",
        "local_tech",
        "local_diff",
        "local_snapshot",
        "local_step",
    ):
        assert n in names, n
    click = next(t for t in TOOLS if t["name"] == "local_click")
    assert "name" in click["inputSchema"]["properties"]
    assert "required" not in click["inputSchema"]
    move = next(t for t in TOOLS if t["name"] == "local_move_to")
    assert "city_id" in move["inputSchema"]["properties"]
    assert "required" not in move["inputSchema"]
    listed = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert listed["result"]["tools"][0]["name"].startswith("local_")


def test_find_window_ignores_wrapper_and_chrome():
    from polytopia_api.driver import is_game_bin, is_game_window, pick_game_process
    from polytopia_api.server import canonicalize_path

    lines = [
        "1111 /home/foo/.steam/steam.sh -applaunch 874390 Polytopia.x86_64",
        "2222 /home/foo/.steam/steamapps/common/The Battle of Polytopia/Polytopia.x86_64",
        "3333 /opt/google/chrome/chrome --app=http://polytopia.local",
        "4444 grep Polytopia.x86_64",
    ]
    hit = pick_game_process(lines)
    assert hit is not None and hit[0] == 2222
    assert is_game_bin(hit[1])
    assert not is_game_bin(lines[0].split(None, 1)[1])
    assert is_game_window("Polytopia", 'WM_CLASS(STRING) = "Polytopia", "Polytopia"')
    assert not is_game_window(
        "polytopia.local",
        'WM_CLASS(STRING) = "Google-chrome", "Google-chrome"',
    )
    assert canonicalize_path("/select_unit") == "/select-unit"
    assert canonicalize_path("/end_turn/") == "/end-turn"
    assert canonicalize_path("/move-to") == "/move-to"
    assert canonicalize_path("/diff") == "/diff"
    assert canonicalize_path("/snapshot") == "/snapshot"


def test_plan_priorities():
    coords.reset()
    coords.set_frame(1920, 1200)
    base_overlay = {
        "back_lit": False,
        "do_it_pixel": False,
        "train_pixel": False,
        "do_it_blobs": [],
        "train_blobs": [],
    }
    unit_idle = {
        "unit": None,
        "can_move": False,
        "no_actions": False,
        "harvest": False,
        "clear_forest": False,
        "train": False,
        "settings": False,
        "capture": False,
        "capture_soon": False,
        "village": False,
    }

    obs = {
        "hud": {"stars": 7, "turn": 10, "score": 1800, "income": 7},
        "unit": {**unit_idle, "clear_forest": True},
        "overlay": dict(base_overlay),
        "confirm_ready": False,
        "move_marks": [],
        "fruit": [],
    }
    assert plan(obs)["name"] == "back"

    obs = {
        "hud": {"stars": 5, "turn": 8},
        "unit": {**unit_idle, "harvest": True},
        "overlay": {**base_overlay, "do_it_pixel": True},
        "confirm_ready": True,
        "move_marks": [],
        "fruit": [],
    }
    assert plan(obs)["name"] == "confirm_harvest"

    obs = {
        "hud": {"stars": 8, "turn": 9},
        "unit": {**unit_idle, "capture": True},
        "overlay": {
            **base_overlay,
            "capture_blobs": [{"x": 220, "y": 1100, "n": 200}],
        },
        "confirm_ready": True,
        "move_marks": [],
        "fruit": [],
    }
    cap = plan(obs)
    assert cap["name"] == "capture"
    assert cap["y"] >= 900

    obs = {
        "hud": {"stars": 8, "turn": 9},
        "unit": {**unit_idle, "can_move": True, "unit": "warrior"},
        "overlay": dict(base_overlay),
        "confirm_ready": False,
        "move_marks": [{"x": 1240, "y": 600, "n": 40}],
        "fruit": [{"x": 1412, "y": 790, "n": 12}],
    }
    assert plan(obs)["name"] == "move"

    obs = {
        "hud": {"stars": 7, "turn": 10},
        "unit": dict(unit_idle),
        "overlay": dict(base_overlay),
        "confirm_ready": False,
        "move_marks": [],
        "fruit": [{"x": 1412, "y": 790, "n": 12}],
    }
    assert plan(obs)["name"] == "click_fruit"

    obs = {
        "hud": {"stars": 1, "turn": 8},
        "unit": {**unit_idle, "no_actions": True, "unit": "warrior"},
        "overlay": dict(base_overlay),
        "confirm_ready": False,
        "move_marks": [],
        "fruit": [{"x": 1412, "y": 790, "n": 12}],
    }
    assert plan(obs)["name"] == "deselect"

    obs = {
        "hud": {"stars": 24, "turn": 12},
        "unit": dict(unit_idle),
        "overlay": dict(base_overlay),
        "confirm_ready": False,
        "move_marks": [],
        "fruit": [],
    }
    assert plan(obs)["name"] == "blocked_end_turn"

    obs = {
        "hud": {"stars": 1, "turn": 8},
        "unit": dict(unit_idle),
        "overlay": dict(base_overlay),
        "confirm_ready": False,
        "move_marks": [],
        "fruit": [],
    }
    assert plan(obs)["name"] == "end_turn"


def test_hud_star_split():
    im = Image.new("RGB", (400, 60), (20, 20, 30))
    d = ImageDraw.Draw(im)
    d.ellipse((180, 15, 210, 45), fill=(220, 180, 40))
    x = _hud_star_x(im)
    assert x is not None and abs(x - 195) < 20, x


def test_stable_entity_ids():
    prev = [{"id": "c0", "x": 100, "y": 100}, {"id": "c1", "x": 400, "y": 200}]
    nxt = [{"id": "c9", "x": 108, "y": 104}, {"id": "c8", "x": 900, "y": 900}]
    out = stabilize(nxt, prev, max_dist=56)
    assert out[0]["id"] == "c0" and out[0]["stable"]
    assert out[1]["id"] == "c8" and not out[1]["stable"]
    sticky = stabilize(
        [{"id": "c9", "x": 204, "y": 82, "tribe": "oumaji", "owner": "enemy", "name": None}],
        [{"id": "c1", "x": 200, "y": 80, "tribe": "vengir", "owner": "enemy", "name": "Disrof"}],
    )
    assert sticky[0]["id"] == "c1"
    assert sticky[0]["tribe"] == "vengir"
    assert sticky[0]["name"] == "Disrof"
    grid_prev = [
        {"id": "c5_3", "tile": [5, 3], "x": 200, "y": 80, "tribe": "vengir", "owner": "enemy", "name": "Disrof"},
        {"id": "c8_4", "tile": [8, 4], "x": 400, "y": 120, "tribe": "bardur", "owner": "own"},
    ]
    grid_nxt = [
        {"id": "c8_4", "tile": [8, 4], "x": 402, "y": 118, "tribe": "bardur", "owner": "own"},
        {"id": "c5_3", "tile": [5, 3], "x": 206, "y": 84, "tribe": "oumaji", "owner": "enemy", "name": None},
    ]
    hashed = stabilize(grid_nxt, grid_prev)
    by_id = {c["id"]: c for c in hashed}
    assert set(by_id) == {"c5_3", "c8_4"}
    assert by_id["c5_3"]["tribe"] == "vengir" and by_id["c5_3"]["name"] == "Disrof"
    # Grid-hash jitter must keep the first sticky id (live: c22_17 vs c22_18).
    jitter = stabilize(
        [{"id": "c22_18", "tile": [22, 18], "x": 202, "y": 88, "tribe": "vengir", "owner": "enemy"}],
        [{"id": "c22_17", "tile": [22, 17], "x": 200, "y": 80, "tribe": "vengir", "owner": "enemy", "name": "Disrof"}],
    )
    assert jitter[0]["id"] == "c22_17"
    assert jitter[0]["name"] == "Disrof"
    missing = stabilize(
        [],
        [{"id": "c22_17", "tile": [22, 17], "x": 200, "y": 80, "tribe": "vengir", "owner": "enemy"}],
        keep_missing=True,
    )
    assert missing and missing[0]["id"] == "c22_17" and missing[0].get("seen") is False
    from polytopia_api.observe import lookup, register_cities, register_units, reset as reset_obs, stabilize_units
    reset_obs()
    register_cities([{"id": "c22_17", "kind": "city", "x": 200, "y": 80, "tile": [22, 17]}])
    assert lookup({"cities": [], "cities_enemy": []}, "c22_17")["id"] == "c22_17"
    register_units([{"id": "u2", "kind": "unit", "x": 140, "y": 80, "owner": "own"}])
    assert lookup({"units": []}, "u2")["id"] == "u2"
    held = stabilize_units(
        [{"x": 400, "y": 200, "kind": "unit"}],
        [{"id": "u5", "x": 140, "y": 80, "kind": "unit", "owner": "own"}],
        keep_missing=True,
    )
    by_u = {u["id"]: u for u in held}
    assert "u5" in by_u and by_u["u5"].get("seen") is False
    assert any(u.get("seen") is True and u["id"] != "u5" for u in held)
    reset_obs()


def test_gold_windows_are_not_oumaji():
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import sample_patch_tribe

    im = Image.new("RGB", (1920, 1200), (30, 40, 28))
    d = ImageDraw.Draw(im)
    d.rectangle((200, 355, 270, 418), fill=(90, 85, 80))
    d.rectangle((214, 348, 256, 378), fill=(150, 74, 144))
    d.rectangle((220, 382, 232, 396), fill=(224, 188, 63))
    d.rectangle((186, 428, 308, 450), fill=(180, 180, 178))
    d.ellipse((278, 432, 296, 448), fill=(224, 188, 63))
    d.rectangle((200, 434, 206, 444), fill=(40, 40, 40))
    arr = as_rgb(im)
    cities = find_cities(arr)
    frost = [c for c in cities if 170 <= int(c["x"]) <= 310]
    assert frost and all(c["tribe"] == "vengir" and c["owner"] == "enemy" for c in frost), cities
    tribe, _ = sample_patch_tribe(arr, 235, 380, radius=14)
    assert tribe == "vengir", tribe


def test_disrof_gold_windows_without_magenta_sample():
    """Live Disrof: gold lamps fill the radius=14 patch; roof sits higher.

    The 397c23d lum filter treated that sample as bright Oumaji and dropped
    the city, so cities_enemy went empty while Disrof was on screen.
    """
    from polytopia_api.detect import as_rgb

    im = Image.new("RGB", (1920, 1200), (30, 40, 28))
    d = ImageDraw.Draw(im)
    d.rectangle((200, 355, 270, 418), fill=(90, 85, 80))
    d.rectangle((208, 360, 262, 410), fill=(224, 188, 63))
    d.rectangle((186, 428, 308, 450), fill=(180, 180, 178))
    d.ellipse((278, 432, 296, 448), fill=(224, 188, 63))
    d.rectangle((200, 434, 206, 444), fill=(40, 40, 40))
    arr = as_rgb(im)
    cities = find_cities(arr)
    frost = [c for c in cities if 170 <= int(c["x"]) <= 310]
    assert frost, cities
    assert all(c["tribe"] == "vengir" and c["owner"] == "enemy" for c in frost), frost
    assert all(c.get("tile") for c in frost)
    assert all(float(c.get("confidence") or 0) >= 0.78 for c in frost), frost
    assert all(
        "gold_lamps" in (c.get("evidence") or []) or "magenta_roof" in (c.get("evidence") or [])
        for c in frost
    )


def test_bardur_own_not_eaten_by_vengir_frost():
    """Live M20: Bardur Ufla/Orkork/Grugru must stay cities_own; frost FPs ≠ Disrof."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    im = Image.new("RGB", (1920, 1200), (30, 40, 28))
    d = ImageDraw.Draw(im)

    def bardur_city(x, y):
        d.rectangle((x, y, x + 60, y + 60), fill=(90, 85, 80))
        d.rectangle((x - 20, y + 70, x + 80, y + 92), fill=(180, 180, 178))
        d.ellipse((x + 62, y + 74, x + 78, y + 90), fill=(224, 188, 63))  # plate star, not windows
        d.rectangle((x, y + 76, x + 6, y + 88), fill=(40, 40, 40))

    bardur_city(400, 300)
    bardur_city(880, 360)
    bardur_city(1180, 520)
    # Real Disrof: dark stone + magenta roof + gold lamps + frost plate
    d.rectangle((200, 355, 270, 418), fill=(90, 85, 80))
    d.rectangle((214, 348, 256, 378), fill=(150, 74, 144))
    d.rectangle((220, 382, 232, 396), fill=(224, 188, 63))
    d.rectangle((186, 428, 308, 450), fill=(180, 180, 178))
    d.ellipse((278, 432, 296, 448), fill=(224, 188, 63))
    d.rectangle((200, 434, 206, 444), fill=(40, 40, 40))
    # Frost fragments with no roof / no lamps (the ~7 vengir FPs)
    for x0 in (520, 700, 1020, 1300, 640, 1480):
        d.rectangle((x0, 180, x0 + 90, 202), fill=(180, 180, 178))
        d.rectangle((x0 + 10, 186, x0 + 16, 196), fill=(40, 40, 40))

    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    enemy = mapped["cities_enemy"]
    assert len(own) == 3, mapped["cities"]
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    vengir = [c for c in enemy if c.get("tribe") == "vengir"]
    assert 1 <= len(vengir) <= 2, vengir
    assert any(
        "magenta_roof" in (c.get("evidence") or []) or "gold_lamps" in (c.get("evidence") or [])
        for c in vengir
    ), vengir
    assert all(float(c.get("confidence") or 0) >= 0.78 for c in vengir), vengir


def _draw_bardur_city(d, cx: int, plate_y: int) -> None:
    """Moonrise Bardur: grey wood + frost plate + gold star (Ufla/Orkork/Grugru)."""
    d.rectangle((cx - 30, plate_y - 78, cx + 30, plate_y - 18), fill=(90, 85, 80))
    d.rectangle((cx - 52, plate_y - 10, cx + 52, plate_y + 12), fill=(180, 180, 178))
    d.rectangle((cx - 40, plate_y - 4, cx - 34, plate_y + 6), fill=(40, 40, 40))
    d.ellipse((cx + 34, plate_y - 6, cx + 50, plate_y + 8), fill=(224, 188, 63))


def _draw_orkork_split_plate(d, cx: int = 557, plate_y: int = 667) -> None:
    """One Orkork longhouse whose frost plate splits ~35px (c15_24 + c16_24)."""
    d.rectangle((cx - 32, plate_y - 78, cx + 32, plate_y - 18), fill=(90, 85, 80))
    d.rectangle((527, 657, 555, 677), fill=(180, 180, 178))
    d.rectangle((533, 662, 538, 672), fill=(40, 40, 40))
    d.rectangle((560, 667, 588, 687), fill=(180, 180, 178))
    d.ellipse((572, 670, 586, 684), fill=(224, 188, 63))


def _draw_mid_map_own_phantom(d, cx: int = 441, plate_y: int = 480) -> None:
    """Sparse mountain wood + frost streak + fruit gold — live c12_16 @(441,455)."""
    d.rectangle((cx - 28, plate_y - 50, cx + 28, plate_y - 12), fill=(110, 112, 118))
    d.rectangle((cx - 10, plate_y - 40, cx + 4, plate_y - 18), fill=(90, 85, 80))
    d.rectangle((cx - 48, plate_y - 8, cx + 48, plate_y + 10), fill=(180, 180, 178))
    d.ellipse((cx + 28, plate_y - 4, cx + 42, plate_y + 8), fill=(224, 188, 63))


def _draw_unit_frost_phantom(d, cx: int, plate_y: int) -> None:
    """Unit-sized wood + frost + gold star — live c17_14 @(610,379), warm_n≈992."""
    d.rectangle((cx - 16, plate_y - 40, cx + 16, plate_y - 8), fill=(90, 85, 80))
    d.rectangle((cx - 36, plate_y - 6, cx + 36, plate_y + 10), fill=(180, 180, 178))
    d.ellipse((cx + 18, plate_y - 4, cx + 32, plate_y + 8), fill=(224, 188, 63))
    d.rectangle((cx - 28, plate_y - 2, cx - 22, plate_y + 6), fill=(40, 40, 40))


def _draw_disrof_city(d, cx: int, plate_y: int, lamps: bool = True, roof: bool = True) -> None:
    """Moonrise Disrof: dark stone + magenta roof and/or gold window lamps."""
    d.rectangle((cx - 35, plate_y - 82, cx + 35, plate_y - 20), fill=(90, 85, 80))
    if roof:
        d.rectangle((cx - 22, plate_y - 96, cx + 22, plate_y - 64), fill=(150, 74, 144))
    if lamps:
        d.rectangle((cx - 16, plate_y - 52, cx - 4, plate_y - 36), fill=(224, 188, 63))
        d.rectangle((cx + 6, plate_y - 52, cx + 18, plate_y - 36), fill=(224, 188, 63))
    d.rectangle((cx - 62, plate_y - 10, cx + 62, plate_y + 12), fill=(180, 180, 178))
    d.rectangle((cx - 48, plate_y - 2, cx - 42, plate_y + 8), fill=(40, 40, 40))
    d.ellipse((cx + 42, plate_y - 6, cx + 58, plate_y + 8), fill=(224, 188, 63))


def test_bardur_gold_star_stays_own():
    """Plate gold star must not flip Ufla-class Bardur into cities_enemy."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map, sample_patch_tribe

    im = Image.new("RGB", (1920, 1200), (30, 40, 28))
    d = ImageDraw.Draw(im)
    _draw_bardur_city(d, 910, 439)
    arr = as_rgb(im)
    mapped = observe_map(arr)
    own = mapped["cities_own"]
    assert own, mapped["cities"]
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    assert mapped["cities_enemy"] == []
    tribe, _ = sample_patch_tribe(arr, 910, 390, radius=14)
    assert tribe == "bardur", tribe


def test_m20_bardur_own_and_one_disrof():
    """Live M20 shape: three Bardur cities on screen + one Disrof, not ~7 Vengir FPs.

    Frost fragments and a nameless 'Arch' plate must not become cities_enemy.
    """
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map, sample_patch_tribe

    im = Image.new("RGB", (1920, 1200), (30, 40, 28))
    d = ImageDraw.Draw(im)
    _draw_bardur_city(d, 640, 520)   # Orkork-class
    _draw_bardur_city(d, 910, 439)   # Ufla-class
    _draw_bardur_city(d, 1180, 640)  # Grugru-class
    _draw_disrof_city(d, 247, 439)
    # frost / snow fragments that used to become extra Vengir + ghost Arch
    for x0, y0 in ((90, 210), (340, 260), (520, 180), (760, 280), (1040, 200), (1400, 310)):
        d.rectangle((x0, y0, x0 + 110, y0 + 22), fill=(180, 180, 178))
        d.rectangle((x0 + 12, y0 + 6, x0 + 18, y0 + 16), fill=(40, 40, 40))
    # ghost plate with no building (OCR used to invent "Arch")
    d.rectangle((430, 720, 560, 742), fill=(180, 180, 178))
    d.rectangle((444, 726, 450, 736), fill=(35, 35, 35))
    d.rectangle((458, 726, 464, 736), fill=(35, 35, 35))
    arr = as_rgb(im)
    mapped = observe_map(arr)
    own = mapped["cities_own"]
    enemy = mapped["cities_enemy"]
    assert len(own) == 3, own
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    assert all(c.get("city_id") == c.get("id") and str(c["id"]).startswith("c") for c in mapped["cities"]), mapped["cities"]
    xs_own = sorted(int(c["x"]) for c in own)
    assert any(abs(x - 640) < 40 for x in xs_own), own
    assert any(abs(x - 910) < 40 for x in xs_own), own
    assert any(abs(x - 1180) < 40 for x in xs_own), own
    vengir = [c for c in enemy if c.get("tribe") == "vengir"]
    assert 1 <= len(vengir) <= 2, enemy
    assert all(c["owner"] == "enemy" for c in vengir)
    assert any(abs(int(c["x"]) - 247) < 50 for c in vengir), vengir
    names = {str(c.get("name") or "").lower() for c in mapped["cities"]}
    assert "arch" not in names, mapped["cities"]
    tribe, _ = sample_patch_tribe(arr, 247, 380, radius=14)
    assert tribe == "vengir", tribe


def _draw_disrof_plate_windows(d, cx: int, plate_y: int) -> None:
    """Disrof gold lamps on the frost plate itself (0bfb6f7 contract), no magenta roof."""
    d.rectangle((cx - 35, plate_y - 75, cx + 35, plate_y - 18), fill=(90, 85, 80))
    d.rectangle((cx - 62, plate_y - 10, cx + 62, plate_y + 12), fill=(180, 180, 178))
    d.rectangle((cx - 48, plate_y - 2, cx - 42, plate_y + 8), fill=(40, 40, 40))
    # Windows in the left/center of the plate — not the right-side level star.
    d.rectangle((cx - 28, plate_y - 6, cx - 14, plate_y + 6), fill=(224, 188, 63))
    d.rectangle((cx - 4, plate_y - 6, cx + 10, plate_y + 6), fill=(224, 188, 63))


def test_frost_plate_gold_windows_stay_enemy_not_own():
    """Frost-plate gold windows stay Vengir enemy; must not invent cities_own.

    Live 0bfb6f7 contract: gold on the nameplate is Disrof lamps, not a Bardur
    city. A right-side gold star still keeps Ufla/Orkork/Grugru as own.
    """
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    im = Image.new("RGB", (1920, 1200), (30, 40, 28))
    d = ImageDraw.Draw(im)
    _draw_bardur_city(d, 640, 520)
    _draw_bardur_city(d, 910, 439)
    _draw_bardur_city(d, 1180, 640)
    _draw_disrof_plate_windows(d, 247, 439)
    # Left-shifted frost cluster (gold punches the plate); still enemy, not own.
    d.rectangle((200, 155, 270, 218), fill=(90, 85, 80))
    d.rectangle((186, 228, 308, 250), fill=(180, 180, 178))
    d.rectangle((198, 232, 212, 246), fill=(224, 188, 63))
    d.rectangle((228, 232, 242, 246), fill=(224, 188, 63))
    d.rectangle((200, 234, 206, 244), fill=(40, 40, 40))
    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    enemy = mapped["cities_enemy"]
    assert len(own) == 3, mapped["cities"]
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    assert all(c.get("city_id") == c.get("id") for c in mapped["cities"]), mapped["cities"]
    xs_own = [int(c["x"]) for c in own]
    assert all(abs(x - 247) > 50 for x in xs_own), own
    vengir = [c for c in enemy if c.get("tribe") == "vengir"]
    assert 1 <= len(vengir) <= 2, mapped["cities"]
    assert all(c["owner"] == "enemy" for c in vengir)
    assert any(abs(int(c["x"]) - 247) < 50 for c in vengir), vengir
    assert any("gold_lamps" in (c.get("evidence") or []) for c in vengir), vengir
    assert all(float(c.get("confidence") or 0) >= 0.78 for c in vengir), vengir


def test_1280_bardur_own_and_plate_window_disrof():
    """Live 1280×800: Bardur cities_own stay non-empty; plate-gold Disrof is enemy."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    im = Image.new("RGB", (1280, 800), (30, 40, 28))
    d = ImageDraw.Draw(im)
    s = 1280 / 1920

    def sc(n: float) -> int:
        return int(round(n * s))

    for cx, cy in ((640, 520), (910, 439), (1180, 640)):
        d.rectangle((sc(cx - 30), sc(cy - 78), sc(cx + 30), sc(cy - 18)), fill=(90, 85, 80))
        d.rectangle((sc(cx - 52), sc(cy - 10), sc(cx + 52), sc(cy + 12)), fill=(180, 180, 178))
        d.rectangle((sc(cx - 40), sc(cy - 4), sc(cx - 34), sc(cy + 6)), fill=(40, 40, 40))
        d.ellipse((sc(cx + 34), sc(cy - 6), sc(cx + 50), sc(cy + 8)), fill=(224, 188, 63))
    cx, py = sc(247), sc(439)
    d.rectangle((cx - sc(35), py - sc(75), cx + sc(35), py - sc(18)), fill=(90, 85, 80))
    d.rectangle((cx - sc(62), py - sc(10), cx + sc(62), py + sc(12)), fill=(180, 180, 178))
    d.rectangle((cx - sc(48), py - sc(2), cx - sc(42), py + sc(8)), fill=(40, 40, 40))
    d.rectangle((cx - sc(28), py - sc(6), cx - sc(14), py + sc(6)), fill=(224, 188, 63))
    d.rectangle((cx - sc(4), py - sc(6), cx + sc(10), py + sc(6)), fill=(224, 188, 63))
    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    enemy = mapped["cities_enemy"]
    assert own, mapped["cities"]
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    vengir = [c for c in enemy if c.get("tribe") == "vengir"]
    assert vengir, mapped["cities"]
    assert all(c["owner"] == "enemy" for c in vengir)
    assert any("gold_lamps" in (c.get("evidence") or []) for c in vengir), vengir
    assert all(c.get("city_id") == c.get("id") for c in mapped["cities"])


def test_attack_timeout_skips_second_observe():
    from unittest.mock import patch

    from polytopia_api import commands

    city = {
        "id": "c5_3",
        "kind": "city",
        "x": 200,
        "y": 80,
        "tile": [5, 3],
        "tile_xy": [200, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }
    state = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "units": [{"id": "u2", "x": 140, "y": 80}],
        "unit": {"no_actions": False, "unit": "warrior"},
        "attack_marks": [{"x": 201, "y": 81, "n": 20}],
    }
    clock = {"t": 0.0}
    observe_n = {"n": 0}

    def fake_time():
        return clock["t"]

    def fake_select(**kwargs):
        clock["t"] = 8.0
        return {"ok": True, "unit": state["unit"]}

    def fake_observe():
        observe_n["n"] += 1
        raise AssertionError("attack must not observe again after the 8s budget")

    with patch.object(commands.time, "time", fake_time), patch.object(
        commands, "remember", return_value=state
    ), patch.object(commands, "observe", side_effect=fake_observe), patch.object(
        commands, "select_unit", side_effect=lambda **k: fake_select()
    ), patch.object(commands, "_click"), patch.object(commands, "_sleep"):
        r = commands.attack(from_id="u2", city_id="c5_3")
    assert r["ok"] is False and r["reason"] == "timeout"
    assert r["elapsed_s"] < 12
    assert observe_n["n"] == 0


def test_snapshot_alerts():
    prev = {
        "hud": {"turn": 18, "stars": 5, "income": 7},
        "units": [{"id": "u0", "x": 10, "y": 10, "tribe": "bardur", "owner": "own"}],
        "cities_own": [{"id": "c0", "x": 40, "y": 40, "tribe": "bardur", "owner": "own"}],
        "cities_enemy": [{"id": "c1", "x": 200, "y": 80, "tribe": "oumaji", "owner": "enemy"}],
        "fog_edge": [{"id": "g0", "x": 300, "y": 100}],
    }
    jumped = {
        "hud": {"turn": 22, "stars": 12, "income": 7},
        "units": [{"id": "u0", "x": 40, "y": 12, "tribe": "bardur", "owner": "own"}],
        "cities_own": [{"id": "c0", "x": 40, "y": 40, "tribe": "bardur", "owner": "own"}],
        "cities_enemy": [],
        "fog_edge": [{"id": "g0", "x": 310, "y": 100}],
        "screenshot": "/tmp/nope.png",
    }
    d = snapshot_diff(prev, jumped, reason="end_turn")
    assert d["turn_delta"] == 4
    assert any(a["kind"] == "turn_jump" for a in d["alerts"]), d["alerts"]
    assert d["units"]["moved"] and d["units"]["moved"][0]["id"] == "u0"
    assert d["cities_enemy"]["removed"] == ["c1"]
    same_turn = {
        "hud": {"turn": 18, "stars": 12, "income": 7},
        "units": prev["units"],
        "cities_own": prev["cities_own"],
        "cities_enemy": prev["cities_enemy"],
        "fog_edge": prev["fog_edge"],
    }
    d2 = snapshot_diff(prev, same_turn, reason="observe")
    assert d2["turn_delta"] == 0
    assert any(a["kind"] == "stars_income_without_turn" for a in d2["alerts"]), d2["alerts"]


def test_snapshot_refuses_stale_turn():
    import tempfile

    from polytopia_api import snapshot as snapmod

    d = tempfile.mkdtemp()
    os.environ["POLYTOPIA_SNAPSHOTS"] = d
    snapmod.reset()
    stale = {
        "hud": {
            "turn": 4,
            "stars": 1,
            "score": 100,
            "stale": True,
            "stale_fields": ["turn"],
            "turn_trusted": False,
        },
        "units": [],
        "cities_own": [],
        "cities_enemy": [],
        "screenshot": None,
    }
    r = snapmod.save(stale, reason="end_turn")
    assert r["ok"] is False
    assert str(r["tag"]).startswith("untrusted-")
    assert not (Path(d) / "T4.json").exists()

    r2 = snapmod.save(stale, reason="end_turn", turn=24)
    assert r2["ok"] and r2["tag"] == "T24" and r2["turn_source"] == "label"
    assert (Path(d) / "T24.json").is_file()

    r3 = snapmod.save(stale, reason="end_turn")
    assert r3["ok"] and r3["tag"] == "T25" and r3["turn_source"] == "clock"
    assert (Path(d) / "T25.json").is_file()
    snapmod.reset()
    os.environ.pop("POLYTOPIA_SNAPSHOTS", None)


def test_observe_hud_untrusts_ocr_behind_floor():
    import tempfile

    from polytopia_api import snapshot as snapmod

    d = tempfile.mkdtemp()
    os.environ["POLYTOPIA_SNAPSHOTS"] = d
    snapmod.reset()
    labeled = {
        "hud": {"turn": 26, "stars": 10, "score": 4550, "stale": False, "turn_trusted": True},
        "units": [],
        "cities_own": [],
        "cities_enemy": [],
        "screenshot": None,
    }
    assert snapmod.save(labeled, reason="manual", turn=26)["ok"]

    behind = {
        "turn": 2,
        "turn_ocr": 2,
        "stars": 10,
        "score": 4550,
        "stale": False,
        "stale_fields": [],
        "turn_trusted": True,
        "missing": [],
    }
    snapmod.apply_turn_floor(behind)
    assert behind["turn_trusted"] is False
    assert behind["turn"] is None
    assert behind["turn_ocr"] == 2
    assert behind["turn_floor"] == 26
    assert behind["turn_untrusted_reason"] == "ocr_turn_behind"
    assert "turn" in behind["stale_fields"]

    ok = {
        "turn": 26,
        "turn_ocr": 26,
        "stale": False,
        "stale_fields": [],
        "turn_trusted": True,
        "missing": [],
    }
    snapmod.apply_turn_floor(ok)
    assert ok["turn_trusted"] is True and ok["turn"] == 26

    nxt = {
        "turn": 27,
        "turn_ocr": 27,
        "stale": False,
        "stale_fields": [],
        "turn_trusted": True,
        "missing": [],
    }
    snapmod.apply_turn_floor(nxt)
    assert nxt["turn_trusted"] is True and nxt["turn"] == 27
    snapmod.reset()
    os.environ.pop("POLYTOPIA_SNAPSHOTS", None)


def test_observe_hud_floor_from_tfile_without_latest():
    import tempfile

    from polytopia_api import snapshot as snapmod

    d = tempfile.mkdtemp()
    os.environ["POLYTOPIA_SNAPSHOTS"] = d
    snapmod.reset()
    (Path(d) / "T26.json").write_text('{"turn":26,"turn_trusted":true}\n', encoding="utf-8")
    behind = {
        "turn": 2,
        "turn_ocr": 2,
        "stale": False,
        "stale_fields": [],
        "turn_trusted": True,
        "missing": [],
    }
    snapmod.apply_turn_floor(behind)
    assert behind["turn_floor"] == 26
    assert behind["turn_trusted"] is False
    assert behind["turn"] is None
    snapmod.reset()
    empty = tempfile.mkdtemp()
    os.environ["POLYTOPIA_SNAPSHOTS"] = empty
    os.environ["POLYTOPIA_TURN"] = "26"
    env_behind = {
        "turn": 2,
        "turn_ocr": 2,
        "stale": False,
        "stale_fields": [],
        "turn_trusted": True,
        "missing": [],
    }
    snapmod.apply_turn_floor(env_behind)
    assert env_behind["turn_floor"] == 26 and env_behind["turn_trusted"] is False
    os.environ.pop("POLYTOPIA_TURN", None)
    os.environ.pop("POLYTOPIA_SNAPSHOTS", None)


def test_snapshot_dir_ignores_empty_cwd():
    import tempfile

    from polytopia_api import snapshot as snapmod

    os.environ.pop("POLYTOPIA_SNAPSHOTS", None)
    snapmod.reset()
    here = Path.cwd()
    other = tempfile.mkdtemp()
    try:
        os.chdir(other)
        d = snapmod.snapshot_dir()
        assert d.is_absolute()
        assert d != Path(other) / "turn_snapshot"
        assert d.name == "turn_snapshot"
    finally:
        os.chdir(here)
    os.environ.pop("POLYTOPIA_SNAPSHOTS", None)


def test_doit_blobs_ignore_map_noise():
    from polytopia_api.detect import find_doit_buttons

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (20, 20, 20))
    d = ImageDraw.Draw(im)
    # specks that used to become dozens of capture_blobs
    for i in range(20):
        x, y = 420 + (i % 8) * 40, 300 + (i // 8) * 50
        d.rectangle((x, y, x + 6, y + 6), fill=(10, 155, 250))
    # real confirm pill near DO_IT
    d.rectangle((700, 420, 790, 490), fill=(8, 155, 250))
    blobs = find_doit_buttons(as_rgb(im))
    assert len(blobs) <= 4, blobs
    assert blobs, blobs
    assert all(b["n"] >= 70 for b in blobs)
    assert any(abs(b["x"] - 745) < 50 and abs(b["y"] - 455) < 50 for b in blobs)


def test_attack_garrison_hp_and_on_city():
    from polytopia_api.combat import unit_on_city
    from polytopia_api.commands import _hp_snapshot

    city = {"id": "c1", "kind": "city", "x": 200, "y": 80, "name": "Disrof", "n": 999}
    obs = {
        "layout": {"frame": [1280, 800]},
        "units": [{"id": "u3", "kind": "unit", "x": 204, "y": 84, "hp": "full", "n": 40}],
        "cities": [city],
        "cities_enemy": [city],
    }
    hp = _hp_snapshot(obs, "c1", 200, 80)
    assert hp.get("id") == "u3" and hp.get("hp") == "full"
    assert hp.get("n") == 40
    gone = {"layout": {"frame": [1280, 800]}, "units": [], "cities": [city]}
    after = _hp_snapshot(gone, "c1", 200, 80)
    assert after == {}
    on = unit_on_city(obs["units"], city, (1280, 800), {})
    assert on["ok"] is True
    far = unit_on_city([{"id": "u9", "x": 800, "y": 400}], city, (1280, 800), {})
    assert far["ok"] is False and far["reason"] == "not_standing_on_city"
    adjacent = unit_on_city(
        [{"id": "u1", "x": 164, "y": 80, "owner": "own"}],
        city,
        (1280, 800),
        {},
    )
    assert adjacent["ok"] is False, adjacent
    on_sprite = unit_on_city(
        [{"id": "u1", "x": 200, "y": 62, "owner": "own"}],
        city,
        (1280, 800),
        {},
        max_hex=0.8,
    )
    assert on_sprite["ok"] is True, on_sprite
    panel = unit_on_city([], city, (1280, 800), {"capture": True})
    assert panel["ok"] is False
    assert panel["reason"] == "not_standing_on_city"


def test_find_train_blob_1280():
    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (20, 20, 20))
    d = ImageDraw.Draw(im)
    d.rectangle((700, 500, 780, 560), fill=(10, 122, 204))
    blobs = find_train_buttons(as_rgb(im))
    assert blobs, blobs
    assert abs(blobs[0]["x"] - 740) < 30
    assert abs(blobs[0]["y"] - 530) < 30
    from polytopia_api.detect import find_train_panel_buttons, overlay_flags

    panel = Image.new("RGB", (1280, 800), (20, 20, 20))
    pd = ImageDraw.Draw(panel)
    pd.rectangle((160, 720, 240, 770), fill=(10, 122, 204))
    panel_blobs = find_train_panel_buttons(as_rgb(panel))
    assert panel_blobs, panel_blobs
    assert abs(panel_blobs[0]["x"] - 200) < 30
    assert panel_blobs[0]["y"] >= 700
    flags = overlay_flags(as_rgb(panel))
    assert flags["train_blobs"], flags["train_blobs"]
    brand = Image.new("RGB", (1280, 800), (20, 20, 20))
    bd = ImageDraw.Draw(brand)
    bd.rectangle((150, 725, 250, 775), fill=(0, 153, 255))
    brand_blobs = find_train_panel_buttons(as_rgb(brand))
    assert brand_blobs, brand_blobs
    assert abs(brand_blobs[0]["x"] - 200) < 30
    assert brand_blobs[0]["y"] >= 700


def test_attack_marks_and_strike():
    from polytopia_api.combat import can_strike
    from polytopia_api.detect import find_attack_marks, is_attack_rgb

    assert is_attack_rgb(210, 60, 55)
    assert not is_attack_rgb(90, 210, 50)
    im = Image.new("RGB", (1280, 800), (20, 20, 20))
    d = ImageDraw.Draw(im)
    d.rectangle((600, 400, 640, 430), fill=(210, 60, 55))
    marks = find_attack_marks(as_rgb(im))
    assert marks, marks
    raft = can_strike("raft", (100, 400), (600, 410), "city", (1280, 800), marks)
    assert raft["ok"] is False and raft["reason"] == "naval_no_land"
    melee = can_strike("warrior", (560, 410), (620, 415), "city", (1280, 800), marks)
    assert melee["ok"] is True


def test_win_path_move_capture_attack():
    from unittest.mock import patch

    from polytopia_api import commands

    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c5_3",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [5, 3],
        "tile_xy": [200, 80],
        "tribe": "vengir",
        "owner": "enemy",
        "n": 999,
    }
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    base = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_own": [],
        "cities_enemy": [city],
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    on_mark = {"id": "m0", "x": 202, "y": 81, "n": 50}
    far_mark = {"id": "m1", "x": 400, "y": 400, "n": 50}
    after_move = {
        **base,
        "move_marks": [on_mark, far_mark],
        "units": [{"id": "u1", "x": 201, "y": 80, "owner": "own"}],
        "unit": {"can_move": False, "capture": True, "no_actions": False},
        "ready": {"capture": True},
    }
    selected = {"ok": True, "unit": {"can_move": True, "no_actions": False, "unit": "warrior"}}

    with patch.object(commands, "remember", return_value={
        **base,
        "move_marks": [on_mark, far_mark],
        "units": [{"id": "u1", "x": 120, "y": 80}],
        "unit": {"can_move": True, "no_actions": False},
    }), patch.object(commands, "observe", return_value=after_move), patch.object(
        commands, "select_unit", return_value=selected
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u1", city_id="c5_3")
    assert r["ok"] is True and r["stood_on_city"] is True
    assert clicks == [(200, 94)], clicks

    clicks.clear()
    no_mark_state = {
        **base,
        "move_marks": [far_mark],
        "units": [{"id": "u1", "x": 120, "y": 80}],
        "unit": {"can_move": True},
    }
    with patch.object(commands, "remember", return_value=no_mark_state), patch.object(
        commands, "observe", return_value=no_mark_state
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u1", city_id="c5_3")
    assert r["ok"] is False and r["reason"] in {"no_mark_on_city_tile", "out_of_range"}
    assert r["stood_on_city"] is False
    assert "move_marks" in r and r.get("city_xy")
    assert r.get("suggested_tile_xy") == [200, 80]
    assert clicks == []

    panel_blob = {"x": 180, "y": 740, "n": 200}
    far_unit = {
        **base,
        "units": [{"id": "u1", "x": 800, "y": 400, "owner": "own"}],
        "unit": {"capture": True, "no_actions": False},
        "ready": {"capture": True},
        "overlay": {"capture_blobs": [panel_blob, {"x": 400, "y": 400, "n": 800}]},
    }
    with patch.object(commands, "remember", return_value=far_unit), patch.object(
        commands, "observe", return_value=far_unit
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.capture(city_id="c5_3")
    assert r["ok"] is False and r["reason"] == "not_standing_on_city"
    assert r["captured"] is False
    assert clicks == []

    on_city = {
        **base,
        "units": [{"id": "u1", "x": 201, "y": 80, "owner": "own"}],
        "unit": {"capture": True, "no_actions": False},
        "ready": {"capture": True},
        "overlay": {"capture_blobs": [panel_blob]},
    }
    captured_state = {
        **on_city,
        "cities_enemy": [],
        "cities_own": [{**city, "owner": "own"}],
        "cities": [{**city, "owner": "own"}],
    }
    with patch.object(commands, "remember", return_value=on_city), patch.object(
        commands, "observe", return_value=captured_state
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.capture(city_id="c5_3")
    assert r["ok"] is True and r["captured"] is True
    assert clicks == [(180, 740)], clicks

    clicks.clear()
    stale_marks = [{"x": 202, "y": 81, "n": 40}]
    idle = {
        **base,
        "units": [{"id": "u9", "kind": "unit", "x": 201, "y": 82, "hp": 10, "n": 40, "owner": "enemy"}],
        "unit": {"no_actions": True, "unit": "warrior", "can_move": False},
        "attack_marks": stale_marks,
    }
    with patch.object(commands, "remember", return_value=idle), patch.object(
        commands, "observe", return_value=idle
    ), patch.object(
        commands, "select_unit", return_value={"ok": True, "unit": idle["unit"]}
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.attack(from_id="u2", city_id="c5_3")
    assert r["ok"] is False and r["reason"] == "no_actions"
    assert r["elapsed_s"] < 8
    assert clicks == []

    garrison = {"id": "u9", "kind": "unit", "x": 201, "y": 82, "hp": 10, "n": 40, "owner": "enemy"}
    mark_garrison = {"x": 203, "y": 83, "n": 40}
    mark_plate = {"x": 200, "y": 140, "n": 40}
    before_atk = {
        **base,
        "units": [garrison, {"id": "u2", "x": 140, "y": 80, "owner": "own"}],
        "unit": {"no_actions": False, "unit": "warrior", "can_move": False},
        "attack_marks": [mark_plate, mark_garrison],
    }
    after_atk = {
        **before_atk,
        "units": [{**garrison, "hp": 7, "n": 28}, {"id": "u2", "x": 140, "y": 80, "owner": "own"}],
    }
    with patch.object(commands, "remember", return_value=before_atk), patch.object(
        commands, "observe", return_value=after_atk
    ), patch.object(
        commands, "select_unit",
        return_value={"ok": True, "unit": before_atk["unit"], "attack_marks": before_atk["attack_marks"]},
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.attack(from_id="u2", city_id="c5_3")
    assert r["ok"] is True and r["hp_dropped"] is True
    assert r["elapsed_s"] < 8
    assert r["hp_before"]["id"] == "u9" and r["hp_before"]["hp"] == 10
    assert r["hp_after"]["hp"] == 7
    assert r.get("garrison_dead") is False
    assert r.get("next") == "attack"
    assert clicks, clicks
    assert clicks[0] == (200, 94), clicks
    assert (203, 83) not in clicks


def test_city_id_stays_resolvable_after_observe_drops_it():
    """Live smoke: first /attack resolved c22_17; later calls forgot it."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c22_17",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [22, 17],
        "tile_xy": [200, 80],
        "tribe": "vengir",
        "owner": "enemy",
        "n": 400,
        "confidence": 0.78,
        "evidence": ["frost_plate", "magenta_roof", "gold_lamps"],
    }
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    blank = {
        "layout": {"frame": [1280, 800]},
        "cities": [],
        "cities_own": [],
        "cities_enemy": [],
        "units": [
            {"id": "u9", "x": 180, "y": 80, "owner": "own"},
            {"id": "u8", "x": 201, "y": 80, "owner": "enemy"},
        ],
        "unit": {"no_actions": False, "unit": "warrior", "can_move": True},
        "attack_marks": [{"x": 400, "y": 400, "n": 20}],
        "move_marks": [{"x": 400, "y": 400, "n": 20}],
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    selected = {"ok": True, "unit": blank["unit"]}
    with patch.object(commands, "remember", return_value=blank), patch.object(
        commands, "observe", return_value=blank
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        atk = commands.attack(from_id="u9", city_id="c22_17")
        mv = commands.move_to(from_id="u9", city_id="c22_17")
        cap = commands.capture(city_id="c22_17")
    assert atk.get("reason") != "need to_id/city_id or x,y", atk
    assert atk.get("dest") and atk["dest"]["id"] == "c22_17"
    assert atk["ok"] is False and atk["reason"] == "no_mark_on_target"
    assert mv.get("reason") != "need x,y or city_id/to_id", mv
    assert mv["ok"] is False and mv["reason"] == "city_occupied"
    assert mv.get("next") == "attack"
    assert mv.get("garrison_id") == "u8"
    assert mv.get("next_call", {}).get("path") == "/attack"
    assert mv.get("suggested_tile_xy") == [200, 80]
    assert cap.get("reason") != "no entity c22_17", cap
    assert cap["ok"] is False and cap["reason"] == "city_occupied"
    assert cap.get("next") == "attack"
    assert clicks == []

    on_city = {
        **blank,
        "cities": [city],
        "cities_enemy": [city],
        "units": [{"id": "u9", "x": 201, "y": 80, "owner": "own"}],
        "unit": {"capture": True, "no_actions": False, "unit": "warrior"},
        "ready": {"capture": True},
        "overlay": {"capture_blobs": [{"x": 180, "y": 740, "n": 200}]},
        "attack_marks": [{"x": 203, "y": 83, "n": 40}],
        "move_marks": [{"x": 202, "y": 81, "n": 50}],
    }
    after_atk = {
        **on_city,
        "units": [
            {"id": "u8", "kind": "unit", "x": 201, "y": 82, "hp": 7, "n": 28, "owner": "enemy"},
            {"id": "u9", "x": 140, "y": 80, "owner": "own"},
        ],
    }
    before_atk = {
        **on_city,
        "units": [
            {"id": "u8", "kind": "unit", "x": 201, "y": 82, "hp": 10, "n": 40, "owner": "enemy"},
            {"id": "u9", "x": 140, "y": 80, "owner": "own"},
        ],
        "unit": {"no_actions": False, "unit": "warrior", "can_move": False},
    }
    captured_state = {
        **on_city,
        "cities_enemy": [],
        "cities_own": [{**city, "owner": "own", "tribe": "bardur"}],
        "cities": [{**city, "owner": "own", "tribe": "bardur"}],
    }
    clicks.clear()
    with patch.object(commands, "remember", return_value=before_atk), patch.object(
        commands, "observe", return_value=after_atk
    ), patch.object(
        commands, "select_unit",
        return_value={"ok": True, "unit": before_atk["unit"]},
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        atk = commands.attack(from_id="u9", city_id="c22_17")
    assert atk["ok"] is True and atk["hp_dropped"] is True, atk
    clicks.clear()
    after_move = {
        **on_city,
        "units": [{"id": "u9", "x": 201, "y": 80, "owner": "own"}],
        "unit": {"can_move": False, "capture": True, "no_actions": False},
        "ready": {"capture": True},
    }
    with patch.object(commands, "remember", return_value=on_city), patch.object(
        commands, "observe", return_value=after_move
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        mv = commands.move_to(from_id="u9", city_id="c22_17")
    assert mv["ok"] is True and mv["stood_on_city"] is True, mv
    clicks.clear()
    with patch.object(commands, "remember", return_value=after_move), patch.object(
        commands, "observe", return_value=captured_state
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        cap = commands.capture(city_id="c22_17")
    assert cap["ok"] is True and cap["captured"] is True, cap
    reset_obs()


def test_vengir_fps_do_not_accumulate_across_frames():
    """Live: unnamed Vengir drifted 2→4→7 because keep_missing held frost FPs."""
    from polytopia_api.entities import session_cities
    from polytopia_api.observe import stabilize

    prev = [
        {
            "id": "c1_1", "tribe": "vengir", "owner": "enemy", "name": None,
            "confidence": 0.64, "evidence": ["frost_plate"], "x": 100, "y": 100,
            "tile": [1, 1], "n": 50, "seen": True,
        },
        {
            "id": "c2_2", "tribe": "vengir", "owner": "enemy", "name": None,
            "confidence": 0.64, "evidence": ["frost_plate"], "x": 200, "y": 100,
            "tile": [2, 2], "n": 50, "seen": True,
        },
        {
            "id": "c22_17", "tribe": "vengir", "owner": "enemy", "name": None,
            "confidence": 0.78, "evidence": ["magenta_roof", "gold_lamps", "frost_plate"],
            "x": 300, "y": 100, "tile": [22, 17], "n": 400, "seen": True,
        },
        {
            "id": "c8_4", "tribe": "bardur", "owner": "own", "name": None,
            "confidence": 0.72, "evidence": ["frost_plate", "gold_star"],
            "x": 500, "y": 200, "tile": [8, 4], "n": 200, "seen": True,
        },
    ]
    nxt = [
        {
            "id": "c3_3", "tribe": "vengir", "owner": "enemy", "name": None,
            "confidence": 0.64, "evidence": ["frost_plate"], "x": 400, "y": 100,
            "tile": [3, 3], "n": 50,
        },
        {
            "id": "c4_4", "tribe": "vengir", "owner": "enemy", "name": None,
            "confidence": 0.64, "evidence": ["frost_plate"], "x": 600, "y": 100,
            "tile": [4, 4], "n": 50,
        },
        {
            "id": "c22_17", "tribe": "vengir", "owner": "enemy", "name": None,
            "confidence": 0.78, "evidence": ["magenta_roof"], "x": 302, "y": 102,
            "tile": [22, 17], "n": 400,
        },
        {
            "id": "c8_4", "tribe": "bardur", "owner": "own", "x": 502, "y": 198,
            "tile": [8, 4], "n": 200, "confidence": 0.72,
        },
    ]
    out = session_cities(stabilize(nxt, prev, keep_missing=True))
    ids = {c["id"] for c in out}
    assert "c22_17" in ids
    assert "c8_4" in ids
    assert "c1_1" not in ids and "c2_2" not in ids
    unnamed = [c for c in out if c.get("tribe") == "vengir" and not c.get("name")]
    frost_only = [c for c in unnamed if "magenta_roof" not in (c.get("evidence") or []) and "gold_lamps" not in (c.get("evidence") or [])]
    assert len(frost_only) <= 2, frost_only
    assert any(c["id"] == "c22_17" for c in unnamed)


def test_twelve_own_cities_do_not_wipe_enemy():
    """Live T44: cities_own had 12 and cities_enemy went [] while Disrot/Rzgórst/Grugru were on screen.

    The combined [:12] after _cap_vengir(others first) dropped every Vengir.
    """
    from polytopia_api.entities import session_cities
    from polytopia_api.observe import stabilize

    own = [
        {
            "id": f"c{i}_{i}",
            "city_id": f"c{i}_{i}",
            "tile": [i, i],
            "tile_xy": [80 + (i % 6) * 180, 120 + (i // 6) * 90],
            "x": 80 + (i % 6) * 180,
            "y": 120 + (i // 6) * 90,
            "plate_y": 140 + (i // 6) * 90,
            "w": 40,
            "h": 12,
            "tribe": "bardur",
            "owner": "own",
            "confidence": 0.72,
            "evidence": ["frost_plate", "gold_star"],
            "n": 200,
        }
        for i in range(12)
    ]
    enemy = [
        {
            "id": "c20_4",
            "city_id": "c20_4",
            "tile": [20, 4],
            "tile_xy": [220, 360],
            "x": 220,
            "y": 360,
            "plate_y": 390,
            "w": 80,
            "tribe": "vengir",
            "owner": "enemy",
            "name": None,
            "confidence": 0.78,
            "evidence": ["magenta_roof", "gold_lamps", "frost_plate"],
            "n": 400,
        },
        {
            "id": "c24_8",
            "city_id": "c24_8",
            "tile": [24, 8],
            "tile_xy": [520, 400],
            "x": 520,
            "y": 400,
            "plate_y": 430,
            "w": 80,
            "tribe": "vengir",
            "owner": "enemy",
            "name": None,
            "confidence": 0.78,
            "evidence": ["magenta_roof", "frost_plate"],
            "n": 360,
        },
        {
            "id": "c28_6",
            "city_id": "c28_6",
            "tile": [28, 6],
            "tile_xy": [820, 380],
            "x": 820,
            "y": 380,
            "plate_y": 410,
            "w": 80,
            "tribe": "vengir",
            "owner": "enemy",
            "name": None,
            "confidence": 0.78,
            "evidence": ["gold_lamps", "frost_plate"],
            "n": 340,
        },
    ]
    out = session_cities(stabilize(own + enemy, None, keep_missing=True))
    own_out = [c for c in out if c.get("owner") == "own"]
    enemy_out = [c for c in out if c.get("owner") == "enemy"]
    assert len(own_out) == 12, own_out
    ids = {c["id"] for c in enemy_out}
    assert ids >= {"c20_4", "c24_8", "c28_6"}, enemy_out
    assert all(c.get("tribe") == "vengir" for c in enemy_out)
    assert all(c.get("city_id") == c.get("id") and c.get("tile") and c.get("tile_xy") for c in enemy_out)


def test_t44_dark_tile_vengir_stay_cities_enemy():
    """Live T44 1280×800: Vengir cities on dark tiles must stay in cities_enemy
    even when the frame already has a full Bardur cities_own list.
    """
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (22, 24, 28))
    d = ImageDraw.Draw(im)
    # Twelve Bardur cities (the T44 cities_own fill). Space plates so they
    # do not merge (gap > 64px).
    own_pts = [
        (120, 180), (340, 180), (560, 180), (780, 180),
        (120, 360), (340, 360), (560, 360), (780, 360),
        (120, 540), (340, 540), (560, 540), (780, 540),
    ]
    for cx, py in own_pts:
        _draw_bardur_city(d, cx, py)
    # Three Disrof-class on dark Vengir ground (Disrot / Rzgórst / Grugru).
    enemy_pts = [(1040, 200), (1160, 380), (1040, 560)]
    for cx, py in enemy_pts:
        d.rectangle((cx - 70, py - 110, cx + 70, py + 24), fill=(28, 22, 32))
        _draw_disrof_city(d, cx, py)
    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    enemy = mapped["cities_enemy"]
    assert len(own) >= 10, mapped["cities"]
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    vengir = [c for c in enemy if c.get("tribe") == "vengir"]
    assert len(vengir) >= 3, mapped["cities"]
    assert all(c["owner"] == "enemy" for c in vengir)
    assert all(c.get("city_id") == c.get("id") and c.get("tile") and c.get("tile_xy") for c in vengir), vengir
    xs = [int(c["x"]) for c in vengir]
    assert any(abs(x - 1040) < 80 for x in xs), vengir
    assert any(abs(x - 1160) < 80 for x in xs), vengir


def test_dark_magenta_roof_classifies_vengir():
    from polytopia_api.entities import classify_tribe

    assert classify_tribe((70, 35, 80)) == "vengir"
    assert classify_tribe((150, 74, 144)) == "vengir"
    # T44 dark-tile roof — used to miss min(r,b) >= 40.
    assert classify_tribe((36, 20, 48)) == "vengir"
    assert classify_tribe((90, 85, 80)) == "bardur"
    from polytopia_api.entities import _is_magenta_roof_rgb
    assert _is_magenta_roof_rgb(150, 74, 144)
    assert _is_magenta_roof_rgb(70, 35, 80)
    assert _is_magenta_roof_rgb(36, 20, 48)
    assert _is_magenta_roof_rgb(48, 36, 58)  # garrison covering the roof
    # Wine-shadow on Bardur wood — used to count as magenta_roof (T46 undercount).
    assert not _is_magenta_roof_rgb(88, 70, 82)
    assert not _is_magenta_roof_rgb(82, 68, 80)


def _own_city(i: int, y: int = 200) -> dict:
    return {
        "id": f"c{i}_0",
        "city_id": f"c{i}_0",
        "tribe": "bardur",
        "owner": "own",
        "x": 80 + (i % 6) * 180,
        "y": y + (i // 6) * 160,
        "tile": [i, 0],
        "tile_xy": [80 + (i % 6) * 180, y + (i // 6) * 160],
        "n": 200,
        "confidence": 0.72,
        "evidence": ["frost_plate", "gold_star"],
        "w": 48,
        "h": 20,
        "plate_y": y + (i // 6) * 160,
    }


def _enemy_city(cid: str, tile: list[int], x: int, y: int, **extra) -> dict:
    ev = extra.pop("evidence", ["magenta_roof", "gold_lamps", "frost_plate"])
    return {
        "id": cid,
        "city_id": cid,
        "tribe": "vengir",
        "owner": "enemy",
        "name": None,
        "confidence": 0.78,
        "evidence": ev,
        "x": x,
        "y": y,
        "tile": tile,
        "tile_xy": [x, y],
        "n": extra.pop("n", 400),
        "w": extra.pop("w", 90),
        "h": extra.pop("h", 22),
        **extra,
    }


def test_session_cities_keeps_sticky_enemy_when_own_is_full():
    """Mid-fight miss of Disrof must not vanish behind 12 own cities."""
    from polytopia_api.entities import session_cities
    from polytopia_api.observe import stabilize

    prev = [_own_city(i) for i in range(12)] + [_enemy_city("c13_10", [13, 10], 880, 310)]
    nxt = [_own_city(i) for i in range(12)]
    out = session_cities(stabilize(nxt, prev, keep_missing=True))
    ids = {c["id"] for c in out}
    assert "c13_10" in ids, out
    hit = next(c for c in out if c["id"] == "c13_10")
    assert hit["owner"] == "enemy" and hit["tribe"] == "vengir"
    assert hit.get("seen") is False
    assert hit.get("tile") == [13, 10]


def test_twelve_own_cities_do_not_drop_enemy_or_flood_frost():
    """T45/T46: 12 Bardur + Disrof-class + frost swarm. Real enemies stay; frost ≤2."""
    from polytopia_api.entities import session_cities

    own = [_own_city(i) for i in range(12)]
    enemies = [
        _enemy_city("c20_5", [20, 5], 900, 300),
        _enemy_city("c22_8", [22, 8], 1100, 520, evidence=["magenta_roof", "frost_plate"]),
        _enemy_city("c18_3", [18, 3], 700, 640, evidence=["gold_lamps", "frost_plate"]),
    ]
    frost = [
        _enemy_city(f"c3_{i}", [3, i], 40 + i * 30, 80, evidence=["frost_plate"], n=50, confidence=0.64)
        for i in range(5)
    ]
    out = session_cities(own + enemies + frost)
    enemy = [c for c in out if c.get("owner") == "enemy"]
    ids = {c["id"] for c in enemy}
    assert {"c20_5", "c22_8", "c18_3"} <= ids, out
    assert all(c.get("tribe") == "vengir" and c.get("tile") for c in enemy if c["id"] in {"c20_5", "c22_8", "c18_3"})
    assert len([c for c in out if c.get("owner") == "own"]) == 12
    frost_only = [
        c
        for c in enemy
        if "magenta_roof" not in (c.get("evidence") or [])
        and "gold_lamps" not in (c.get("evidence") or [])
    ]
    assert len(frost_only) <= 2, frost_only


def test_late_domination_observe_keeps_visible_vengir():
    """Pixel contract: 12 Bardur on screen + Disrof/Rzgórst still fill cities_enemy."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    im = Image.new("RGB", (1920, 1200), (30, 40, 28))
    d = ImageDraw.Draw(im)
    bardur_xy = [
        (180, 200), (420, 200), (660, 200), (900, 200),
        (180, 420), (420, 420), (660, 420), (900, 420),
        (180, 640), (420, 640), (660, 640), (900, 640),
    ]
    for cx, py in bardur_xy:
        _draw_bardur_city(d, cx, py)
    _draw_disrof_city(d, 1500, 280)   # Disrof / Disrot
    _draw_disrof_city(d, 1740, 520)   # Rzgórst-class
    for x0, y0 in ((80, 160), (1100, 170), (1280, 360), (1400, 700)):
        d.rectangle((x0, y0, x0 + 110, y0 + 22), fill=(180, 180, 178))
        d.rectangle((x0 + 12, y0 + 6, x0 + 18, y0 + 16), fill=(40, 40, 40))
    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    enemy = mapped["cities_enemy"]
    assert len(own) >= 12, mapped["cities"]
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    vengir = [c for c in enemy if c.get("tribe") == "vengir"]
    assert vengir, mapped["cities"]
    assert all(c["owner"] == "enemy" for c in vengir)
    assert all(c.get("city_id") == c.get("id") and c.get("tile") and c.get("tile_xy") for c in vengir), vengir
    xs = [int(c["x"]) for c in vengir]
    assert any(abs(x - 1500) < 80 for x in xs), vengir
    assert any(abs(x - 1740) < 80 for x in xs), vengir
    strong = [
        c
        for c in vengir
        if "magenta_roof" in (c.get("evidence") or []) or "gold_lamps" in (c.get("evidence") or [])
    ]
    assert len(strong) >= 2, vengir
    frost_only = [c for c in vengir if c not in strong]
    assert len(frost_only) <= 2, frost_only


def test_nearby_vengir_cities_do_not_merge():
    """Disrof + Rzgórst within the old 80px vengir merge must stay two cities."""
    from polytopia_api.entities import session_cities

    a = _enemy_city("c20_5", [20, 5], 900, 300, w=40, plate_y=300)
    b = _enemy_city(
        "c21_6",
        [21, 6],
        970,
        320,
        w=40,
        plate_y=320,
        evidence=["magenta_roof", "frost_plate"],
    )
    out = session_cities([a, b])
    ids = {c["id"] for c in out if c.get("owner") == "enemy"}
    assert ids >= {"c20_5", "c21_6"}, out


def test_nearby_bardur_not_eaten_by_disrof():
    """Live e1f3eed: cities_own=2 while many Bardur sat ~1 hex from Vengir FPs."""
    from polytopia_api.entities import session_cities
    from polytopia_api.observe import stabilize

    own = _own_city(4, y=300)
    own["id"] = own["city_id"] = "c12_10"
    own["tile"] = [12, 10]
    own["x"], own["y"] = 820, 300
    own["tile_xy"] = [820, 300]
    own["plate_y"] = 300
    own["w"] = 48
    enemy = _enemy_city("c13_10", [13, 10], 880, 310, w=90, plate_y=310)
    out = session_cities([own, enemy])
    own_ids = {c["id"] for c in out if c.get("owner") == "own"}
    enemy_ids = {c["id"] for c in out if c.get("owner") == "enemy"}
    assert "c12_10" in own_ids, out
    assert "c13_10" in enemy_ids, out
    sticky = stabilize([own], [enemy], keep_missing=True)
    own_hit = [c for c in sticky if c.get("id") == "c12_10" or c.get("owner") == "own"]
    assert own_hit and own_hit[0]["owner"] == "own" and own_hit[0]["tribe"] == "bardur", sticky
    assert own_hit[0].get("id") != "c13_10"


def test_crowded_bardur_own_and_vengir_enemy():
    """P0: many visible Bardur stay cities_own; Disrof stays cities_enemy (not 2 vs 7)."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (22, 24, 28))
    d = ImageDraw.Draw(im)
    own_pts = [
        (140, 170), (300, 170), (460, 170), (620, 170),
        (140, 340), (300, 340), (460, 340), (620, 340),
        (140, 510), (300, 510), (460, 510), (620, 510),
    ]
    for cx, py in own_pts:
        _draw_bardur_city(d, cx, py)
    _draw_disrof_city(d, 1040, 220)
    _draw_disrof_city(d, 1160, 400, lamps=False)
    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    enemy = mapped["cities_enemy"]
    assert len(own) >= 10, mapped["cities"]
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    vengir = [c for c in enemy if c.get("tribe") == "vengir"]
    assert len(vengir) >= 2, mapped["cities"]
    assert all(c["owner"] == "enemy" for c in vengir)


def test_bardur_wood_shadow_stays_own():
    """Live T46: wine-shadow on Bardur wood labeled many own cities as Vengir."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (22, 24, 28))
    d = ImageDraw.Draw(im)
    own_pts = [
        (140, 170), (300, 170), (460, 170), (620, 170),
        (140, 340), (300, 340), (460, 340), (620, 340),
        (140, 510), (300, 510), (460, 510),
    ]
    for cx, py in own_pts:
        _draw_bardur_city(d, cx, py)
        d.rectangle((cx - 18, py - 70, cx - 6, py - 40), fill=(88, 70, 82))
        d.rectangle((cx + 8, py - 64, cx + 18, py - 36), fill=(82, 68, 80))
    _draw_disrof_city(d, 1040, 220)
    _draw_disrof_city(d, 1160, 400)
    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    enemy = mapped["cities_enemy"]
    assert len(own) >= 8, mapped["cities"]
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    vengir = [c for c in enemy if c.get("tribe") == "vengir"]
    assert len(vengir) >= 2, mapped["cities"]
    assert all(c["owner"] == "enemy" for c in vengir)


def test_three_bardur_not_hud_fog_c18_1():
    """Three real longhouses stay cities_own; HUD/fog c18_1 @ y≈45 does not."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map, session_cities

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (22, 24, 28))
    d = ImageDraw.Draw(im)
    # Three real Bardur longhouses (not the T46 two-city board).
    _draw_bardur_city(d, 280, 280)
    _draw_bardur_city(d, 520, 420)
    _draw_bardur_city(d, 760, 560)
    # HUD stars + frost chrome at the top edge (hashed as c18_1 / gy=1).
    _draw_bardur_city(d, 648, 64)
    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    assert len(own) == 3, mapped["cities"]
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    assert all(int(c.get("y") or 0) > 96 for c in own), own
    assert all(int((c.get("tile") or [0, 99])[1]) > 1 for c in own), own
    xs = sorted(int(c["x"]) for c in own)
    assert any(abs(x - 280) < 40 for x in xs), own
    assert any(abs(x - 520) < 40 for x in xs), own
    assert any(abs(x - 760) < 40 for x in xs), own

    real = _own_city(7, y=300)
    real["id"] = real["city_id"] = "c7_16"
    real["tile"] = [7, 16]
    real_b = _own_city(12, y=420)
    real_b["id"] = real_b["city_id"] = "c12_16"
    real_b["tile"] = [12, 16]
    real_c = _own_city(16, y=560)
    real_c["id"] = real_c["city_id"] = "c16_24"
    real_c["tile"] = [16, 24]
    ghost = {
        "id": "c18_1", "city_id": "c18_1", "tribe": "bardur", "owner": "own",
        "name": None, "confidence": 0.72,
        "evidence": ["frost_plate", "gold_star", "bardur_wood"],
        "x": 648, "y": 45, "plate_y": 64, "tile": [18, 1], "n": 400, "w": 90,
        "seen": True,
    }
    out = session_cities([real, real_b, real_c, ghost])
    ids = {c["id"] for c in out if c.get("owner") == "own"}
    assert ids == {"c7_16", "c12_16", "c16_24"}, out
    assert "c18_1" not in ids


def test_two_bardur_dedupe_orkork_split_and_mid_phantom():
    """Live T46 after #31/#32: Game Stats Bardur 2; observe listed 4–5.

    Leftover junk: Orkork c15_24@(541,657)+c16_24@(574,667) ~35px, sparse
    c12_16@(441,455), and unit-frost c17_14@(610,379) with warm_n≈992 (beats
    the 500 floor). Grugru is Vengir, not a third Bardur city.
    """
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map, session_cities

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (22, 24, 28))
    d = ImageDraw.Draw(im)
    _draw_bardur_city(d, 250, 280)  # Bufla
    _draw_orkork_split_plate(d)
    _draw_mid_map_own_phantom(d)
    _draw_unit_frost_phantom(d, 610, 400)  # c17_14-class, warm enough to beat 500
    _draw_bardur_city(d, 648, 64)  # HUD/fog c18_1
    _draw_disrof_city(d, 980, 260)
    _draw_disrof_city(d, 1100, 420)
    _draw_disrof_city(d, 1040, 580)
    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    assert len(own) == 2, mapped["cities"]
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    xs = sorted(int(c["x"]) for c in own)
    ys = [int(c["y"]) for c in own]
    assert any(abs(x - 250) < 40 for x in xs), own
    assert any(abs(x - 557) < 50 for x in xs), own
    assert all(not (abs(int(c["x"]) - 441) < 30 and abs(int(c["y"]) - 455) < 40) for c in own), own
    assert all(abs(int(c["x"]) - 610) > 40 or abs(int(c.get("plate_y") or c["y"]) - 400) > 40 for c in own), own
    assert all(int(c.get("y") or 0) > 96 for c in own), own
    vengir = [c for c in mapped["cities_enemy"] if c.get("tribe") == "vengir"]
    assert vengir, mapped["cities"]
    # Close pair must not survive as two own ids ~35px apart.
    for i, a in enumerate(own):
        for b in own[i + 1 :]:
            d2 = (int(a["x"]) - int(b["x"])) ** 2 + (int(a["y"]) - int(b["y"])) ** 2
            assert d2 > 48 * 48, own

    left = _own_city(15, y=657)
    left["id"] = left["city_id"] = "c15_24"
    left["tile"] = [15, 24]
    left["x"], left["y"] = 541, 657
    left["tile_xy"] = [541, 657]
    left["plate_y"] = 657
    left["n"] = 884
    left["w"] = 56
    right = _own_city(16, y=667)
    right["id"] = right["city_id"] = "c16_24"
    right["tile"] = [16, 24]
    right["x"], right["y"] = 574, 667
    right["tile_xy"] = [574, 667]
    right["plate_y"] = 667
    right["n"] = 90
    right["w"] = 29
    bufla = _own_city(7, y=265)
    bufla["id"] = bufla["city_id"] = "c7_16"
    bufla["tile"] = [7, 16]
    bufla["x"], bufla["y"] = 253, 265
    phantom = {
        "id": "c12_16", "city_id": "c12_16", "tribe": "bardur", "owner": "own",
        "name": None, "confidence": 0.72,
        "evidence": ["frost_plate", "gold_star", "bardur_wood"],
        "x": 441, "y": 455, "plate_y": 480, "tile": [12, 16], "n": 1144, "w": 61,
        "warm_n": 345, "seen": True,
    }
    unit_frost = {
        "id": "c17_14", "city_id": "c17_14", "tribe": "bardur", "owner": "own",
        "name": None, "confidence": 0.72,
        "evidence": ["frost_plate", "gold_star", "bardur_wood"],
        "x": 610, "y": 379, "plate_y": 400, "tile": [17, 14], "n": 893, "w": 59,
        "warm_n": 992, "wood_w": 25, "wood_h": 23, "wood_dens": 0.17, "seen": True,
    }
    out = session_cities([bufla, left, right, phantom, unit_frost])
    own_ids = {c["id"] for c in out if c.get("owner") == "own"}
    assert "c7_16" in own_ids, out
    assert not ({"c15_24", "c16_24"} <= own_ids), out
    assert "c12_16" not in own_ids and "c17_14" not in own_ids, out
    assert len([c for c in out if c.get("owner") == "own"]) == 2, out


def test_unit_frost_c17_14_not_cities_own():
    """#32 warm_n>=500 still counted unit-on-snow as own. Wood shape drops it."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (22, 24, 28))
    d = ImageDraw.Draw(im)
    _draw_bardur_city(d, 252, 470)
    _draw_bardur_city(d, 541, 690)
    _draw_unit_frost_phantom(d, 610, 400)
    _draw_disrof_city(d, 980, 260)
    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    assert len(own) == 2, mapped["cities"]
    assert all(abs(int(c["x"]) - 610) > 40 or abs(int(c.get("plate_y") or c["y"]) - 400) > 40 for c in own), own
    vengir = [c for c in mapped["cities_enemy"] if c.get("tribe") == "vengir"]
    assert vengir, mapped["cities"]


def test_two_bardur_match_screen_not_frost_phantoms():
    """Live T46: Game Stats Bardur 2; observe listed 5–7 (c15_15, c15_8, c21_23)."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (22, 24, 28))
    d = ImageDraw.Draw(im)
    _draw_bardur_city(d, 400, 280)
    _draw_bardur_city(d, 720, 420)
    # Unnamed frost fragments (the ghosts).
    for x0, y0 in ((90, 160), (520, 150), (980, 180), (240, 520), (1080, 560)):
        d.rectangle((x0, y0, x0 + 100, y0 + 20), fill=(180, 180, 178))
        d.rectangle((x0 + 12, y0 + 6, x0 + 18, y0 + 14), fill=(40, 40, 40))
    # Fruit-gold on a frost streak — used to count as a gold-star city.
    d.ellipse((530, 128, 548, 146), fill=(224, 188, 63))
    # Cool mountain grey + frost + gold (not Bardur wood).
    d.rectangle((960, 340, 1020, 400), fill=(110, 112, 118))
    d.rectangle((940, 408, 1044, 428), fill=(180, 180, 178))
    d.ellipse((1020, 410, 1036, 426), fill=(224, 188, 63))
    d.rectangle((952, 414, 958, 424), fill=(40, 40, 40))
    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    assert len(own) == 2, mapped["cities"]
    assert all(c["tribe"] == "bardur" and c["owner"] == "own" for c in own), own
    assert all(
        "gold_star" in (c.get("evidence") or []) or "gold_lamps" in (c.get("evidence") or [])
        for c in own
    ), own
    xs = sorted(int(c["x"]) for c in own)
    assert any(abs(x - 400) < 40 for x in xs), own
    assert any(abs(x - 720) < 40 for x in xs), own


def test_own_phantoms_do_not_accumulate_across_frames():
    """Sticky keep_missing used to grow cities_own 2→5–7 after frost FPs."""
    from polytopia_api.entities import session_cities
    from polytopia_api.observe import stabilize

    real_a = _own_city(7, y=300)
    real_a["id"] = real_a["city_id"] = "c7_16"
    real_a["tile"] = [7, 16]
    real_b = _own_city(12, y=420)
    real_b["id"] = real_b["city_id"] = "c12_16"
    real_b["tile"] = [12, 16]
    ghosts = []
    for cid, tile, x, y in (
        ("c15_15", [15, 15], 200, 160),
        ("c15_8", [15, 8], 640, 200),
        ("c21_23", [21, 23], 980, 520),
    ):
        ghosts.append({
            "id": cid, "city_id": cid, "tribe": "bardur", "owner": "own",
            "name": None, "confidence": 0.72, "evidence": ["frost_plate"],
            "x": x, "y": y, "tile": tile, "n": 40, "w": 36, "seen": True,
        })
    prev = [real_a, real_b, *ghosts]
    nxt = [dict(real_a, x=real_a["x"] + 1), dict(real_b, x=real_b["x"] + 1)]
    out = session_cities(stabilize(nxt, prev, keep_missing=True))
    own = [c for c in out if c.get("owner") == "own"]
    ids = {c["id"] for c in own}
    assert ids == {"c7_16", "c12_16"}, own
    assert "c15_15" not in ids and "c15_8" not in ids and "c21_23" not in ids


def test_recruit_timeout_skips_full_observe():
    """Live: /recruit hung on HUD OCR after every nameplate click."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c5_12",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [5, 12],
        "tribe": "bardur",
        "owner": "own",
    }
    register_cities([city])
    clock = {"t": 0.0}
    observe_n = {"n": 0}

    def fake_time():
        return clock["t"]

    def fake_observe(*args, **kwargs):
        observe_n["n"] += 1
        raise AssertionError("recruit must not observe again after the 8s budget")

    def fake_click(x, y, space="screen", repeats=1):
        clock["t"] = 8.0
        return {"ok": True, "x": int(x), "y": int(y)}

    with patch.object(commands.time, "time", fake_time), patch.object(
        commands, "remember", return_value=None
    ), patch.object(commands, "observe", side_effect=fake_observe), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.recruit(city_id="c5_12")
    assert r["ok"] is False and r["reason"] == "timeout", r
    assert r["elapsed_s"] < 12
    assert observe_n["n"] == 0
    reset_obs()


def test_recruit_uses_marks_observe_for_train():
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c4_11",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [4, 11],
        "tribe": "bardur",
        "owner": "own",
    }
    register_cities([city])
    modes: list[str] = []
    panel = {
        "layout": {"frame": [1280, 800]},
        "overlay": {"train_blobs": [{"x": 180, "y": 740, "n": 200}]},
        "unit": {"train": True},
        "ready": {"train": True},
        "hud": {},
        "turn_diff": None,
    }

    def fake_observe(*args, **kwargs):
        modes.append(str(kwargs.get("mode") or (args[0] if args else "full")))
        return panel

    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    with patch.object(commands, "remember", return_value=None), patch.object(
        commands, "observe", side_effect=fake_observe
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.recruit(city_id="c4_11")
    assert r["ok"] is True, r
    assert (180, 740) in clicks
    assert modes and all(m == "marks" for m in modes), modes
    reset_obs()


def test_recruit_clicks_roof_then_train():
    """Roof then nameplate open TRAIN. Foot/tile-first selected Disband on live 99876c5."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c7_16",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [7, 16],
        "tile_xy": [200, 80],
        "tribe": "bardur",
        "owner": "own",
    }
    register_cities([city])
    clicks: list[tuple[int, int]] = []
    n_obs = {"n": 0}

    def fake_observe(*args, **kwargs):
        n_obs["n"] += 1
        opened = n_obs["n"] >= 1
        return {
            "layout": {"frame": [1280, 800]},
            "overlay": {"train_blobs": [{"x": 184, "y": 740, "n": 200}] if opened else []},
            "unit": {"train": True} if opened else {},
            "ready": {"train": True} if opened else {},
            "hud": {},
            "turn_diff": None,
        }

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    pts = commands._recruit_click_points(city, (1280, 800), None)
    assert pts and pts[0][2] in {"roof", "building", "plate"}, pts
    assert "tile" not in [p[2] for p in pts[:3]], pts

    with patch.object(commands, "remember", return_value=None), patch.object(
        commands, "observe", side_effect=fake_observe
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.recruit(city_id="c7_16")
    assert r["ok"] is True, r
    assert (184, 740) in clicks
    wheres = [t.get("where") for t in (r.get("tried") or [])]
    assert wheres and wheres[0] in {"roof", "building", "plate"}, r
    assert "city_foot" not in wheres
    assert "plate_below" not in wheres
    reset_obs()


def test_recruit_backs_off_disband_then_clicks_train():
    """Occupied city foot opens Disband — BACK, then roof TRAIN in UNIT_CROP."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c5_12",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [5, 12],
        "tile_xy": [200, 80],
        "tribe": "bardur",
        "owner": "own",
    }
    register_cities([city])
    clicks: list[tuple[int, int]] = []
    backs = {"n": 0}
    n_obs = {"n": 0}

    def fake_observe(*args, **kwargs):
        n_obs["n"] += 1
        if n_obs["n"] == 1:
            return {
                "layout": {"frame": [1280, 800]},
                "overlay": {"train_blobs": []},
                "unit": {"train": False, "disband": True, "unit": "warrior", "raw": "Disband"},
                "ready": {},
                "hud": {},
                "turn_diff": None,
            }
        return {
            "layout": {"frame": [1280, 800]},
            "overlay": {"train_blobs": [{"x": 184, "y": 740, "n": 180}]},
            "unit": {"train": True, "disband": False, "raw": "Train"},
            "ready": {"train": True},
            "hud": {},
            "turn_diff": None,
        }

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    def fake_back():
        backs["n"] += 1
        return {"ok": True}

    with patch.object(commands, "remember", return_value=None), patch.object(
        commands, "observe", side_effect=fake_observe
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ), patch.object(commands.driver, "back", side_effect=fake_back):
        r = commands.recruit(city_id="c5_12")
    assert r["ok"] is True, r
    assert (184, 740) in clicks
    assert 184 <= clicks[-1][0] <= 200 or clicks[-1] == (184, 740)
    assert clicks[-1][1] >= int(800 * 0.78), clicks
    assert backs["n"] >= 1
    assert "disband" in (r.get("dismissed") or [])
    tech = coords.point("TECH_TREE")
    assert tech not in clicks
    reset_obs()


def test_recruit_clicks_brand_blue_panel_pill():
    """Live: city TRAIN is brand-blue (Capture/DO IT range), not the dark modal TRAIN."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c4_11",
        "kind": "city",
        "x": 220,
        "y": 90,
        "plate_y": 120,
        "tile": [4, 11],
        "tribe": "bardur",
        "owner": "own",
    }
    register_cities([city])
    clicks: list[tuple[int, int]] = []
    panel = {
        "layout": {"frame": [1280, 800]},
        "overlay": {
            "train_blobs": [],
            "capture_blobs": [{"x": 190, "y": 742, "n": 160}],
            "do_it_blobs": [],
        },
        "unit": {"train": False, "capture": False},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    with patch.object(commands, "remember", return_value=None), patch.object(
        commands, "observe", return_value=panel
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.recruit(city_id="c4_11")
    assert r["ok"] is True, r
    assert (190, 742) in clicks
    assert all(not coords.in_dock_zone(x, y) for x, y in clicks), clicks
    reset_obs()


def test_recruit_clicks_train_despite_leftover_unit_ocr():
    """Live: city panel OCR reads a nearby warrior — must not BACK; click brand-blue TRAIN."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c5_12",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [5, 12],
        "tribe": "bardur",
        "owner": "own",
    }
    register_cities([city])
    clicks: list[tuple[int, int]] = []
    backs = {"n": 0}
    panel = {
        "layout": {"frame": [1280, 800]},
        "overlay": {
            "train_blobs": [],
            "capture_blobs": [{"x": 188, "y": 744, "n": 150}],
            "do_it_blobs": [],
        },
        "unit": {"train": False, "disband": False, "unit": "warrior", "raw": "Warrior"},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    def fake_back():
        backs["n"] += 1
        return {"ok": True}

    with patch.object(commands, "remember", return_value=None), patch.object(
        commands, "observe", return_value=panel
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ), patch.object(commands.driver, "back", side_effect=fake_back):
        r = commands.recruit(city_id="c5_12")
    assert r["ok"] is True, r
    assert (188, 744) in clicks
    assert backs["n"] == 0, r
    assert "unit" not in (r.get("dismissed") or [])
    tech = coords.point("TECH_TREE")
    assert tech not in clicks
    reset_obs()


def test_recruit_waits_for_train_after_nameplate():
    """Panel fade-in: first marks observe is empty, second has TRAIN — do not spray foot."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c4_11",
        "kind": "city",
        "x": 220,
        "y": 90,
        "plate_y": 120,
        "tile": [4, 11],
        "tribe": "bardur",
        "owner": "own",
    }
    register_cities([city])
    clicks: list[tuple[int, int]] = []
    n_obs = {"n": 0}

    def fake_observe(*args, **kwargs):
        n_obs["n"] += 1
        opened = n_obs["n"] >= 2
        return {
            "layout": {"frame": [1280, 800]},
            "overlay": {"train_blobs": [{"x": 176, "y": 738, "n": 180}] if opened else []},
            "unit": {"train": True, "raw": "Train"} if opened else {"train": False, "raw": ""},
            "ready": {"train": True} if opened else {},
            "hud": {},
            "turn_diff": None,
        }

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    with patch.object(commands, "remember", return_value=None), patch.object(
        commands, "observe", side_effect=fake_observe
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.recruit(city_id="c4_11")
    assert r["ok"] is True, r
    assert (176, 738) in clicks
    wheres = [t.get("where") for t in (r.get("tried") or [])]
    assert wheres[0] in {"roof", "plate"}, r
    assert "city_foot" not in wheres
    reset_obs()


def test_adjacent_bardur_cities_stay_two_own():
    """Live: cities_own 3–4 while neighboring Bardur sat ~1 hex apart (banners kiss)."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map, session_cities

    a = _own_city(4, y=300)
    a["id"] = a["city_id"] = "c12_10"
    a["tile"] = [12, 10]
    a["x"], a["y"] = 400, 300
    a["tile_xy"] = [400, 300]
    a["plate_y"] = 300
    a["w"] = 90
    b = _own_city(5, y=300)
    b["id"] = b["city_id"] = "c13_10"
    b["tile"] = [13, 10]
    b["x"], b["y"] = 480, 302
    b["tile_xy"] = [480, 302]
    b["plate_y"] = 302
    b["w"] = 90
    out = session_cities([a, b])
    own_ids = {c["id"] for c in out if c.get("owner") == "own"}
    assert own_ids >= {"c12_10", "c13_10"}, out

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (22, 24, 28))
    d = ImageDraw.Draw(im)
    _draw_bardur_city(d, 400, 300)
    _draw_bardur_city(d, 480, 300)
    mapped = observe_map(as_rgb(im))
    own = mapped["cities_own"]
    assert len(own) == 2, mapped["cities"]
    xs = sorted(int(c["x"]) for c in own)
    assert any(abs(x - 400) < 40 for x in xs), own
    assert any(abs(x - 480) < 40 for x in xs), own


def test_unit_id_stays_resolvable_after_observe_drops_it():
    """Live: after /attack timeout, next call was 'no entity u2'."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_units, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    register_units([{"id": "u2", "kind": "unit", "x": 140, "y": 80, "owner": "own"}])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    blank = {
        "layout": {"frame": [1280, 800]},
        "units": [],
        "cities": [],
        "unit": {"no_actions": False, "unit": "warrior"},
        "move_marks": [],
        "attack_marks": [],
        "hud": {},
        "ready": {},
        "overlay": {},
        "turn_diff": None,
    }
    with patch.object(commands, "remember", return_value=blank), patch.object(
        commands, "observe", return_value=blank
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.select_unit(id="u2", light=True)
    assert r["ok"] is True, r
    assert r.get("reason") != "no entity u2"
    assert clicks == [(140, 80)], clicks
    reset_obs()


def test_move_to_clicks_blue_ring_at_city_foot():
    """Live: blue on Disrof sits at the building foot (~plate), ~0.9 from roof tile_xy.

    7ab379e treated that ring as adjacent and clicked the roof instead, so every
    /move-to returned not_stood_on_city.
    """
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = _disrof_city()
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    foot = {"id": "m_foot", "x": 200, "y": 104, "n": 22}
    adj = {"id": "m_adj", "x": 164, "y": 80, "n": 80}
    before = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "cities_own": [],
        "move_marks": [adj, foot],
        "units": [{"id": "u1", "x": 164, "y": 80, "owner": "own"}],
        "unit": {"can_move": True, "no_actions": False, "unit": "rider"},
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    after = {
        **before,
        "move_marks": [],
        "units": [{"id": "u1", "x": 201, "y": 94, "owner": "own"}],
        "unit": {"can_move": False, "capture": True, "no_actions": False, "unit": "rider"},
        "ready": {"capture": True},
    }
    selected = {"ok": True, "unit": before["unit"]}
    with patch.object(commands, "remember", return_value=before), patch.object(
        commands, "observe", return_value=after
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u1", city_id="c5_3")
    assert clicks and clicks[0] == (200, 104), clicks
    assert (164, 80) not in clicks
    assert (200, 80) not in clicks
    assert r["ok"] is True and r["stood_on_city"] is True, r
    reset_obs()


def test_move_to_never_returns_ok_null():
    """Live: /move-to {u14, c16_10} came back empty-ish {ok:null}."""
    from unittest.mock import patch

    from polytopia_api import commands

    with patch.object(commands, "select_unit", return_value={}), patch.object(
        commands, "remember", return_value=None
    ), patch.object(commands, "observe", return_value={"layout": {"frame": [1280, 800]}}):
        r = commands.move_to(from_id="u14", city_id="c16_10")
    assert r.get("ok") is False, r
    assert r.get("stood_on_city") is False
    assert r.get("reason")


def _disrof_city():
    return {
        "id": "c5_3",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [5, 3],
        "tile_xy": [200, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }


def test_move_to_does_not_click_adjacent_mark():
    """Live: rider adjacent to Disrof — API clicked the adjacent blue, ok:true, stood_on_city:false."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = _disrof_city()
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    adj = {"id": "m_adj", "x": 164, "y": 80, "n": 80}
    foot = {"id": "m_foot", "x": 200, "y": 104, "n": 22}
    before = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "cities_own": [],
        "move_marks": [adj, foot],
        "units": [{"id": "u1", "x": 164, "y": 80, "owner": "own"}],
        "unit": {"can_move": True, "no_actions": False, "unit": "rider"},
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    after = {
        **before,
        "move_marks": [],
        "units": [{"id": "u1", "x": 201, "y": 94, "owner": "own"}],
        "unit": {"can_move": False, "capture": True, "no_actions": False, "unit": "rider"},
        "ready": {"capture": True},
    }
    selected = {"ok": True, "unit": before["unit"]}
    with patch.object(commands, "remember", return_value=before), patch.object(
        commands, "observe", return_value=after
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u1", city_id="c5_3")
    assert clicks and clicks[0] == (200, 104), clicks
    assert (164, 80) not in clicks
    assert (200, 80) not in clicks
    assert r["ok"] is True and r["stood_on_city"] is True, r
    reset_obs()


def test_move_to_no_mark_does_not_click_building():
    """In range but no blue on the city hex: honest no_mark, do not click the roof."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = _disrof_city()
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    stuck = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "move_marks": [{"x": 164, "y": 80, "n": 80}],
        "units": [{"id": "u1", "x": 164, "y": 80, "owner": "own"}],
        "unit": {"can_move": True, "no_actions": False, "unit": "rider"},
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    selected = {"ok": True, "unit": stuck["unit"]}
    with patch.object(commands, "remember", return_value=stuck), patch.object(
        commands, "observe", return_value=stuck
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u1", city_id="c5_3")
    assert r.get("ok") is False, r
    assert r.get("stood_on_city") is False
    assert r.get("reason") == "no_mark_on_city_tile"
    assert r.get("path_reason") == "no_mark_on_city_tile"
    assert clicks == []
    reset_obs()


def test_move_to_out_of_range_reason():
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = _disrof_city()
    register_cities([city])
    far = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "move_marks": [{"x": 50, "y": 400, "n": 40}],
        "units": [{"id": "u1", "x": 50, "y": 400, "owner": "own"}],
        "unit": {"can_move": True, "no_actions": False, "unit": "warrior"},
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    selected = {"ok": True, "unit": far["unit"]}
    with patch.object(commands, "remember", return_value=far), patch.object(
        commands, "observe", return_value=far
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click"
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u1", city_id="c5_3")
    assert r.get("ok") is False
    assert r.get("stood_on_city") is False
    assert r.get("reason") == "out_of_range"
    assert r.get("path_reason") == "out_of_range"
    reset_obs()


def test_move_to_ok_false_when_still_adjacent():
    """Clicked the on-city blue but the unit is still next to Disrof."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = _disrof_city()
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    stuck = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "move_marks": [{"x": 200, "y": 104, "n": 22}, {"x": 164, "y": 80, "n": 80}],
        "units": [{"id": "u1", "x": 164, "y": 80, "owner": "own"}],
        "unit": {"can_move": True, "no_actions": False, "unit": "rider"},
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    selected = {"ok": True, "unit": stuck["unit"]}
    with patch.object(commands, "remember", return_value=stuck), patch.object(
        commands, "observe", return_value=stuck
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u1", city_id="c5_3")
    assert r.get("ok") is False, r
    assert r.get("stood_on_city") is False
    assert r.get("reason") == "not_stood_on_city"
    assert clicks and clicks[0] == (200, 104), clicks
    assert (164, 80) not in clicks
    reset_obs()


def test_move_to_reselects_after_city_panel():
    """Roof click opens the city panel; re-select the unit and click the foot blue."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = _disrof_city()
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    roof = {"id": "m_roof", "x": 200, "y": 83, "n": 18}
    foot = {"id": "m_foot", "x": 200, "y": 104, "n": 22}
    base = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "units": [{"id": "u1", "x": 164, "y": 80, "owner": "own"}],
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    before = {
        **base,
        "move_marks": [roof],
        "unit": {"can_move": True, "no_actions": False, "unit": "rider"},
    }
    city_panel = {
        **base,
        "move_marks": [],
        "unit": {"train": True, "can_move": False, "no_actions": False},
    }
    reselected = {
        **base,
        "move_marks": [foot],
        "unit": {"can_move": True, "no_actions": False, "unit": "rider"},
    }
    stood = {
        **base,
        "move_marks": [],
        "units": [{"id": "u1", "x": 201, "y": 94, "owner": "own"}],
        "unit": {"can_move": False, "capture": True, "no_actions": False, "unit": "rider"},
        "ready": {"capture": True},
    }
    mem = {"state": before}
    observes = [city_panel, stood]

    def fake_remember():
        return mem["state"]

    def fake_observe(mode="full"):
        nxt = observes.pop(0) if observes else stood
        mem["state"] = nxt
        return nxt

    selects = {"n": 0}

    def fake_select(**_kw):
        selects["n"] += 1
        if selects["n"] >= 2:
            mem["state"] = reselected
            return {"ok": True, "unit": reselected["unit"]}
        return {"ok": True, "unit": before["unit"]}

    with patch.object(commands, "remember", side_effect=fake_remember), patch.object(
        commands, "observe", side_effect=fake_observe
    ), patch.object(commands, "select_unit", side_effect=fake_select), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u1", city_id="c5_3")
    assert selects["n"] >= 2, selects
    assert (200, 104) in clicks, clicks
    assert r["ok"] is True and r["stood_on_city"] is True, r
    reset_obs()


def test_move_to_xy_walks_onto_city_not_plate():
    """Live: POST /move-to {x,y} used city plate coords and never stood ON the hex."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = _disrof_city()
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    foot = {"x": 200, "y": 104, "n": 22}
    before = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "move_marks": [foot],
        "units": [{"id": "u1", "x": 164, "y": 80, "owner": "own"}],
        "unit": {"can_move": True, "no_actions": False, "unit": "rider"},
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    after = {
        **before,
        "move_marks": [],
        "units": [{"id": "u1", "x": 201, "y": 94, "owner": "own"}],
        "unit": {"can_move": False, "capture": True, "no_actions": False, "unit": "rider"},
        "ready": {"capture": True},
    }
    selected = {"ok": True, "unit": before["unit"]}
    with patch.object(commands, "remember", return_value=before), patch.object(
        commands, "observe", return_value=after
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u1", x=200, y=110)
    assert r["ok"] is True and r["stood_on_city"] is True, r
    assert clicks and clicks[0][1] < 110, clicks
    assert clicks[0] == (200, 104), clicks
    reset_obs()


def test_city_tile_center_lifts_plate_xy():
    from polytopia_api.combat import city_tile_center, city_walk_center

    city = {"x": 200, "y": 80, "plate_y": 110, "tile_xy": [200, 110]}
    c = city_tile_center(city, (1280, 800))
    assert c is not None
    assert c[1] < 110
    assert abs(c[1] - 83) <= 2
    w = city_walk_center({"x": 200, "y": 80, "plate_y": 110}, (1280, 800))
    assert w is not None
    assert 90 <= w[1] <= 100
    assert w[1] > c[1]
    roof_only = city_walk_center({"x": 200, "y": 80, "tile_xy": [200, 80]}, (1280, 800))
    assert roof_only is not None and roof_only[1] > 80
    from polytopia_api.combat import city_move_aim
    foot = city_move_aim({"x": 200, "y": 81}, {"x": 200, "y": 80, "plate_y": 110, "tile_xy": [200, 80]}, (1280, 800))
    assert foot is not None and foot[1] > 81


def test_move_on_hex_finds_foot_ring():
    from polytopia_api.combat import hex_pitch
    from polytopia_api.detect import as_rgb, move_on_hex

    im = Image.new("RGB", (1280, 800), (20, 20, 20))
    px = im.load()
    for y in range(98, 110):
        for x in range(192, 208):
            px[x, y] = (146, 204, 255)
    arr = as_rgb(im)
    hit = move_on_hex(arr, 200, 94, hex_pitch(1280, 800))
    assert hit["ok"] is True, hit
    assert abs(hit["x"] - 200) < 8
    assert 98 <= hit["y"] <= 110


def _m20_disrof_mountain(im: Image.Image, hp_bar: bool = True) -> None:
    """Live M20: Disrof tile_xy≈[600,144], Bardur on the mountain immediately south."""
    d = ImageDraw.Draw(im)
    d.rectangle((565, 100, 635, 165), fill=(90, 85, 80))
    d.rectangle((580, 95, 620, 130), fill=(150, 74, 144))
    d.rectangle((550, 168, 650, 188), fill=(180, 180, 178))
    d.rectangle((558, 172, 570, 184), fill=(224, 188, 63))
    d.rectangle((588, 172, 600, 184), fill=(224, 188, 63))
    d.rectangle((575, 174, 578, 182), fill=(40, 40, 40))
    d.ellipse((575, 188, 625, 232), fill=(230, 230, 235))
    d.rectangle((588, 204, 612, 222), fill=(110, 85, 72))
    if hp_bar:
        d.rectangle((590, 184, 608, 188), fill=(90, 210, 50))
    d.rectangle((636, 458, 652, 461), fill=(90, 210, 50))
    d.rectangle((638, 466, 650, 486), fill=(110, 85, 72))


def _m20_disrof_port_east(
    im: Image.Image,
    hp_bar: bool = True,
    cover_roof: bool = False,
    cover_plate_right: bool = False,
) -> None:
    """Live T37: Disrof on screen, Bardur on the port immediately east.

    cities_enemy went [] because the garrison / port unit ate magenta + plate
    contrast, and the east hex was never scanned for leather (south-of-plate only).
    """
    d = ImageDraw.Draw(im)
    # Water + wooden port east of the city hex.
    d.rectangle((630, 130, 720, 200), fill=(80, 200, 240))
    d.rectangle((632, 150, 678, 184), fill=(120, 90, 70))
    d.rectangle((565, 100, 635, 165), fill=(90, 85, 80))
    d.rectangle((580, 95, 620, 130), fill=(150, 74, 144))
    if cover_roof:
        d.ellipse((572, 88, 628, 138), fill=(48, 36, 58))
    d.rectangle((550, 168, 650, 188), fill=(180, 180, 178))
    d.rectangle((558, 172, 570, 184), fill=(224, 188, 63))
    d.rectangle((588, 172, 600, 184), fill=(224, 188, 63))
    d.rectangle((575, 174, 578, 182), fill=(40, 40, 40))
    if cover_plate_right:
        d.rectangle((612, 166, 650, 188), fill=(110, 85, 72))
    # Bardur warrior on the port (east neighbor of walk center ~[636, 162]).
    d.rectangle((640, 154, 664, 178), fill=(110, 85, 72))
    if hp_bar:
        d.rectangle((642, 146, 660, 151), fill=(90, 210, 50))
    # Distant southern warrior — must not be the only unit listed.
    d.rectangle((636, 458, 652, 461), fill=(90, 210, 50))
    d.rectangle((638, 466, 650, 486), fill=(110, 85, 72))


def test_observe_detects_unit_on_mountain_south_of_disrof():
    """Live T36: nearest listed units were y≈462; warrior on mountain south of Disrof missing.

    HP bar sits on the frost-plate edge; the pixel under the bar is snow (water).
    """
    from polytopia_api.combat import hex_pitch, tile_dist
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    assert is_snow_rgb(230, 230, 235)
    assert not is_water_rgb(230, 230, 235)
    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (30, 40, 28))
    _m20_disrof_mountain(im, hp_bar=True)
    mapped = observe_map(as_rgb(im))
    enemy = mapped["cities_enemy"]
    assert enemy, mapped["cities"]
    disrof = min(enemy, key=lambda c: abs(int(c["x"]) - 600) + abs(int(c.get("tile_xy", [0, c["y"]])[1]) - 144))
    assert abs(int(disrof["x"]) - 600) < 80, disrof
    near = [
        u
        for u in mapped["units"]
        if abs(int(u["x"]) - 570) < 60 and 165 <= int(u["y"]) <= 240
    ]
    assert near, mapped["units"]
    u = near[0]
    assert u.get("owner") in {"own", "unknown"} or u.get("tribe") in {"bardur", "unknown"}
    assert u.get("tile")
    pitch = hex_pitch(1280, 800)
    origin = (int(disrof["x"]), int(disrof.get("plate_y") or disrof["y"]))
    assert tile_dist(int(u["x"]), int(u["y"]), origin[0], origin[1], pitch) <= 2.3, (u, origin, disrof)
    south = [u for u in mapped["units"] if int(u["y"]) >= 430]
    assert south, mapped["units"]


def test_leather_unit_south_of_disrof_without_hp_bar():
    """Nameplate can fully cover the lime bar; Bardur hide on the south hex still counts."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (30, 40, 28))
    _m20_disrof_mountain(im, hp_bar=False)
    mapped = observe_map(as_rgb(im))
    near = [
        u
        for u in mapped["units"]
        if abs(int(u["x"]) - 570) < 60 and 165 <= int(u["y"]) <= 240
    ]
    assert near, mapped["units"]
    assert any(u.get("owner") == "own" or u.get("tribe") == "bardur" for u in near), near


def _t37_disrof(mapped):
    enemy = mapped["cities_enemy"]
    assert enemy, mapped["cities"]
    disrof = min(
        enemy,
        key=lambda c: abs(int(c["x"]) - 600) + abs(int(c.get("tile_xy", [0, c["y"]])[1]) - 144),
    )
    assert abs(int(disrof["x"]) - 600) < 80, disrof
    assert disrof.get("tribe") == "vengir" and disrof.get("owner") == "enemy", disrof
    own_at = [
        c for c in mapped["cities_own"]
        if abs(int(c["x"]) - 600) < 50
    ]
    assert not own_at, own_at
    return disrof


def test_observe_keeps_disrof_with_unit_on_east_port():
    """Live T37: cities_enemy [] while Disrof was on screen with a Bardur on the east port."""
    from polytopia_api.combat import hex_pitch, tile_dist
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (30, 40, 28))
    _m20_disrof_port_east(im, hp_bar=True)
    mapped = observe_map(as_rgb(im))
    disrof = _t37_disrof(mapped)
    near = [
        u
        for u in mapped["units"]
        if int(u["x"]) >= 620 and 140 <= int(u["y"]) <= 200
    ]
    assert near, mapped["units"]
    u = near[0]
    assert u.get("owner") in {"own", "unknown"} or u.get("tribe") in {"bardur", "unknown"}
    assert u.get("tile")
    pitch = hex_pitch(1280, 800)
    origin = (int(disrof["x"]), int(disrof.get("plate_y") or disrof["y"]))
    assert tile_dist(int(u["x"]), int(u["y"]), origin[0], origin[1], pitch) <= 2.3, (u, origin, disrof)
    assert any(int(u["y"]) >= 430 for u in mapped["units"]), mapped["units"]


def test_observe_keeps_disrof_when_garrison_covers_roof():
    """Garrison on the city hex must not empty cities_enemy."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (30, 40, 28))
    _m20_disrof_port_east(im, hp_bar=True, cover_roof=True, cover_plate_right=True)
    mapped = observe_map(as_rgb(im))
    disrof = _t37_disrof(mapped)
    near = [
        u
        for u in mapped["units"]
        if int(u["x"]) >= 620 and 140 <= int(u["y"]) <= 200
    ]
    assert near, (mapped["units"], disrof)


def test_observe_east_port_unit_without_hp_bar():
    """Port-east leather still counts when the nameplate/garrison covers the lime bar."""
    from polytopia_api.detect import as_rgb
    from polytopia_api.entities import observe_map

    coords.reset()
    coords.set_frame(1280, 800)
    im = Image.new("RGB", (1280, 800), (30, 40, 28))
    _m20_disrof_port_east(im, hp_bar=False)
    mapped = observe_map(as_rgb(im))
    _t37_disrof(mapped)
    near = [
        u
        for u in mapped["units"]
        if int(u["x"]) >= 620 and 140 <= int(u["y"]) <= 200
    ]
    assert near, mapped["units"]
    assert any(u.get("owner") == "own" or u.get("tribe") == "bardur" for u in near), near


def test_sticky_bardur_stays_own_if_frame_flips_vengir():
    """Live T46: after /recruit click, c7_16 flipped cities_own → cities_enemy."""
    prev = [{
        "id": "c7_16",
        "city_id": "c7_16",
        "tile": [7, 16],
        "tile_xy": [400, 300],
        "x": 400,
        "y": 300,
        "plate_y": 330,
        "tribe": "bardur",
        "owner": "own",
        "name": None,
        "evidence": ["frost_plate", "gold_star"],
        "confidence": 0.72,
        "n": 380,
    }]
    nxt = [{
        "id": "c7_16",
        "city_id": "c7_16",
        "tile": [7, 16],
        "x": 402,
        "y": 302,
        "plate_y": 332,
        "tribe": "vengir",
        "owner": "enemy",
        "evidence": ["frost_plate", "magenta_roof"],
        "confidence": 0.70,
        "n": 360,
    }]
    out = stabilize(nxt, prev, keep_missing=True)
    hit = [c for c in out if c.get("id") == "c7_16"]
    assert hit and hit[0]["tribe"] == "bardur" and hit[0]["owner"] == "own", hit


def test_sticky_disrof_stays_enemy_if_frame_flips_bardur():
    """A later observe that samples the adjacent Bardur must not invent cities_own."""
    from polytopia_api.observe import stabilize

    prev = [{
        "id": "c17_5",
        "city_id": "c17_5",
        "tile": [17, 5],
        "tile_xy": [600, 144],
        "x": 600,
        "y": 144,
        "plate_y": 178,
        "tribe": "vengir",
        "owner": "enemy",
        "name": None,
        "evidence": ["magenta_roof", "gold_lamps", "frost_plate"],
        "confidence": 0.78,
        "n": 400,
    }]
    nxt = [{
        "id": "c17_5",
        "city_id": "c17_5",
        "tile": [17, 5],
        "x": 602,
        "y": 146,
        "plate_y": 180,
        "tribe": "bardur",
        "owner": "own",
        "evidence": ["frost_plate", "gold_star"],
        "confidence": 0.72,
        "n": 380,
    }]
    out = stabilize(nxt, prev, keep_missing=True)
    hit = [c for c in out if c.get("id") == "c17_5"]
    assert hit and hit[0]["tribe"] == "vengir" and hit[0]["owner"] == "enemy", hit
    assert "magenta_roof" in (hit[0].get("evidence") or [])


def test_stabilize_units_matches_by_tile():
    from polytopia_api.observe import stabilize_units

    prev = [{"id": "u7", "x": 600, "y": 210, "tile": [17, 8], "owner": "own"}]
    nxt = [{"x": 602, "y": 184, "tile": [17, 8], "owner": "own", "kind": "unit"}]
    out = stabilize_units(nxt, prev, keep_missing=True)
    seen = [u for u in out if u.get("seen") is not False]
    assert seen and seen[0]["id"] == "u7"


def test_move_to_from_mountain_unit_stands_on_city():
    """Select the mountain-south unit, click city-hex blue, stood_on_city."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, register_units, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c16_5",
        "kind": "city",
        "x": 600,
        "y": 144,
        "plate_y": 178,
        "tile": [16, 5],
        "tile_xy": [600, 144],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    register_units([{"id": "u3", "kind": "unit", "x": 600, "y": 210, "owner": "own", "tile": [16, 8]}])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    foot = {"id": "m_foot", "x": 600, "y": 162, "n": 18}
    before = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "move_marks": [foot],
        "units": [{"id": "u3", "x": 600, "y": 210, "owner": "own", "near_city_id": "c16_5"}],
        "unit": {"can_move": True, "no_actions": False, "unit": "warrior"},
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    after = {
        **before,
        "move_marks": [],
        "units": [{"id": "u3", "x": 600, "y": 162, "owner": "own"}],
        "unit": {"can_move": False, "capture": True, "no_actions": False, "unit": "warrior"},
        "ready": {"capture": True},
    }
    selected = {"ok": True, "unit": before["unit"]}
    with patch.object(commands, "remember", return_value=before), patch.object(
        commands, "observe", return_value=after
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u3", city_id="c16_5")
    assert clicks and clicks[0] == (600, 162), clicks
    assert r["ok"] is True and r["stood_on_city"] is True, r
    panel_blob = {"x": 180, "y": 740, "n": 200}
    cap_state = {
        **after,
        "overlay": {"capture_blobs": [panel_blob]},
        "ready": {"capture": True},
        "unit": {"capture": True, "no_actions": False},
    }
    with patch.object(commands, "remember", return_value=cap_state), patch.object(
        commands, "observe", return_value={**cap_state, "cities_enemy": [], "cities_own": [{**city, "owner": "own", "tribe": "bardur"}]}
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        cap = commands.capture(city_id="c16_5")
    assert cap.get("captured") is True, cap
    reset_obs()


def test_capture_when_on_disrof_with_capture_flag():
    """Live: unit ON Disrof, panel says Capture, API returned not_standing_on_city.

    HP bar / sticky coords sit ~1.0 hex from walk (nameplate / east-port listing).
    Capture button is game-truth for standing ON the city.
    """
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c17_5",
        "kind": "city",
        "x": 600,
        "y": 144,
        "plate_y": 178,
        "tile": [17, 5],
        "tile_xy": [600, 144],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    blob = {"x": 180, "y": 740, "n": 200}
    # Sticky / HP-bar listing still at the east-port hex (~1.0 from walk).
    on_flag = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "cities_own": [],
        "units": [{"id": "u4", "x": 636, "y": 162, "owner": "own", "near_city_id": "c17_5", "near_city_hex": 1.0}],
        "unit": {"capture": True, "no_actions": False, "unit": "warrior"},
        "ready": {"capture": True},
        "overlay": {"capture_blobs": [blob]},
        "hud": {},
        "turn_diff": None,
    }
    after = {
        **on_flag,
        "cities_enemy": [],
        "cities_own": [{**city, "owner": "own", "tribe": "bardur"}],
        "unit": {"capture": False},
        "ready": {},
    }
    with patch.object(commands, "remember", return_value=on_flag), patch.object(
        commands, "observe", return_value=after
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        cap = commands.capture(city_id="c17_5")
    assert cap.get("ok") is True and cap.get("captured") is True, cap
    assert clicks == [(180, 740)], clicks

    clicks.clear()
    ocr_only = {
        **on_flag,
        "units": [{"id": "u4", "x": 600, "y": 178, "owner": "unknown"}],
        "overlay": {"capture_blobs": [], "do_it_blobs": [{"x": 400, "y": 400, "n": 800}]},
    }
    with patch.object(commands, "remember", return_value=ocr_only), patch.object(
        commands, "observe", return_value=after
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        cap = commands.capture(city_id="c17_5")
    assert cap.get("ok") is True and cap.get("captured") is True, cap
    assert clicks and clicks[0][1] >= int(800 * 0.78), clicks
    assert clicks[0] != (400, 400)

    clicks.clear()
    adjacent_no_flag = {
        **on_flag,
        "unit": {"capture": False, "can_move": True, "unit": "warrior"},
        "ready": {},
        "overlay": {},
    }
    with patch.object(commands, "remember", return_value=adjacent_no_flag), patch.object(
        commands, "observe", return_value=adjacent_no_flag
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        cap = commands.capture(city_id="c17_5")
    assert cap.get("ok") is False and cap.get("reason") == "not_standing_on_city", cap
    assert clicks == []
    reset_obs()


def test_capture_opens_panel_when_already_on_city():
    """Live T41: /move-to already_on_city then /capture saw no Capture blob.

    Occupant is ON the hex but the unit panel is closed (or showing TRAIN).
    Open the sprite foot so the Capture pill appears, then click it.
    """
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c19_12",
        "kind": "city",
        "x": 201,
        "y": 80,
        "plate_y": 110,
        "tile": [19, 12],
        "tile_xy": [201, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    blob = {"x": 180, "y": 740, "n": 200}
    occupant = {"id": "u14", "x": 201, "y": 80, "owner": "own"}
    closed = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "cities_own": [],
        "units": [occupant],
        "unit": {"capture": False, "train": False, "unit": None},
        "ready": {},
        "overlay": {},
        "hud": {},
        "turn_diff": None,
    }
    opened = {
        **closed,
        "unit": {"capture": True, "no_actions": False, "unit": "warrior"},
        "ready": {"capture": True},
        "overlay": {"capture_blobs": [blob]},
    }
    owned = {
        **opened,
        "cities_enemy": [],
        "cities_own": [{**city, "owner": "own", "tribe": "bardur"}],
        "cities": [{**city, "owner": "own", "tribe": "bardur"}],
        "unit": {"capture": False},
        "ready": {},
    }
    observes = [opened, owned]
    with patch.object(commands, "remember", return_value=closed), patch.object(
        commands, "observe", side_effect=lambda *a, **k: observes.pop(0)
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ):
        cap = commands.capture(city_id="c19_12")
    assert cap.get("ok") is True and cap.get("captured") is True, cap
    assert cap.get("panel_opened") is True
    assert cap.get("stood_on_city") is True
    assert clicks, clicks
    assert clicks[0] != (180, 740), clicks
    assert clicks[-1] == (180, 740), clicks
    assert clicks[0][0] == 201
    assert clicks[0][1] >= 80
    reset_obs()


def test_capture_after_stood_on_city_hp_bar_offset():
    """Live T46: /move-to stood_on_city:true then /capture not_standing_on_city.

    HP bar sits ~0.7 hex off the walk foot — tight 0.55 missed; STANDING_HEX 0.8 hits.
    """
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c15_12",
        "kind": "city",
        "x": 201,
        "y": 80,
        "plate_y": 110,
        "tile": [15, 12],
        "tile_xy": [201, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    blob = {"x": 180, "y": 740, "n": 200}
    # ~0.61 hex east of the walk foot (plate_y-0.45*pitch ≈ 94).
    occupant = {"id": "u36", "x": 223, "y": 94, "owner": "own"}
    closed = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "cities_own": [],
        "units": [occupant],
        "unit": {"capture": False, "train": False, "unit": None},
        "ready": {},
        "overlay": {},
        "hud": {},
        "turn_diff": None,
    }
    opened = {
        **closed,
        "unit": {"capture": True, "no_actions": False, "unit": "warrior"},
        "ready": {"capture": True},
        "overlay": {"capture_blobs": [blob]},
    }
    owned = {
        **opened,
        "cities_enemy": [],
        "cities_own": [{**city, "owner": "own", "tribe": "bardur"}],
        "cities": [{**city, "owner": "own", "tribe": "bardur"}],
        "unit": {"capture": False},
        "ready": {},
    }
    observes = [opened, owned]
    with patch.object(commands, "remember", return_value=closed), patch.object(
        commands, "observe", side_effect=lambda *a, **k: observes.pop(0)
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ):
        cap = commands.capture(city_id="c15_12")
    assert cap.get("reason") != "not_standing_on_city", cap
    assert cap.get("stood_on_city") is True, cap
    assert cap.get("ok") is True and cap.get("captured") is True, cap
    assert clicks, clicks
    reset_obs()


def test_capture_soon_does_not_click_blindly():
    """Entering city this turn: panel says ready next turn — do not click Capture."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c19_12",
        "kind": "city",
        "x": 201,
        "y": 80,
        "plate_y": 110,
        "tile": [19, 12],
        "tile_xy": [201, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    occupant = {"id": "u14", "x": 201, "y": 80, "owner": "own"}
    closed = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "cities_own": [],
        "units": [occupant],
        "unit": {"capture": False},
        "ready": {},
        "overlay": {},
        "hud": {},
        "turn_diff": None,
    }
    soon = {
        **closed,
        "unit": {
            "capture": False,
            "capture_soon": True,
            "village": True,
            "raw": "Entering Village! Will be ready to capture next turn.",
        },
        "overlay": {},
    }
    with patch.object(commands, "remember", return_value=closed), patch.object(
        commands, "observe", return_value=soon
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ):
        cap = commands.capture(city_id="c19_12")
    assert cap.get("ok") is False, cap
    assert cap.get("reason") == "not_ready_until_next_turn", cap
    assert cap.get("next") == "end-turn"
    assert cap.get("captured") is False
    assert cap.get("stood_on_city") is True
    assert (180, 740) not in clicks
    reset_obs()


def test_attack_fast_no_mark_not_timeout():
    """Negative /attack must return no_mark quickly — not burn the 7.5s budget."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    city = {
        "id": "c5_3",
        "kind": "city",
        "x": 200,
        "y": 80,
        "tile": [5, 3],
        "tile_xy": [200, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    state = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "units": [{"id": "u2", "x": 164, "y": 80, "owner": "own"}],
        "unit": {"no_actions": False, "unit": "warrior"},
        "attack_marks": [],
    }
    with patch.object(commands, "remember", return_value=state), patch.object(
        commands, "observe", side_effect=AssertionError("no extra observe on no_mark")
    ), patch.object(
        commands, "select_unit", return_value={"ok": True, "unit": state["unit"]}
    ), patch.object(commands, "_click"), patch.object(commands, "_sleep"):
        r = commands.attack(from_id="u2", city_id="c5_3")
    assert r["ok"] is False
    assert r["reason"] in {"no_mark_on_target", "no_attack_marks", "out_of_range"}
    assert r["elapsed_s"] < 2
    reset_obs()


def test_select_unit_clicks_live_unit_not_sticky_ghost():
    """seen:false sticky + settings OCR must not be the click target."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_units, reset as reset_obs

    reset_obs()
    ghost = {
        "id": "u25", "kind": "unit", "x": 100, "y": 700, "owner": "own",
        "seen": False, "sticky": True, "tile": [3, 25],
    }
    live = {
        "id": "u25", "kind": "unit", "x": 640, "y": 162, "owner": "own",
        "seen": True, "tile": [17, 6],
    }
    register_units([ghost])
    mem = {
        "units": [live, ghost],
        "unit": {"settings": True, "can_move": False},
        "move_marks": [],
        "attack_marks": [],
        "ready": {},
        "hud": {},
        "layout": {"frame": [1280, 800]},
    }
    live_panel = {
        **mem,
        "unit": {"can_move": True, "settings": False, "unit": "warrior"},
        "move_marks": [{"x": 600, "y": 162, "n": 12}],
    }
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    with patch.object(commands, "remember", return_value=mem), patch.object(
        commands, "observe", return_value=live_panel
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ):
        r = commands.select_unit(id="u25", light=True)
    assert clicks, clicks
    assert clicks[0] == (640, 162), clicks
    assert r["ok"] is True
    reset_obs()


def test_dock_zone_blocks_end_turn_pixels():
    coords.reset()
    coords.set_frame(1280, 800)
    assert coords.in_dock_zone(765, 746)
    assert not coords.in_dock_zone(400, 400)
    zone = coords.dock_zone()
    assert zone["x0"] > 500 and zone["y0"] > 650


def test_find_units_keeps_mountain_south_of_city():
    """Live: first /observe dropped the Bardur on the mountain south of Disrof (top-16 + snow)."""
    im = Image.new("RGB", (1280, 800), (30, 40, 28))
    d = ImageDraw.Draw(im)
    for i in range(40):
        x = 80 + (i % 5) * 90
        y = 430 + (i // 5) * 28
        d.rectangle((x, y, x + 36, y + 6), fill=(90, 210, 50))
    d.rectangle((580, 248, 640, 320), fill=(240, 245, 250))
    d.rectangle((600, 258, 616, 263), fill=(90, 210, 50))
    d.rectangle((598, 272, 618, 298), fill=(90, 85, 80))
    arr = as_rgb(im)
    cities = [{"x": 600, "y": 144, "tile_xy": [600, 144], "plate_y": 170, "tribe": "vengir", "owner": "enemy"}]
    units = find_units(arr, cities=cities)
    near = [u for u in units if abs(int(u["x"]) - 608) < 45 and 250 <= int(u["y"]) <= 320]
    assert near, [(u["id"], u["x"], u["y"], u.get("n")) for u in units]
    assert any(str(u.get("owner") or "") in {"own", "unknown"} for u in near), near


def test_select_unit_prefers_live_over_ghost_sticky():
    """Live: u25 seen:false sticky clicked Settings; the warrior was at (607,277)."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_units, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    register_units([
        {"id": "u25", "kind": "unit", "x": 100, "y": 100, "owner": "own", "seen": False, "sticky": True},
    ])
    live = {"id": "u25", "kind": "unit", "x": 607, "y": 277, "owner": "own", "seen": True}
    state = {
        "layout": {"frame": [1280, 800]},
        "units": [live],
        "unit": {"can_move": True, "unit": "warrior", "settings": False, "no_actions": False},
        "move_marks": [{"x": 599, "y": 160, "n": 20}],
        "attack_marks": [],
        "hud": {},
        "ready": {},
        "overlay": {},
        "turn_diff": None,
    }
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    with patch.object(commands, "remember", return_value=state), patch.object(
        commands, "observe", return_value=state
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.select_unit(id="u25", light=True)
    assert r["ok"] is True, r
    assert clicks and clicks[0] == (607, 277), clicks
    assert (100, 100) not in clicks
    reset_obs()


def test_move_to_clicks_foot_not_roof_centroid():
    """Live: move_hex centroid at tile_xy [599,142] selected the building; unit stayed on the mountain."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = _disrof_city()
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    roof = {"id": "m_roof", "x": 200, "y": 81, "n": 40, "kind": "move_hex"}
    before = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "cities_own": [],
        "move_marks": [roof],
        "units": [{"id": "u1", "x": 164, "y": 80, "owner": "own", "seen": True}],
        "unit": {"can_move": True, "no_actions": False, "unit": "rider", "settings": False},
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    after = {
        **before,
        "move_marks": [],
        "units": [{"id": "u1", "x": 201, "y": 94, "owner": "own", "seen": True}],
        "unit": {"can_move": False, "capture": True, "no_actions": False, "unit": "rider"},
        "ready": {"capture": True},
    }
    selected = {"ok": True, "unit": before["unit"]}
    with patch.object(commands, "remember", return_value=before), patch.object(
        commands, "observe", return_value=after
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u1", city_id="c5_3")
    assert clicks, clicks
    assert clicks[0][1] > 81, clicks
    assert (200, 80) not in clicks and (200, 81) not in clicks
    assert r["ok"] is True and r["stood_on_city"] is True, r
    reset_obs()


def test_city_occupied_attack_then_stand_and_capture():
    """Live T39: /move-to Disrof returned city_occupied; cannot stand while garrison lives.

    Path: attack until garrison dead → move-to stood_on_city → capture.
    Roof-blue must not be clicked while the defender is on the tile. Raft
    cannot hit land — suggest the adjacent Bardur warrior.
    """
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.combat import land_attacker_near
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c17_5",
        "kind": "city",
        "x": 600,
        "y": 144,
        "plate_y": 174,
        "tile": [17, 5],
        "tile_xy": [600, 144],
        "tribe": "vengir",
        "owner": "enemy",
        "n": 400,
        "confidence": 0.78,
        "evidence": ["frost_plate", "magenta_roof"],
    }
    register_cities([city])
    garrison = {
        "id": "u40", "kind": "unit", "x": 602, "y": 148, "hp": 10, "n": 40,
        "owner": "enemy", "seen": True,
    }
    land = {
        "id": "u25", "kind": "unit", "x": 607, "y": 180, "owner": "own",
        "unit": "warrior", "seen": True,
    }
    raft = {
        "id": "u12", "kind": "unit", "x": 636, "y": 144, "owner": "own",
        "unit": "raft", "seen": True,
    }
    near = land_attacker_near([garrison, land, raft], city, (1280, 800))
    assert near and near["id"] == "u25", near

    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    base = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "cities_own": [],
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    occupied = {
        **base,
        "units": [garrison, land, raft],
        "unit": {"can_move": True, "no_actions": False, "unit": "warrior"},
        "move_marks": [{"x": 600, "y": 145, "n": 40, "kind": "move"}],
        "attack_marks": [{"x": 599, "y": 158, "n": 30}],
    }
    selected_land = {"ok": True, "unit": occupied["unit"]}
    with patch.object(commands, "remember", return_value=occupied), patch.object(
        commands, "observe", return_value=occupied
    ), patch.object(commands, "select_unit", return_value=selected_land), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        mv = commands.move_to(from_id="u25", city_id="c17_5")
    assert mv["ok"] is False and mv["reason"] == "city_occupied", mv
    assert mv["stood_on_city"] is False
    assert mv.get("next") == "attack"
    assert mv.get("garrison_id") == "u40"
    assert mv.get("attacker_id") == "u25"
    assert mv.get("next_call", {}).get("path") == "/attack"
    assert mv["next_call"]["body"]["from_id"] == "u25"
    assert mv["next_call"]["body"]["city_id"] == "c17_5"
    assert clicks == [], clicks

    clicks.clear()
    before_atk = {
        **occupied,
        "unit": {"no_actions": False, "unit": "warrior", "can_move": False},
        "attack_marks": [{"x": 599, "y": 158, "n": 30}],
        "move_marks": [],
    }
    after_hit = {
        **before_atk,
        "units": [{**garrison, "hp": 5, "n": 20}, land, raft],
    }
    with patch.object(commands, "remember", return_value=before_atk), patch.object(
        commands, "observe", return_value=after_hit
    ), patch.object(
        commands, "select_unit",
        return_value={"ok": True, "unit": before_atk["unit"]},
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ):
        atk = commands.attack(from_id="u25", city_id="c17_5")
    assert atk["ok"] is True and atk["hp_dropped"] is True, atk
    assert atk.get("garrison_dead") is False
    assert atk.get("next") == "attack"
    assert atk["elapsed_s"] < 8
    assert clicks, clicks

    clicks.clear()
    after_kill = {
        **before_atk,
        "units": [land, raft],
        "attack_marks": [],
        "move_marks": [{"x": 600, "y": 158, "n": 40}],
    }
    with patch.object(commands, "remember", return_value=before_atk), patch.object(
        commands, "observe", return_value=after_kill
    ), patch.object(
        commands, "select_unit",
        return_value={"ok": True, "unit": before_atk["unit"]},
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ):
        atk2 = commands.attack(from_id="u25", city_id="c17_5")
    assert atk2["ok"] is True and atk2["hp_dropped"] is True, atk2
    assert atk2.get("garrison_dead") is True
    assert atk2.get("next") == "move-to"
    assert atk2.get("next_call", {}).get("path") == "/move-to"

    clicks.clear()
    raft_sel = {
        **occupied,
        "unit": {"no_actions": False, "unit": "raft", "can_move": True},
    }
    with patch.object(commands, "remember", return_value=raft_sel), patch.object(
        commands, "observe", return_value=raft_sel
    ), patch.object(
        commands, "select_unit",
        return_value={"ok": True, "unit": raft_sel["unit"]},
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ):
        raft_atk = commands.attack(from_id="u12", city_id="c17_5")
    assert raft_atk["ok"] is False and raft_atk["reason"] == "naval_no_land", raft_atk
    assert raft_atk.get("attacker_id") == "u25"
    assert raft_atk.get("next_call", {}).get("body", {}).get("from_id") == "u25"
    assert clicks == [], clicks

    clicks.clear()
    empty_city = {
        **base,
        "units": [land, raft],
        "unit": {"can_move": True, "no_actions": False, "unit": "warrior"},
        "move_marks": [{"x": 600, "y": 158, "n": 50}],
        "attack_marks": [],
    }
    after_stand = {
        **empty_city,
        "units": [{**land, "x": 600, "y": 158}, raft],
        "unit": {"can_move": False, "capture": True, "no_actions": False, "unit": "warrior"},
        "ready": {"capture": True},
        "move_marks": [],
    }
    with patch.object(commands, "remember", return_value=empty_city), patch.object(
        commands, "observe", return_value=after_stand
    ), patch.object(commands, "select_unit", return_value=selected_land), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        stand = commands.move_to(from_id="u25", city_id="c17_5")
    assert stand["ok"] is True and stand["stood_on_city"] is True, stand
    assert clicks, clicks

    clicks.clear()
    panel_blob = {"x": 180, "y": 740, "n": 200}
    on_city = {
        **after_stand,
        "overlay": {"capture_blobs": [panel_blob]},
        "ready": {"capture": True},
        "unit": {"capture": True, "no_actions": False, "unit": "warrior"},
    }
    captured_state = {
        **on_city,
        "cities_enemy": [],
        "cities_own": [{**city, "owner": "own", "tribe": "bardur"}],
        "cities": [{**city, "owner": "own", "tribe": "bardur"}],
    }
    with patch.object(commands, "remember", return_value=on_city), patch.object(
        commands, "observe", return_value=captured_state
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ):
        cap = commands.capture(city_id="c17_5")
    assert cap["ok"] is True and cap["captured"] is True, cap
    assert clicks == [(180, 740)], clicks
    reset_obs()


def test_attack_on_hex_finds_foot_ring():
    from polytopia_api.combat import hex_pitch
    from polytopia_api.detect import as_rgb, attack_on_hex, attack_tint_at, find_attack_marks

    im = Image.new("RGB", (1280, 800), (20, 20, 20))
    px = im.load()
    for y in range(90, 100):
        for x in range(196, 206):
            px[x, y] = (210, 60, 55)
    arr = as_rgb(im)
    hit = attack_on_hex(arr, 200, 94, hex_pitch(1280, 800))
    assert hit["ok"] is True, hit
    assert abs(hit["x"] - 200) < 10
    assert 88 <= hit["y"] <= 102
    tint = attack_tint_at(arr, 200, 94, radius=18, min_n=5)
    assert tint["ok"] is True, tint
    marks = find_attack_marks(arr)
    assert marks, marks


def test_attack_clicks_city_foot_not_hp_bar():
    """Live: clustered red sat on the garrison HP bar; click missed, hp_dropped false."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c19_24",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [19, 24],
        "tile_xy": [200, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    garrison = {"id": "u8", "kind": "unit", "x": 201, "y": 82, "hp": 10, "n": 40, "owner": "enemy"}
    land = {"id": "u13", "x": 164, "y": 80, "owner": "own", "unit": "warrior"}
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    before = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "units": [garrison, land],
        "unit": {"no_actions": False, "unit": "warrior", "can_move": False},
        "attack_marks": [{"x": 203, "y": 83, "n": 40}],
        "overlay": {},
        "ready": {},
        "hud": {},
    }
    after = {
        **before,
        "units": [{**garrison, "hp": 5, "n": 20}, land],
    }
    with patch.object(commands, "remember", return_value=before), patch.object(
        commands, "observe", return_value=after
    ), patch.object(
        commands, "select_unit",
        return_value={"ok": True, "unit": before["unit"]},
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ):
        r = commands.attack(from_id="u13", city_id="c19_24")
    assert r["ok"] is True and r["hp_dropped"] is True, r
    assert r.get("garrison_dead") is False
    assert clicks, clicks
    assert clicks[0] == (200, 94), clicks
    assert clicks[0][1] > 83, clicks
    reset_obs()


def test_attack_waits_for_late_red_mark():
    """Light select often screenshots before the Vengir hex turns red."""
    import tempfile
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c19_24",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [19, 24],
        "tile_xy": [200, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    garrison = {"id": "u8", "kind": "unit", "x": 201, "y": 82, "hp": 10, "n": 40, "owner": "enemy"}
    land = {"id": "u13", "x": 164, "y": 80, "owner": "own", "unit": "warrior"}
    empty = Image.new("RGB", (1280, 800), (20, 20, 20))
    red = empty.copy()
    rd = ImageDraw.Draw(red)
    rd.rectangle((194, 88, 206, 100), fill=(210, 60, 55))
    tmp = Path(tempfile.mkdtemp())
    empty_path = str(tmp / "empty.png")
    red_path = str(tmp / "red.png")
    empty.save(empty_path)
    red.save(red_path)
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    blank = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "units": [garrison, land],
        "unit": {"no_actions": False, "unit": "warrior", "can_move": False},
        "attack_marks": [],
        "screenshot": empty_path,
        "overlay": {},
        "ready": {},
        "hud": {},
    }
    lit = {**blank, "screenshot": red_path}
    dropped = {
        **lit,
        "units": [{**garrison, "hp": 7, "n": 28}, land],
    }
    observes = {"n": 0}

    def fake_observe(**kwargs):
        observes["n"] += 1
        return dropped if observes["n"] > 1 else lit

    with patch.object(commands, "remember", return_value=blank), patch.object(
        commands, "observe", side_effect=fake_observe
    ), patch.object(
        commands, "select_unit",
        return_value={"ok": True, "unit": blank["unit"]},
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ):
        r = commands.attack(from_id="u13", city_id="c19_24")
    assert r["ok"] is True and r["hp_dropped"] is True, r
    assert clicks, clicks
    assert clicks[0] == (200, 94), clicks
    assert observes["n"] >= 2
    reset_obs()


def test_attack_retries_when_hp_stays():
    """First foot click can miss; re-select and click a nudged foot until HP drops."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c19_24",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [19, 24],
        "tile_xy": [200, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    garrison = {"id": "u8", "kind": "unit", "x": 201, "y": 82, "hp": 10, "n": 40, "owner": "enemy"}
    land = {"id": "u13", "x": 164, "y": 80, "owner": "own", "unit": "warrior"}
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    still = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "units": [garrison, land],
        "unit": {"no_actions": False, "unit": "warrior", "can_move": False},
        "attack_marks": [{"x": 200, "y": 94, "n": 30}],
        "overlay": {},
        "ready": {},
        "hud": {},
    }
    dropped = {
        **still,
        "units": [{**garrison, "hp": 4, "n": 16}, land],
    }
    observes = {"n": 0}

    def fake_observe(**kwargs):
        observes["n"] += 1
        return dropped if observes["n"] > 3 else still

    with patch.object(commands, "remember", return_value=still), patch.object(
        commands, "observe", side_effect=fake_observe
    ), patch.object(
        commands, "select_unit",
        return_value={"ok": True, "unit": still["unit"]},
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(
        commands, "_sleep"
    ):
        r = commands.attack(from_id="u13", city_id="c19_24")
    assert r["ok"] is True and r["hp_dropped"] is True, r
    assert r.get("strike_retried") is True, r
    assert len(clicks) >= 2, clicks
    assert clicks[0] == (200, 94), clicks
    reset_obs()


def test_attack_refuses_own_garrison():
    """Live T46: city labeled enemy, garrison was Bardur own — hp_dropped:false poison."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c15_12",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [15, 12],
        "tile_xy": [200, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    own_on_tile = {"id": "u4", "kind": "unit", "x": 201, "y": 82, "hp": 10, "n": 40, "owner": "own"}
    land = {"id": "u36", "x": 164, "y": 80, "owner": "own", "unit": "warrior"}
    before = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "units": [own_on_tile, land],
        "unit": {"no_actions": False, "unit": "warrior", "can_move": False},
        "attack_marks": [{"x": 203, "y": 83, "n": 40}],
        "overlay": {},
        "ready": {},
        "hud": {},
    }
    with patch.object(commands, "remember", return_value=before), patch.object(
        commands, "observe", return_value=before
    ), patch.object(
        commands, "select_unit",
        return_value={"ok": True, "unit": before["unit"], "attack_marks": before["attack_marks"]},
    ), patch.object(commands, "_click", side_effect=fake_click), patch.object(commands, "_sleep"):
        r = commands.attack(from_id="u36", city_id="c15_12")
    assert r["ok"] is False and r["reason"] == "own_garrison", r
    assert r.get("hp_dropped") is False
    assert clicks == []
    reset_obs()


def test_move_to_adjacent_empty_marks_clicks_city_foot():
    """Live T39: u3→c14_24 tile_dist≈0.94 / move_range=1 returned no_move_marks."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, register_units, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c14_24",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [14, 24],
        "tile_xy": [200, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    register_units([{"id": "u3", "kind": "unit", "x": 164, "y": 80, "owner": "own"}])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    before = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "move_marks": [],
        "units": [{"id": "u3", "x": 164, "y": 80, "owner": "own", "seen": True}],
        "unit": {"can_move": True, "no_actions": False, "unit": "warrior", "settings": False},
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    after = {
        **before,
        "units": [{"id": "u3", "x": 201, "y": 94, "owner": "own", "seen": True}],
        "unit": {"can_move": False, "capture": True, "no_actions": False, "unit": "warrior"},
        "ready": {"capture": True},
    }
    selected = {"ok": True, "unit": before["unit"]}
    with patch.object(commands, "remember", return_value=before), patch.object(
        commands, "observe", return_value=after
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u3", city_id="c14_24")
    assert r.get("reason") != "no_move_marks", r
    assert r.get("reason") != "need x,y or city_id/to_id", r
    assert clicks, clicks
    assert clicks[0][1] > 81, clicks
    assert (200, 80) not in clicks
    assert r["ok"] is True and r["stood_on_city"] is True, r
    reset_obs()


def test_move_to_waits_for_walk_then_stood_on_city():
    """Live T41: city_foot click returned not_stood_on_city while the sprite was still adjacent."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, register_units, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = _disrof_city()
    register_cities([city])
    register_units([{"id": "u36", "kind": "unit", "x": 164, "y": 80, "owner": "own"}])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    foot = {"id": "m_foot", "x": 200, "y": 104, "n": 22}
    before = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "move_marks": [foot],
        "units": [{"id": "u36", "x": 164, "y": 80, "owner": "own", "seen": True}],
        "unit": {"can_move": True, "no_actions": False, "unit": "warrior", "settings": False},
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    stood = {
        **before,
        "move_marks": [],
        "units": [{"id": "u36", "x": 201, "y": 94, "owner": "own", "seen": True}],
        "unit": {"can_move": False, "capture": True, "no_actions": False, "unit": "warrior"},
        "ready": {"capture": True},
    }
    observes = [before, stood]
    selects = {"n": 0}

    def fake_observe(mode="full"):
        nxt = observes.pop(0) if observes else stood
        return nxt

    def fake_select(**_kw):
        selects["n"] += 1
        return {"ok": True, "unit": before["unit"]}

    with patch.object(commands, "remember", return_value=before), patch.object(
        commands, "observe", side_effect=fake_observe
    ), patch.object(commands, "select_unit", side_effect=fake_select), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u36", city_id="c5_3")
    assert clicks and clicks[0] == (200, 104), clicks
    assert (200, 80) not in clicks
    assert selects["n"] == 1, selects
    assert r.get("stand_retried") is False, r
    assert r["ok"] is True and r["stood_on_city"] is True, r
    reset_obs()


def test_move_to_reselects_when_foot_click_stays_adjacent():
    """Foot click + wait still adjacent → re-select and click the ring again until ON tile."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, register_units, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = _disrof_city()
    register_cities([city])
    register_units([{"id": "u36", "kind": "unit", "x": 164, "y": 80, "owner": "own"}])
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    foot = {"x": 200, "y": 104, "n": 22}
    before = {
        "layout": {"frame": [1280, 800]},
        "cities": [city],
        "cities_enemy": [city],
        "move_marks": [foot],
        "units": [{"id": "u36", "x": 164, "y": 80, "owner": "own", "seen": True}],
        "unit": {"can_move": True, "no_actions": False, "unit": "warrior", "settings": False},
        "overlay": {},
        "ready": {},
        "hud": {},
        "turn_diff": None,
    }
    reselected = {
        **before,
        "move_marks": [foot],
        "unit": {"can_move": True, "no_actions": False, "unit": "warrior"},
    }
    stood = {
        **before,
        "move_marks": [],
        "units": [{"id": "u36", "x": 201, "y": 94, "owner": "own", "seen": True}],
        "unit": {"can_move": False, "capture": True, "no_actions": False, "unit": "warrior"},
        "ready": {"capture": True},
    }
    mem = {"state": before}
    observes = {"n": 0}
    selects = {"n": 0}

    def fake_remember():
        return mem["state"]

    def fake_observe(mode="full"):
        observes["n"] += 1
        if selects["n"] >= 2:
            mem["state"] = stood
            return stood
        return before

    def fake_select(**_kw):
        selects["n"] += 1
        if selects["n"] >= 2:
            mem["state"] = reselected
            return {"ok": True, "unit": reselected["unit"]}
        return {"ok": True, "unit": before["unit"]}

    with patch.object(commands, "remember", side_effect=fake_remember), patch.object(
        commands, "observe", side_effect=fake_observe
    ), patch.object(commands, "select_unit", side_effect=fake_select), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u36", city_id="c5_3")
    assert selects["n"] >= 2, selects
    assert r.get("stand_retried") is True, r
    assert clicks and all(c[1] > 81 for c in clicks), clicks
    assert (200, 80) not in clicks
    assert r["ok"] is True and r["stood_on_city"] is True, r
    reset_obs()


def test_second_move_to_resolves_sticky_city_id():
    """Live T39: follow-up u4→c14_24 said need x,y despite passing city_id."""
    from unittest.mock import patch

    from polytopia_api import commands
    from polytopia_api.observe import register_cities, reset as reset_obs

    reset_obs()
    coords.reset()
    coords.set_frame(1280, 800)
    city = {
        "id": "c14_24",
        "kind": "city",
        "x": 200,
        "y": 80,
        "plate_y": 110,
        "tile": [14, 24],
        "tile_xy": [200, 80],
        "tribe": "vengir",
        "owner": "enemy",
    }
    register_cities([city])
    blank = {
        "layout": {"frame": [1280, 800]},
        "cities": [],
        "cities_own": [],
        "cities_enemy": [],
        "units": [{"id": "u4", "x": 164, "y": 80, "owner": "own"}],
        "unit": {"can_move": True, "no_actions": False, "unit": "warrior"},
        "move_marks": [],
        "attack_marks": [],
        "overlay": {},
        "ready": {},
        "hud": {"turn": 3, "turn_trusted": False},
        "turn_diff": None,
    }
    after = {
        **blank,
        "units": [{"id": "u4", "x": 201, "y": 94, "owner": "own"}],
        "unit": {"can_move": False, "capture": True, "unit": "warrior"},
        "ready": {"capture": True},
    }
    clicks: list[tuple[int, int]] = []

    def fake_click(x, y, space="screen", repeats=1):
        clicks.append((int(x), int(y)))
        return {"ok": True, "x": int(x), "y": int(y)}

    selected = {"ok": True, "unit": blank["unit"]}
    with patch.object(commands, "remember", return_value=blank), patch.object(
        commands, "observe", return_value=after
    ), patch.object(commands, "select_unit", return_value=selected), patch.object(
        commands, "_click", side_effect=fake_click
    ), patch.object(commands, "_sleep"):
        r = commands.move_to(from_id="u4", city_id="c14_24")
    assert r.get("reason") != "need x,y or city_id/to_id", r
    assert r.get("dest") and r["dest"]["id"] == "c14_24", r
    assert clicks, clicks
    reset_obs()


def test_lookup_city_id_ignores_units():
    from polytopia_api.observe import lookup, register_cities, reset as reset_obs

    reset_obs()
    city = {"id": "c14_24", "city_id": "c14_24", "kind": "city", "x": 200, "y": 80, "tile": [14, 24]}
    register_cities([city])
    mem = {
        "cities": [],
        "cities_enemy": [],
        "units": [{"id": "u4", "x": 10, "y": 10, "near_city_id": "c14_24", "city_id": "c14_24"}],
        "move_marks": [{"id": "c14_24", "x": 1, "y": 1}],
    }
    hit = lookup(mem, "c14_24")
    assert hit and hit.get("kind") == "city" and hit.get("x") == 200, hit
    reset_obs()


if __name__ == "__main__":
    test_parse_hud()
    test_parse_unit()
    test_water_vs_move()
    test_find_move_marks_ignores_water()
    test_empirical_end_turn_1280()
    test_find_move_marks_1280()
    test_entities_on_synthetic_map()
    test_capture_and_recruit_targets()
    test_mcp_lists_local_tools()
    test_find_window_ignores_wrapper_and_chrome()
    test_plan_priorities()
    test_hud_star_split()
    test_stable_entity_ids()
    test_gold_windows_are_not_oumaji()
    test_disrof_gold_windows_without_magenta_sample()
    test_bardur_own_not_eaten_by_vengir_frost()
    test_bardur_gold_star_stays_own()
    test_m20_bardur_own_and_one_disrof()
    test_frost_plate_gold_windows_stay_enemy_not_own()
    test_1280_bardur_own_and_plate_window_disrof()
    test_attack_timeout_skips_second_observe()
    test_snapshot_alerts()
    test_snapshot_refuses_stale_turn()
    test_observe_hud_untrusts_ocr_behind_floor()
    test_observe_hud_floor_from_tfile_without_latest()
    test_snapshot_dir_ignores_empty_cwd()
    test_find_train_blob_1280()
    test_doit_blobs_ignore_map_noise()
    test_attack_garrison_hp_and_on_city()
    test_attack_marks_and_strike()
    test_win_path_move_capture_attack()
    test_city_id_stays_resolvable_after_observe_drops_it()
    test_vengir_fps_do_not_accumulate_across_frames()
    test_twelve_own_cities_do_not_wipe_enemy()
    test_t44_dark_tile_vengir_stay_cities_enemy()
    test_dark_magenta_roof_classifies_vengir()
    test_session_cities_keeps_sticky_enemy_when_own_is_full()
    test_twelve_own_cities_do_not_drop_enemy_or_flood_frost()
    test_late_domination_observe_keeps_visible_vengir()
    test_nearby_vengir_cities_do_not_merge()
    test_nearby_bardur_not_eaten_by_disrof()
    test_crowded_bardur_own_and_vengir_enemy()
    test_bardur_wood_shadow_stays_own()
    test_two_bardur_match_screen_not_frost_phantoms()
    test_three_bardur_not_hud_fog_c18_1()
    test_two_bardur_dedupe_orkork_split_and_mid_phantom()
    test_unit_frost_c17_14_not_cities_own()
    test_own_phantoms_do_not_accumulate_across_frames()
    test_recruit_timeout_skips_full_observe()
    test_recruit_uses_marks_observe_for_train()
    test_recruit_clicks_roof_then_train()
    test_recruit_backs_off_disband_then_clicks_train()
    test_recruit_clicks_brand_blue_panel_pill()
    test_recruit_clicks_train_despite_leftover_unit_ocr()
    test_recruit_waits_for_train_after_nameplate()
    test_adjacent_bardur_cities_stay_two_own()
    test_unit_id_stays_resolvable_after_observe_drops_it()
    test_move_to_clicks_blue_ring_at_city_foot()
    test_move_to_never_returns_ok_null()
    test_move_to_does_not_click_adjacent_mark()
    test_move_to_no_mark_does_not_click_building()
    test_move_to_out_of_range_reason()
    test_move_to_ok_false_when_still_adjacent()
    test_move_to_reselects_after_city_panel()
    test_move_to_xy_walks_onto_city_not_plate()
    test_city_tile_center_lifts_plate_xy()
    test_move_on_hex_finds_foot_ring()
    test_observe_detects_unit_on_mountain_south_of_disrof()
    test_leather_unit_south_of_disrof_without_hp_bar()
    test_observe_keeps_disrof_with_unit_on_east_port()
    test_observe_keeps_disrof_when_garrison_covers_roof()
    test_observe_east_port_unit_without_hp_bar()
    test_sticky_bardur_stays_own_if_frame_flips_vengir()
    test_sticky_disrof_stays_enemy_if_frame_flips_bardur()
    test_stabilize_units_matches_by_tile()
    test_move_to_from_mountain_unit_stands_on_city()
    test_capture_when_on_disrof_with_capture_flag()
    test_capture_opens_panel_when_already_on_city()
    test_capture_after_stood_on_city_hp_bar_offset()
    test_capture_soon_does_not_click_blindly()
    test_attack_fast_no_mark_not_timeout()
    test_dock_zone_blocks_end_turn_pixels()
    test_find_units_keeps_mountain_south_of_city()
    test_select_unit_prefers_live_over_ghost_sticky()
    test_move_to_clicks_foot_not_roof_centroid()
    test_city_occupied_attack_then_stand_and_capture()
    test_attack_on_hex_finds_foot_ring()
    test_attack_clicks_city_foot_not_hp_bar()
    test_attack_waits_for_late_red_mark()
    test_attack_retries_when_hp_stays()
    test_attack_refuses_own_garrison()
    test_move_to_adjacent_empty_marks_clicks_city_foot()
    test_move_to_waits_for_walk_then_stood_on_city()
    test_move_to_reselects_when_foot_click_stays_adjacent()
    test_second_move_to_resolves_sticky_city_id()
    test_lookup_city_id_ignores_units()
    print("ok")
