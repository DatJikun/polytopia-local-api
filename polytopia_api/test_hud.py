from pathlib import Path

import os

from PIL import Image, ImageDraw

from polytopia_api import coords
from polytopia_api.commands import _city_click_points, capture_target, recruit_target
from polytopia_api.detect import as_rgb, find_move_marks, find_train_buttons, is_move_rgb, is_water_rgb
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
    u = parse_unit_panel("Bardur Raft\nSelect a blue mark to move.")
    assert u["unit"] == "raft" and u["can_move"]


def test_water_vs_move():
    assert is_water_rgb(111, 207, 247)
    assert is_water_rgb(58, 199, 249)
    assert is_water_rgb(143, 255, 255)
    assert is_move_rgb(146, 204, 255)
    assert not is_move_rgb(111, 207, 247)
    assert not is_move_rgb(58, 199, 249)


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
    # own (Bardur) city: grey building + white nameplate
    d.rectangle((880, 360, 940, 420), fill=(90, 85, 80))
    d.rectangle((860, 430, 960, 448), fill=(240, 240, 235))
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
    assert capture_target(obs) is None  # map DO IT is not Capture
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
    assert recruit_target(obs) == (3, 4)
    coords.set_frame(1280, 800)
    ocr_only = {
        "overlay": {"train_blobs": [], "train_pixel": False},
        "ready": {"train": True},
        "unit": {"train": True},
    }
    assert recruit_target(ocr_only) is None
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
    assert clicks == [(202, 81)], clicks

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
    assert r["ok"] is False and r["reason"] == "no_mark_on_city_tile"
    assert r["stood_on_city"] is False
    assert "move_marks" in r and r.get("city_xy")
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
    assert clicks == [(203, 83)], clicks


def test_dock_zone_blocks_end_turn_pixels():
    coords.reset()
    coords.set_frame(1280, 800)
    assert coords.in_dock_zone(765, 746)
    assert not coords.in_dock_zone(400, 400)
    zone = coords.dock_zone()
    assert zone["x0"] > 500 and zone["y0"] > 650


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
    test_dock_zone_blocks_end_turn_pixels()
    print("ok")
