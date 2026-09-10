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
    assert "bardur" in tribes and "oumaji" in tribes, cities
    assert villages, villages
    assert classify_tribe((210, 180, 60)) == "oumaji"
    assert classify_tribe((90, 85, 80)) == "bardur"
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
        "overlay": {"do_it_blobs": [{"x": 10, "y": 20}], "train_blobs": []},
        "ready": {"capture": True},
        "unit": {"capture": True},
    }
    assert capture_target(obs) == (10, 20)
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
        "overlay": {**base_overlay, "do_it_pixel": True, "do_it_blobs": [{"x": 1, "y": 2}]},
        "confirm_ready": True,
        "move_marks": [],
        "fruit": [],
    }
    assert plan(obs)["name"] == "capture"

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


def test_find_train_blob_1280():
    im = Image.new("RGB", (1280, 800), (20, 20, 20))
    d = ImageDraw.Draw(im)
    d.rectangle((700, 500, 780, 560), fill=(10, 122, 204))
    blobs = find_train_buttons(as_rgb(im))
    assert blobs, blobs
    assert abs(blobs[0]["x"] - 740) < 30
    assert abs(blobs[0]["y"] - 530) < 30


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
    test_snapshot_alerts()
    test_snapshot_refuses_stale_turn()
    test_find_train_blob_1280()
    print("ok")
