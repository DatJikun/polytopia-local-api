from PIL import Image

from polytopia_api import coords
from polytopia_api.detect import find_move_marks, is_move_rgb, is_water_rgb
from polytopia_api.driver import parse_hud, parse_unit_panel
from polytopia_api.play import plan


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


def test_parse_unit():
    u = parse_unit_panel("Bardur Warrior\nSelect a blue mark to move.")
    assert u["can_move"] and u["unit"] == "warrior" and not u["no_actions"]
    u = parse_unit_panel("Bardur Catapult\nNo actions left. Press 'Next Turn' to move this unit again.")
    assert u["no_actions"] and u["unit"] == "catapult"
    u = parse_unit_panel("Harvest Fruit\nExtract this resource to upgrade your city.")
    assert u["harvest"]
    u = parse_unit_panel("Clear Forest")
    assert u["clear_forest"]


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
    # water lake
    for y in range(700, 820):
        for x in range(900, 1100):
            px[x, y] = (58, 199, 249)
    # move highlight blob
    for y in range(590, 610):
        for x in range(1230, 1255):
            px[x, y] = (146, 204, 255)
    marks = find_move_marks(__import__("numpy").asarray(im).astype("int16"))
    assert marks, marks
    assert all(abs(m["x"] - 1242) < 20 and abs(m["y"] - 600) < 20 for m in marks)


def test_scale_1280x800():
    coords.set_frame(1920, 1200)
    assert tuple(coords.END_TURN) == (1111, 1130)
    assert tuple(coords.DO_IT) == (1117, 681)

    coords.set_frame(1280, 800)
    sx, sy = 1280 / 1920, 800 / 1200
    assert abs(sx - sy) < 1e-9  # same 16:10
    ex, ey = tuple(coords.END_TURN)
    assert ex == round(1111 * sx)
    assert ey == round(1130 * sy)
    dx, dy = tuple(coords.DO_IT)
    assert dx == round(1117 * sx)
    assert dy == round(681 * sy)
    assert coords.xy(1111, 1130) == (ex, ey)  # POST /click space=design
    # HUD crop stays in the top band
    hud = tuple(coords.HUD_CROP)
    assert hud[1] == 0 and hud[3] < 80
    info = coords.layout_info()
    assert info["frame"] == [1280, 800]
    assert info["scale"] == [0.6667, 0.6667]
    coords.set_frame(1920, 1200)


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


def test_plan_priorities():
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


if __name__ == "__main__":
    test_parse_hud()
    test_parse_unit()
    test_water_vs_move()
    test_find_move_marks_ignores_water()
    test_scale_1280x800()
    test_find_move_marks_1280()
    test_plan_priorities()
    print("ok")
