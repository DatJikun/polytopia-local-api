"""Local HTTP API wrapping the Polytopia Unity window.

  python3 -m polytopia_api.server   # http://127.0.0.1:8765
  python3 -m polytopia_api.mcp      # MCP stdio → same server (local_* tools)

think_turn is the brain. /step is a playbook fallback only.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "polytopia_api"

from . import commands
from . import coords
from . import driver
from . import snapshot
from .observe import observe as do_observe
from .play import plan as do_plan
from .play import play_n
from .play import step as do_step

HOST = "127.0.0.1"
PORT = 8765

ROUTES = {
    "GET": [
        "/",
        "/health",
        "/hud",
        "/observe",
        "/diff",
        "/pixel",
        "/screenshot",
    ],
    "POST": [
        "/click",
        "/tech",
        "/settings",
        "/confirm",
        "/end-turn",
        "/back",
        "/key",
        "/calibrate",
        "/select-unit",
        "/move-to",
        "/attack",
        "/capture",
        "/recruit",
        "/snapshot",
        "/step",
        "/plan",
        "/play",
    ],
}


def _json(handler: BaseHTTPRequestHandler, code: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise driver.PolytopiaError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise driver.PolytopiaError("JSON body must be an object")
    return data


def _xy(body: dict, *keys: str) -> tuple[int | None, int | None]:
    for a, b in (keys[i : i + 2] for i in range(0, len(keys), 2)):
        if body.get(a) is not None and body.get(b) is not None:
            return int(body[a]), int(body[b])
    return None, None


def canonicalize_path(path: str) -> str:
    """`/select_unit` and `/select-unit` are the same route."""
    path = (path or "/").rstrip("/") or "/"
    if path != "/":
        path = path.replace("_", "-")
    return path


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[polytopia-api] " + (fmt % args) + "\n")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = canonicalize_path(parsed.path)
        q = parse_qs(parsed.query)
        try:
            if path == "/":
                _json(self, 200, {
                    "ok": True,
                    "mcp": [
                        "local_health",
                        "local_observe",
                        "local_hud",
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
                        "local_back",
                        "local_confirm",
                        "local_calibrate",
                        "local_step",
                    ],
                    "routes": ROUTES,
                })
                return
            if path == "/health":
                info = driver.find_window()
                driver.sync_layout(info)
                _json(self, 200, {
                    "ok": True,
                    "game": info,
                    "layout": coords.layout_info(),
                    "snapshots": str(snapshot.snapshot_dir()),
                    "turn_floor": snapshot.trusted_floor(),
                    "mcp": "python3 -m polytopia_api.mcp",
                })
                return
            if path == "/hud":
                _json(self, 200, driver.hud())
                return
            if path == "/observe":
                _json(self, 200, do_observe())
                return
            if path == "/pixel":
                space = (q.get("space") or ["screen"])[0]
                x = int((q.get("x") or ["0"])[0])
                y = int((q.get("y") or ["0"])[0])
                driver.sync_layout()
                if space == "design":
                    x, y = coords.xy(x, y)
                r, g, b = driver.pixel(x, y)
                _json(self, 200, {"x": x, "y": y, "space": space, "rgb": [r, g, b]})
                return
            if path == "/diff":
                _json(self, 200, snapshot.last_diff() or {"ok": False, "reason": "no diff yet"})
                return
            if path == "/screenshot":
                shot = driver.screenshot()
                data = shot.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
                return
            _json(self, 404, {"error": "not found", "path": path})
        except driver.PolytopiaError as e:
            _json(self, 409, {"error": str(e)})
        except Exception as e:
            _json(self, 500, {"error": str(e)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = canonicalize_path(parsed.path)
        try:
            body = _read_json(self)
            space = str(body.get("space") or "screen")
            if path == "/click":
                if body.get("name"):
                    repeats = int(body.get("repeats") or 1)
                    _json(self, 200, driver.click_named(str(body["name"]), repeats=repeats))
                    return
                x = int(body["x"])
                y = int(body["y"])
                repeats = int(body.get("repeats") or 1)
                _json(self, 200, driver.click(x, y, repeats=repeats, space=space))
                return
            if path in {"/tech", "/tech-tree"}:
                _json(self, 200, driver.click_named("TECH_TREE"))
                return
            if path == "/settings":
                _json(self, 200, driver.click_named("SETTINGS"))
                return
            if path == "/confirm":
                _json(self, 200, driver.confirm())
                return
            if path == "/end-turn":
                turn = body.get("turn")
                _json(self, 200, commands.end_turn(turn=int(turn) if turn is not None else None))
                return
            if path == "/select-unit":
                x, y = _xy(body, "x", "y")
                _json(self, 200, commands.select_unit(
                    x=x, y=y, id=body.get("id"),
                    city_id=body.get("city_id"), space=space,
                ))
                return
            if path == "/move-to":
                x, y = _xy(body, "x", "y")
                fx, fy = _xy(body, "from_x", "from_y")
                _json(self, 200, commands.move_to(
                    x=x, y=y, from_x=fx, from_y=fy,
                    from_id=body.get("from_id") or body.get("unit_id"),
                    city_id=body.get("city_id"),
                    to_id=body.get("to_id"),
                    space=space,
                ))
                return
            if path == "/attack":
                x, y = _xy(body, "x", "y")
                fx, fy = _xy(body, "from_x", "from_y")
                _json(self, 200, commands.attack(
                    from_id=body.get("from_id") or body.get("unit_id"),
                    to_id=body.get("to_id"),
                    from_x=fx, from_y=fy,
                    x=x, y=y,
                    city_id=body.get("city_id"),
                    space=space,
                ))
                return
            if path == "/capture":
                _json(self, 200, commands.capture(
                    city_id=body.get("city_id"), space=space,
                ))
                return
            if path == "/recruit":
                x, y = _xy(body, "x", "y")
                _json(self, 200, commands.recruit(
                    x=x, y=y, id=body.get("id"),
                    city_id=body.get("city_id"),
                    unit=body.get("unit"), space=space,
                ))
                return
            if path == "/snapshot":
                obs = do_observe()
                turn = body.get("turn")
                _json(self, 200, snapshot.save(
                    obs,
                    reason=str(body.get("reason") or "manual"),
                    turn=int(turn) if turn is not None else None,
                ))
                return
            if path == "/calibrate":
                name = str(body.get("name") or "")
                if not name:
                    raise KeyError("name")
                x = int(body["x"])
                y = int(body["y"])
                driver.sync_layout()
                rec = coords.set_empirical(name, x, y)
                saved = coords.save_empirical()
                rec["saved"] = str(saved)
                rec["layout"] = coords.layout_info()
                _json(self, 200, rec)
                return
            if path == "/step":
                _json(self, 200, do_step())
                return
            if path == "/plan":
                obs = do_observe()
                _json(self, 200, {"plan": do_plan(obs), "observe": obs})
                return
            if path == "/play":
                n = int(body.get("n") or 1)
                _json(self, 200, play_n(n))
                return
            if path == "/back":
                _json(self, 200, driver.back())
                return
            if path == "/key":
                _json(self, 200, driver.press(str(body.get("key") or "")))
                return
            _json(self, 404, {"error": "not found", "path": path})
        except KeyError as e:
            _json(self, 400, {"error": f"missing field {e}"})
        except driver.PolytopiaError as e:
            _json(self, 409, {"error": str(e)})
        except Exception as e:
            _json(self, 500, {"error": str(e)})


def main() -> None:
    coords.load_empirical()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"polytopia api http://{HOST}:{PORT}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
