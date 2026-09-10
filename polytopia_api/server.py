"""Local HTTP API wrapping the Polytopia Unity window.

  python3 -m polytopia_api.server

  GET  /health
  GET  /hud
  GET  /observe
  GET  /screenshot
  GET  /pixel?x=1117&y=681
  POST /click     {"x":1111,"y":1130,"repeats":1}
  POST /confirm
  POST /end-turn
  POST /back
  POST /step          # one playbook action
  POST /play {"n":5}  # up to n steps, stops at End Turn
  POST /key       {"key":"space"}   # Escape is refused
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Allow `python3 server.py` from this folder.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "polytopia_api"

from . import coords
from . import driver
from .observe import observe as do_observe
from .play import plan as do_plan
from .play import play_n
from .play import step as do_step

HOST = "127.0.0.1"
PORT = 8765


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
        path = parsed.path.rstrip("/") or "/"
        q = parse_qs(parsed.query)
        try:
            if path == "/health":
                info = driver.find_window()
                _json(self, 200, {"ok": True, "game": info, "coords": {
                    "do_it": coords.DO_IT,
                    "train": coords.TRAIN,
                    "end_turn": coords.END_TURN,
                    "back": coords.BACK,
                }})
                return
            if path == "/hud":
                _json(self, 200, driver.hud())
                return
            if path == "/observe":
                _json(self, 200, do_observe())
                return
            if path == "/pixel":
                x = int((q.get("x") or ["0"])[0])
                y = int((q.get("y") or ["0"])[0])
                r, g, b = driver.pixel(x, y)
                _json(self, 200, {"x": x, "y": y, "rgb": [r, g, b]})
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
        path = parsed.path.rstrip("/") or "/"
        try:
            body = _read_json(self)
            if path == "/click":
                x = int(body["x"])
                y = int(body["y"])
                repeats = int(body.get("repeats") or 1)
                _json(self, 200, driver.click(x, y, repeats=repeats))
                return
            if path == "/confirm":
                _json(self, 200, driver.confirm())
                return
            if path == "/end-turn":
                _json(self, 200, driver.end_turn())
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
