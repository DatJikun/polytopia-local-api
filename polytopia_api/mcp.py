"""MCP stdio connector: local_* tools → http://127.0.0.1:8765

  python3 -m polytopia_api.mcp

Cursor: copy mcp.example.json into ~/.cursor/mcp.json (or project .cursor/mcp.json).
think_turn should call these instead of curl.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    __package__ = "polytopia_api"

API = os.environ.get("POLYTOPIA_API", "http://127.0.0.1:8765").rstrip("/")

TOOLS: list[dict[str, Any]] = [
    {
        "name": "local_health",
        "description": "Polytopia local driver health + empirical layout (dock coords, frame, sources).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "local_observe",
        "description": "Screenshot observe: HUD, stable units/cities (Disrof stays cities_enemy while a friendly unit is on the port/mountain next to it; those units keep sticky ids), fog_edge. Villages off unless POLYTOPIA_VILLAGES=1. Includes turn_diff vs last End Turn snapshot.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "local_hud",
        "description": "OCR HUD only (turn, stars, score, income).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "local_select_unit",
        "description": "Click a unit/city. Pass screen x,y from observe, or id/city_id (u0, c0).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "id": {"type": "string"},
                "city_id": {"type": "string"},
                "space": {"type": "string", "description": "screen (default) or design"},
            },
        },
    },
    {
        "name": "local_move_to",
        "description": "Walk onto a city tile (city_id or x,y). If the city is garrisoned, returns city_occupied with next=/attack (do not walk). Else clicks the building foot. stood_on_city true only when the unit is ON the hex.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "from_x": {"type": "integer"},
                "from_y": {"type": "integer"},
                "from_id": {"type": "string"},
                "city_id": {"type": "string"},
                "to_id": {"type": "string"},
                "space": {"type": "string"},
            },
        },
    },
    {
        "name": "local_attack",
        "description": "Attack the city garrison: select from_id, click a red hex on city_id. hp_dropped until garrison_dead, then next=/move-to. Rafts cannot hit land — response suggests a land attacker_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_id": {"type": "string"},
                "to_id": {"type": "string"},
                "city_id": {"type": "string"},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "from_x": {"type": "integer"},
                "from_y": {"type": "integer"},
                "space": {"type": "string"},
            },
        },
    },
    {
        "name": "local_capture",
        "description": "Press Capture blob when a unit stands ON the city (city_id). Building first, not scaled DO IT.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city_id": {"type": "string"},
                "space": {"type": "string"},
            },
        },
    },
    {
        "name": "local_recruit",
        "description": "Select city_id (nameplate) then click a detected TRAIN blob. Radial unit portraits still need a map click.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "id": {"type": "string"},
                "city_id": {"type": "string"},
                "unit": {"type": "string"},
                "space": {"type": "string"},
            },
        },
    },
    {
        "name": "local_end_turn",
        "description": "End Turn + snapshot. Pass turn=N if HUD is stale — will not write T4.json from cached OCR.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "turn": {"type": "integer", "description": "Trusted turn label. Required when HUD stale and no clock yet."},
            },
        },
    },
    {
        "name": "local_click",
        "description": "Click by name (TECH_TREE) or x,y. Raw x,y in the dock zone is denied — never End Turn via computerUse.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "TECH_TREE / SETTINGS / GAME_STATS / END_TURN"},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "repeats": {"type": "integer"},
                "space": {"type": "string"},
            },
        },
    },
    {
        "name": "local_tech",
        "description": "Click TECH_TREE by name (not an offset next to End Turn).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "local_diff",
        "description": "Last structured turn/frame diff (turn/stars, units moved, cities own/enemy, fog_edge, alerts).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "local_snapshot",
        "description": "Save observe JSON+PNG under turn_snapshot/. Pass turn=N — stale HUD is not a filename.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "turn": {"type": "integer"},
            },
        },
    },
    {
        "name": "local_back",
        "description": "Close Settings/tech overlay. Never Escape.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "local_confirm",
        "description": "Click DO IT / TRAIN if the brand-blue confirm is up.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "local_calibrate",
        "description": "Save an empirical screen point (e.g. name=END_TURN, x=765, y=746) as source of truth.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["name", "x", "y"],
        },
    },
    {
        "name": "local_step",
        "description": "Playbook fallback: one auto action. Brain stays in think_turn.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

_GET = {
    "local_health": "/health",
    "local_observe": "/observe",
    "local_hud": "/hud",
    "local_diff": "/diff",
}
_POST = {
    "local_select_unit": "/select-unit",
    "local_move_to": "/move-to",
    "local_attack": "/attack",
    "local_capture": "/capture",
    "local_recruit": "/recruit",
    "local_end_turn": "/end-turn",
    "local_click": "/click",
    "local_tech": "/tech",
    "local_snapshot": "/snapshot",
    "local_back": "/back",
    "local_confirm": "/confirm",
    "local_calibrate": "/calibrate",
    "local_step": "/step",
}


def _http(method: str, path: str, body: dict | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(API + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return {"ok": False, "reason": "empty_response"}
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return {"ok": False, "reason": "bad_response", "raw": parsed}
            if parsed.get("ok") is None:
                parsed["ok"] = False if parsed.get("error") else True
            return parsed
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": raw or str(e)}
        parsed.setdefault("http", e.code)
        if parsed.get("ok") is None:
            parsed["ok"] = False
        return parsed
    except urllib.error.URLError as e:
        return {
            "ok": False,
            "error": f"API {API} unreachable ({e}). Start: python3 -m polytopia_api.server",
        }


def call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {}
    if name in _GET:
        return _http("GET", _GET[name])
    if name in _POST:
        return _http("POST", _POST[name], args)
    return {"error": f"unknown tool {name}"}


def _send(msg: dict[str, Any]) -> None:
    payload = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _read() -> dict[str, Any] | None:
    header = b""
    stdin = sys.stdin.buffer
    while b"\r\n\r\n" not in header and b"\n\n" not in header:
        chunk = stdin.read(1)
        if not chunk:
            return None
        header += chunk
        # NDJSON fallback (no headers)
        if header.startswith(b"{") and header.endswith(b"\n"):
            return json.loads(header.decode("utf-8"))
    if b"\r\n\r\n" in header:
        pre, rest = header.split(b"\r\n\r\n", 1)
    else:
        pre, rest = header.split(b"\n\n", 1)
    length = 0
    for line in pre.split(b"\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    body = rest
    need = length - len(body)
    while need > 0:
        chunk = stdin.read(need)
        if not chunk:
            break
        body += chunk
        need = length - len(body)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _result(rid: Any, payload: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": bool(payload.get("error") and payload.get("ok") is False),
        },
    }


def handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    method = msg.get("method")
    rid = msg.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "polytopia-local", "version": "0.3.0"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name = str(params.get("name") or "")
        payload = call_tool(name, params.get("arguments") or {})
        return _result(rid, payload)
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if rid is None:
        return None
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"unknown method {method}"},
    }


def main() -> None:
    while True:
        msg = _read()
        if msg is None:
            break
        reply = handle(msg)
        if reply is not None:
            _send(reply)


if __name__ == "__main__":
    main()
