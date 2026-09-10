# Polytopia local API

Nieoficjalna nakładka na okno **The Battle of Polytopia** (Unity): screenshot + OCR HUD + detekcja pikseli + `xdotool`. To **nie** jest API Hello Games.

Mózg zostaje w `think_turn`. Ten driver daje **stan mapy + komendy** (szybciej niż computerUse). `POST /step` jest tylko fallbackiem playbooka.

## Dock: empiria, nie 2/3

Koordy playbooka są z 1920×1200. Naiwne `× 2/3` na **1280×800 psuje End Turn**. Źródło prawdy to klik zmierzony na Twoim pulpicie:

| przycisk | 1920×1200 (design) | 1280×800 |
|---|---|---|
| End Turn | 1111, 1130 | **765, 746** (empiria) |

`GET /health` → `layout.source.END_TURN: "empirical"` i `layout.screen.END_TURN: [765, 746]`. Reszta bez pomiaru leci ze skali (oznaczone `"scaled"`).

```bash
# dopisz kolejny punkt, gdy coś mija
curl -s -X POST http://127.0.0.1:8765/calibrate \
  -H 'Content-Type: application/json' \
  -d '{"name":"DO_IT","x":740,"y":450}'
```

Zapis: `layout.json` w cwd albo `~/.config/polytopia-local-api/layout.json` (`POLYTOPIA_LAYOUT=`). `POLYTOPIA_TRIBE=bardur` (domyślnie) rozróżnia własne/wrogie miasta.

## Start

```bash
python3 -m polytopia_api.server   # http://127.0.0.1:8765
python3 -m polytopia_api.mcp      # local_* dla Cursor MCP
```

`POST /click` i komendy mapy biorą **piksele okna** (`space=screen`). 1920×1200 tylko z `"space":"design"`.

## MCP `local_*` (zamiast curl)

Skopiuj `mcp.example.json` do `~/.cursor/mcp.json` albo zostaw `.cursor/mcp.json` w klonie. Connector gada z `:8765`, więc **serwer HTTP musi działać**.

| tool | HTTP |
|---|---|
| `local_health` | `GET /health` |
| `local_observe` | `GET /observe` |
| `local_hud` | `GET /hud` |
| `local_select_unit` | `POST /select-unit` `{"id":"u0"}` albo `{"x","y"}` |
| `local_move_to` | `POST /move-to` `{"x","y"}` opcjonalnie `from_id` / `from_x,from_y` |
| `local_capture` | `POST /capture` |
| `local_recruit` | `POST /recruit` `{"id":"c0"}` — radial jednostek nadal na mapie |
| `local_end_turn` | `POST /end-turn` |
| `local_click` / `local_back` / `local_confirm` / `local_calibrate` | jak nazwa |
| `local_step` | playbook fallback |

`GET /observe` (screen pixels, `hits_space: "screen"`):

- `units[]` — HP-bar, `id` `u0`…
- `cities` / `cities_own` / `cities_enemy` — nameplate + kolor plemienia
- `villages[]`
- `ready.capture` / `ready.train` / `ready.move` / `ready.harvest`
- HUD (`turn` / `stars` / `score`) — crop + whitelist cyfr; jak OCR skłamie, patrz `hud.raw` / `hud.digits`

Detekcja mapy jest heurystyką z pikseli, nie native state.

## Testy

```bash
PYTHONPATH=. python3 polytopia_api/test_hud.py
```
