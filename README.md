# Polytopia local API

Nieoficjalna nakładka na okno **The Battle of Polytopia** (Unity): screenshot + OCR HUD + detekcja pikseli + `xdotool`. To **nie** jest API Hello Games.

Mózg zostaje w `think_turn`. Ten driver daje **stan mapy + komendy**. Diff tura→tura (`turn_snapshot/`) to **warstwa kontroli**, nie zamiennik API. `POST /step` jest tylko fallbackiem playbooka.

## Dock: empiria, nie 2/3

Koordy playbooka są z 1920×1200. Naiwne `× 2/3` na **1280×800 psuje End Turn**. Źródło prawdy to klik zmierzony na pulpicie:

| przycisk | 1920×1200 (design) | 1280×800 |
|---|---|---|
| End Turn | 1111, 1130 | **765, 746** (empiria) |
| Settings / Stats / Tech | rząd docku | **przybite do End Turn** z bezpieczną przerwą |

Tech **nie** jest offsetem ~19 px od End Turn (po skalowaniu ~13 px). To był skok **T18→T22**: klik w Tech lądował na End Turn. `POST /click {"name":"TECH_TREE"}` odmawia, gdy punkt jest `< 64` px od End Turn. `GET /health` → `layout.tech_end_dist` i `layout.tech_too_close`.

`find_window` bierze tylko proces, którego **argv0 kończy się na `Polytopia.x86_64`**, plus okno o nazwie/klasie dokładnie `Polytopia` — nie wrapper Steama i nie Chrome `polytopia.local`.

Ścieżki: `/select_unit` = `/select-unit` (to samo dla `end_turn` / `move_to`).

```bash
curl -s -X POST http://127.0.0.1:8765/calibrate \
  -H 'Content-Type: application/json' \
  -d '{"name":"TECH_TREE","x":677,"y":738}'
```

Zapis: `layout.json` w cwd albo `~/.config/polytopia-local-api/layout.json` (`POLYTOPIA_LAYOUT=`). `POLYTOPIA_TRIBE=bardur` (domyślnie) rozróżnia własne/wrogie miasta. Wioski są **wyłączone** (fałszywe huty na ziemi); włącz `POLYTOPIA_VILLAGES=1`.

## Start

```bash
python3 -m polytopia_api.server   # http://127.0.0.1:8765
python3 -m polytopia_api.mcp      # local_* dla Cursor MCP
```

`POST /click` i komendy mapy biorą **piksele okna** (`space=screen`). 1920×1200 tylko z `"space":"design"`.

## Kontrakt dla think_turn

1. `GET /health` — okno + `layout.screen.END_TURN [765,746]`; Tech ≥64 px od End Turn (`tech_too_close: false`).
2. `POST /click {"name":"TECH_TREE"}` (albo `POST /tech`) — nigdy offset obok End Turn.
3. `GET /observe` — `units` / `cities_own` / `cities_enemy` ze **stabilnymi id** i kolorem plemienia; `villages` zwykle `[]`; `fog_edge`; HUD `turn`/`stars`/`score` z OCR + cropów cyfr. `hud.turn_trusted` jest **false** gdy OCR tury jest poniżej last labeled/clocked (`turn_floor` z `latest.json`, najwyższego `T{n}.json`, albo `POLYTOPIA_TURN=`); `turn` wtedy `null`, `turn_ocr` zostaje (np. 2 przy T26). `stale` / `missing` gdy cache. `turn_diff.alerts`.
4. Mapa: `POST /recruit {"city_id":"c0"}` — klika **nameplate** miasta, potem wykryty blob TRAIN (nie 2/3 z 1920, to na 1280 jest środek mapy). Radial jednostek nadal na mapie.
5. `POST /end-turn` — snapshot. **Stale HUD nie nazywa pliku** (`T4.json` przy T24). Podaj `{"turn":25}` albo po pierwszym labelu zegar `last+1`. Bez tego: `untrusted-*.json` + `ok: false`.

## Diff tura→tura (kontrola)

Na End Turn (albo `POST /snapshot`):

- JSON + PNG w `turn_snapshot/` (`POLYTOPIA_SNAPSHOTS=`).
- Numer tury: ręczny `turn=N`, świeże OCR (`turn_trusted`), albo zegar po End Turn. Nigdy cache/stale.
- Diff: turn/stars, lista unitów (pozycje), miasta own/enemy, fog edge.
- Alert `turn_jump` gdy **zaufana** tura skacze o **>1**.
- Alert `stale_hud` gdy OCR nie zasługuje na `T{n}`.
- Alert `stars_income_without_turn` gdy ★ rosną o income przy `turn_delta==0`.

`GET /diff` / `local_diff` zwraca ostatni structured diff. PNG jest do weryfikacji ludzkiej; bot czyta JSON.

## MCP `local_*` (zamiast curl)

Skopiuj `mcp.example.json` do `~/.cursor/mcp.json` albo zostaw `.cursor/mcp.json` w klonie. Connector gada z `:8765`, więc **serwer HTTP musi działać**.

| tool | HTTP |
|---|---|
| `local_health` | `GET /health` |
| `local_observe` | `GET /observe` |
| `local_hud` | `GET /hud` |
| `local_select_unit` | `POST /select-unit` `{"id":"u0"}` / `{"city_id":"c0"}` / `{"x","y"}` |
| `local_move_to` | `POST /move-to` `{"city_id":"c1"}` albo `{"x","y"}`, opcjonalnie `from_id` |
| `local_capture` | `POST /capture` `{"city_id":"c1"}` |
| `local_recruit` | `POST /recruit` `{"city_id":"c0"}` — nameplate → TRAIN blob; radial na mapie |
| `local_end_turn` | `POST /end-turn` `{"turn":25}` gdy HUD stale |
| `local_click` | `POST /click` `{"name":"TECH_TREE"}` albo `{"x","y"}` |
| `local_tech` | `POST /tech` |
| `local_diff` | `GET /diff` |
| `local_snapshot` | `POST /snapshot` |
| `local_back` / `local_confirm` / `local_calibrate` | jak nazwa |
| `local_step` | playbook fallback |

`GET /observe` (screen pixels, `hits_space: "screen"`):

- `units[]` — HP-bar, `id` `u0`…, `tribe` / `owner`
- `cities` / `cities_own` / `cities_enemy` — nameplate + kolor plemienia (id trzymają się klatek, gdy blob ruszy <56 px)
- `villages[]` — puste, chyba że `POLYTOPIA_VILLAGES=1`
- `fog_edge[]`
- `ready.capture` / `ready.train` / `ready.move` / `ready.harvest`
- HUD: crop + whitelist cyfr, split wokół złotej gwiazdy. `turn_trusted` + `turn_floor` z last snapshot; OCR 2 przy T26 → `turn: null`, `turn_ocr: 2`. `hud.stale` / `hud.missing` / `hud.raw` / `hud.digits`

Detekcja mapy jest heurystyką z pikseli, nie native state.

## Testy

```bash
PYTHONPATH=. python3 polytopia_api/test_hud.py
```
