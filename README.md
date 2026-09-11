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
3. `GET /observe` — `units` / `cities_own` / `cities_enemy` ze **stabilnym `city_id` = `c{gx}_{gy}`** (hash kafelka, nie indeks listy) i `cities_enemy[].tile: [gx,gy]`. Enemy: `tribe==vengir` albo ciemny kamień + nameplate; jasne oumaji-ghosty bez purple/gold-on-dark odpadają. `villages` zwykle `[]`; `fog_edge`; HUD `turn`/`stars`/`score` z OCR + cropów cyfr. `hud.turn_trusted` jest **false** gdy OCR tury jest poniżej last labeled/clocked (`turn_floor` z `latest.json`, najwyższego `T{n}.json`, albo `POLYTOPIA_TURN=`); `turn` wtedy `null`, `turn_ocr` zostaje (np. 2 przy T26). `stale` / `missing` gdy cache. `turn_diff.alerts`.
4. W-path (Disrof): `POST /attack {"from_id":"uX","city_id":"cY"}` — czerwony mark przy **garrison unit xy**, HP z paska unita na kafelku (nie `n`/`w` plate), `hp_dropped` albo jasne `no_actions` (<8s). `POST /move-to {"from_id":"uZ","city_id":"cY"}` — niebieski mark najbliższy **środka tile** (nie plate); `stood_on_city`; bez marka w 0.6 hex → `no_mark_on_city_tile`. `POST /capture {"city_id":"cY"}` tylko gdy unit ON (<0.4 hex) + blob w UNIT_CROP; inaczej `not_standing_on_city` bez kliku; `captured` gdy enemy straci `city_id`. `POST /recruit {"city_id":"c0"}` — nameplate → TRAIN blob.
5. `POST /end-turn` — snapshot. **Stale HUD nie nazywa pliku**. Podaj `{"turn":25}` albo zegar `last+1`.
6. Raw `POST /click` w strefie docka (prawy dół / End Turn) jest **zabroniony** — computerUse nie może kończyć tury. `layout.dock_zone` w `/health`.

```text
POST /back
GET  /observe  → city_id vengir + tile [gx,gy]
POST /attack   {"from_id":"uX","city_id":"cY"} → hp_dropped true (albo jasny no_actions)
POST /move-to  {"from_id":"uZ","city_id":"cY"} → stood_on_city true
POST /capture  {"city_id":"cY"} → captured true
```

## Diff tura→tura (kontrola)

Na End Turn (albo `POST /snapshot`):

- JSON + PNG: `POLYTOPIA_SNAPSHOTS=` (absolute), inaczej katalog z `T*.json` (cwd / repo / `~/.config/polytopia-local-api/turn_snapshot`), domyślnie **repo** `turn_snapshot/` — nie cichy CWD. `GET /health` → `snapshots` + `turn_floor`.
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
| `local_move_to` | `POST /move-to` `{"from_id","city_id"}` — mark na tile, `stood_on_city` |
| `local_attack` | `POST /attack` `{"from_id","city_id"}` — garrison HP + `hp_dropped` / `no_actions` |
| `local_capture` | `POST /capture` `{"city_id"}` — tylko ON + panel blob, `captured` |
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
- `cities` / `cities_own` / `cities_enemy` — Moonrise **frosted** nameplate (szary ~180, nie 215-white) + kolor plemienia. `id` = `c{gx}_{gy}`, pole `tile`. Vengir = fioletowy dach; **złote okna / gwiazdka poziomu ≠ Oumaji**. Jasny oumaji-looking bez purple/gold-on-dark odpada. `name` z OCR środka płytki (nie ikona/gwiazda). Lookup po `city_id` albo nazwie. Enemy zostaje bez nazwy.
- `POST /move-to` — niebieskie marki po select `from_id`; cel = środek budynku (~`plate_y` minus ~0.75 hex). `stood_on_city` po re-observe.
- `POST /capture` — gate ON tile (<0.4 hex) **oraz** `unit.capture` / blob w UNIT_CROP. Bez tego `not_standing_on_city`, zero klików. `captured` gdy `cities_enemy` straci ten `city_id` albo owner→own. Hałaśliwe mapowe `do_it_blobs` **nie** klikają.
- `attack_marks[]` — czerwone hexy ataku
- `selected_unit` — typ z panelu, `range`, `naval`, `land_attack`
- `villages[]` — puste, chyba że `POLYTOPIA_VILLAGES=1`
- `fog_edge[]`
- `ready.capture` / `ready.train` / `ready.move` / `ready.harvest`
- HUD: crop + whitelist cyfr, split wokół złotej gwiazdy. `turn_trusted` + `turn_floor` z last snapshot; OCR 2 przy T26 → `turn: null`, `turn_ocr: 2`. `hud.stale` / `hud.missing` / `hud.raw` / `hud.digits`

Detekcja mapy jest heurystyką z pikseli, nie native state.

## Testy

```bash
PYTHONPATH=. python3 polytopia_api/test_hud.py
```
