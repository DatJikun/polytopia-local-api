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
3. `GET /observe` — `units` / `cities_own` / `cities_enemy` ze **stabilnym `id` i `city_id` = `c{gx}_{gy}`** (hash kafelka, nie indeks listy) i `cities_enemy[].tile: [gx,gy]`. **`city_id` i `unit_id` zostają na całą turę** — attack/move-to/capture resolvują `c22_17` / `u2` nawet gdy kolejny observe zgubi frost plate albo pasek HP (indeks sticky + `seen:false`). **Unit na górze od razu na południe od Disrof** (`tile_xy≈[600,144]`) jest w `units` na pierwszym observe (sticky `id` + `tile`, `near_city_id`) — pasek HP nad śniegiem nie jest wodą; skóra Bardur na hexie południowym też. `cities_own` to Bardur (`owner:own` / `tribe:bardur`); gold star na prawej stronie frost plate **nie** robi z nich Vengir. Enemy Vengir = **magenta roof albo gold lamps na budynku albo złote okna na frost plate** (`evidence`, `confidence` ≥0.78) — frost-plate gold zostaje enemy, **nie** wymyśla `cities_own`. Gołe frost fragmenty i ghost `Arch` odpadają (max 2 unnamed vengir) i **nie kumulują się** między observe (2→4→7). Jasne oumaji-ghosty bez purple/gold-on-dark odpadają. `villages` zwykle `[]`; `fog_edge`; HUD `turn`/`stars`/`score` z OCR + cropów cyfr. `hud.turn_trusted` jest **false** gdy OCR tury jest poniżej last labeled/clocked (`turn_floor` z `latest.json`, najwyższego `T{n}.json`, albo `POLYTOPIA_TURN=`); `turn` wtedy `null`, `turn_ocr` zostaje (np. 2 przy T26). `stale` / `missing` gdy cache. `turn_diff.alerts`.
4. W-path (Disrof): `POST /attack {"from_id":"uX","city_id":"cY"}` — czerwony mark przy **kafelku miasta** (nie pasek HP giganta), HP z paska unita na kafelku (nie `n`/`w` plate), `hp_dropped` albo jasne `no_actions` / `no_mark` **szybko** (nie timeout 7.5s). `POST /move-to {"from_id":"uZ","city_id":"cY"}` — klika **stopę hexu** (nie centroid dachu/`tile_xy` i nie sąsiad). Pierścień blue pod budynkiem jest ON tile. Select nie klika sticky `seen:false` (Settings/dock) — bierze żywy pasek HP (góra na południe od miasta). Brak marka → `no_mark_on_city_tile` / `out_of_range` (bez kliku w dach). Klik w miasto (TRAIN/Settings) → BACK, re-select żywego unita i klik w blue. `ok` true **tylko** przy `stood_on_city`. `POST /capture {"city_id":"cY"}` tylko gdy unit ON (nie adjacent) + blob w UNIT_CROP; inaczej `not_standing_on_city` bez kliku; `captured` gdy enemy straci `city_id`. `POST /recruit {"city_id":"c0"}` — nameplate → TRAIN blob w panelu (nie tylko modal na mapie).
5. `POST /end-turn` — snapshot. **Stale HUD nie nazywa pliku**. Podaj `{"turn":25}` albo zegar `last+1`.
6. Raw `POST /click` w strefie docka (prawy dół / End Turn) jest **zabroniony** — computerUse nie może kończyć tury. `layout.dock_zone` w `/health`.

```text
POST /back
GET  /observe  → Disrof in cities_enemy (vengir, high conf / evidence, city_id=id) AND Bardur in cities_own
POST /attack   {"from_id":"uX","city_id":"cY"} → hp_dropped true (albo jasny no_actions / no_mark, <8s)
POST /move-to  {"from_id":"uZ","city_id":"cY"} → stood_on_city true when the unit can walk onto the tile this turn (ok false if still adjacent)
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

- `units[]` — HP-bar, `id` `u0`… **sticky na turę** (lookup `u2` działa po ataku nawet gdy frame zgubi pasek)
- `cities` / `cities_own` / `cities_enemy` — Moonrise **frosted** nameplate (szary ~180, nie 215-white) + kolor plemienia. `id` **i** `city_id` = `c{gx}_{gy}`, pole `tile`. **Sticky na turę** (lookup po `city_id` działa po ataku nawet gdy frame zgubi miasto). `cities_own` = Bardur mimo złotej gwiazdy na prawej stronie plate. Vengir = fioletowy dach **albo złote lampy na budynku albo złote okna na frost plate** (`evidence: magenta_roof|gold_lamps`, wyższy `confidence`); frost-plate gold zostaje enemy (nie wymyśla own). Gołe frost FPs i OCR-ghost `Arch` odpadają i nie rosną w sesji. Jasny oumaji-looking na gorącej białej płytce odpada. `name` opcjonalne (OCR płytek wyłączone — `/attack` <8s). Lookup po `city_id`.
- `POST /move-to` — niebieskie marki po select `from_id`; cel = **stopa hexu** (~`plate_y` − 0.45 pitch), nie dach i nie sąsiedni niebieski. Klika znaleziony blue / lokalny pierścień pikseli; **nie** syntetyzuje kliku w `tile_xy` bez blue. `stood_on_city` po re-observe; `ok` true tylko wtedy. Negatyw: `no_mark_on_city_tile` / `out_of_range` / `not_stood_on_city` + `path_reason` + `suggested_tile_xy`. `ok` zawsze bool (nigdy `null`). `{x,y}` na kafelku miasta idzie tą samą ścieżką co `city_id`.
- `POST /attack` — po select tylko lekki observe (panel + marki, bez HUD/OCR miast). Cel kliku = kafelek miasta (czerwony hex), nie pasek HP. Brak czerwonego → szybki `no_mark` / `no_attack_marks`, nie timeout 7.5s.
- `POST /capture` — gate ON tile (~0.8 hex, nie adjacent ~1.0) **oraz** `unit.capture` / blob w UNIT_CROP. Bez tego `not_standing_on_city`, zero klików. `captured` gdy `cities_enemy` straci ten `city_id` albo owner→own. Hałaśliwe mapowe `do_it_blobs` **nie** klikają.
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
