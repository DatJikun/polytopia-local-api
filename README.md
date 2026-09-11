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
3. `GET /observe` — `units` / `cities_own` / `cities_enemy` ze **stabilnym `id` i `city_id` = `c{gx}_{gy}`** (hash kafelka, nie indeks listy) i `cities_enemy[].tile: [gx,gy]`. **`city_id` i `unit_id` zostają na całą turę** — attack/move-to/capture resolvują `c22_17` / `u2` nawet gdy kolejny observe zgubi frost plate albo pasek HP (indeks sticky + `seen:false`). Druga komenda w tej samej turze **nie** wraca `need x,y or city_id/to_id` dla znanego `c14_24`. **Disrof zostaje w `cities_enemy`** gdy jest na ekranie — nawet z garnizonem i unitami Bardur na **porcie na wschód** albo górze na południe; nie znika i nie wpada do `cities_own`. **Pełna lista `cities_own` (T44/T45: 12 Bardur) nie czyści `cities_enemy`** — widoczne Vengir (Disrot / Rzgórst / Grugru na ciemnych kafelkach) zostają z sticky `city_id` + `tile_xy`; cap własnych miast nie zjada slotów enemy. **Sąsiad Bardur 1 hex od Vengir nie wpada do `cities_enemy`** — `cities_own` zostaje pełne (nie 2/3–4 przy wielu miastach na ekranie; sąsiednie Bardur 1 hex od siebie **nie zlewają się**) i **nie flipuje own→enemy** po `/recruit`. **Unit na górze od razu na południe od Disrof** (`tile_xy≈[600,144]`) **albo na porcie na wschód** jest w `units` na pierwszym observe (sticky `id` + `tile`, `near_city_id`) — pasek HP nad śniegiem/wodą nie jest wodą; skóra Bardur na hexie sąsiednim też. Nie tylko wojownicy na dole (~y=462). `cities_own` to Bardur (`owner:own` / `tribe:bardur`) **w liczbie z ekranu / Game Stats** (T46: 2 Bufla/Orkork, nie 4 z duplikatu Orkork `c15_24`+`c16_24` ~35px i mid-map phantom `c12_16`, nie HUD/fog `c18_1` @ y≈45, nie 5–7 ghostów `c15_15`/`c15_8`/`c21_23`) — frost-plate + gold star/lamps na **longhouse** **poniżej HUD**; ten sam kafelek / bliskie płyty (~35px) zlewają się, sąsiednie miasta ~80px nie; nienazwane/low-evidence/top-chrome/sparse-wood phantomów odpadają i **nie kumulują się** między observe. Gold star na prawej stronie frost plate **nie** robi z nich Vengir. Enemy Vengir = **magenta roof albo gold lamps na budynku albo złote okna na frost plate** (`evidence`, `confidence` ≥0.78) — frost-plate gold zostaje enemy, **nie** wymyśla `cities_own`. Gołe frost fragmenty, ghost `Arch` i nienazwany Imperius bez frost plate odpadają (max 2 unnamed frost-FP; **wszystkie** Disrof-class zostają) i **nie kumulują się** między observe (2→4→7). Jasne oumaji-ghosty bez purple/gold-on-dark odpadają. `villages` zwykle `[]`; `fog_edge`; HUD `turn`/`stars`/`score` z OCR + cropów cyfr. `hud.turn_trusted` jest **false** gdy OCR tury jest poniżej last labeled/clocked (`turn_floor` z `latest.json`, najwyższego `T{n}.json`, albo `POLYTOPIA_TURN=`); `turn` wtedy `null`, `turn_ocr` zostaje (np. 2 przy T26). `stale` / `missing` gdy cache. `turn_diff.alerts`.
4. W-path (Disrof): jeśli `/move-to` zwraca `city_occupied`, **najpierw atakuj garnizon** (`next_call` `/attack` z `attacker_id` lądowym — nie tratwa). `POST /attack {"from_id":"uX","city_id":"cY"}` — poczekaj na czerwony mark przy **stopie kafelka** (nie pasek HP; light select bywa za szybki), kliknij stopę, czekaj na `hp_dropped`; gdy HP stoi — re-select i drugi klik. Aż `garrison_dead` / `next:move-to`, albo jasne `no_actions` / `no_mark` / `naval_no_land` **szybko**. Potem `POST /move-to` — klika **stopę hexu** (nie dach). Gdy unit jest ~1 hex od miasta i clustered `move_marks` są puste, i tak klika stopę / ring (nie `no_move_marks`). Po kliku czeka aż unit **stanie ON** (re-select / wait / weryfikacja kafelka). `ok` true **tylko** przy `stood_on_city`. `POST /capture` gdy unit ON (ten sam próg co `/move-to`, w tym offset paska HP ~0.7 hex): jeśli panel nie pokazuje Capture, **najpierw select occupanta** (stopa sprite, nie dach / TRAIN). Blob w UNIT_CROP → klik; `entering city` / `capture_soon` → `not_ready_until_next_turn`. Garnizon żyje → `city_occupied`. Garnizon **own** na city_id oznaczonym enemy → `/attack` `own_garrison` (nie bić Bardur). `POST /recruit {"city_id":"c0"}` — **dach/nameplate najpierw** (nie stopa unita / Disband; kafelek to fallback gdy hex pusty) → TRAIN w UNIT_CROP (także brand-blue), confirm w panelu, nigdy Tech; **marks observe**, budget <8s jak `/attack`. Wine-shadow na drewnie Bardur **nie** robi z miast `cities_enemy`.
5. `POST /end-turn` — snapshot. **Stale HUD nie nazywa pliku**. Podaj `{"turn":25}` albo zegar `last+1`.
6. Raw `POST /click` w strefie docka (prawy dół / End Turn) jest **zabroniony** — computerUse nie może kończyć tury. `layout.dock_zone` w `/health`.

```text
POST /back
GET  /observe  → Disrof in cities_enemy (vengir, high conf / evidence, city_id=id) AND Bardur in cities_own
POST /move-to  {"from_id","city_id"} → city_occupied + next=/attack while garrison lives (no walk click)
POST /attack   {"from_id":"land_unit","city_id":"cY"} → hp_dropped until garrison_dead (city-foot red, retry if HP stays; or clear no_mark / naval_no_land, <8s)
POST /move-to  {"from_id","city_id"} → stood_on_city true once the tile is empty
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
| `local_attack` | `POST /attack` `{"from_id","city_id"}` — city-foot red + `hp_dropped` / retry / `no_actions` |
| `local_capture` | `POST /capture` `{"city_id"}` — ON tile, otwiera panel occupanta gdy brak bloba, `captured` |
| `local_recruit` | `POST /recruit` `{"city_id":"c0"}` — dach/nameplate → TRAIN/confirm w UNIT_CROP (nie Disband/Tech); marks observe <8s; radial na mapie |
| `local_end_turn` | `POST /end-turn` `{"turn":25}` gdy HUD stale |
| `local_click` | `POST /click` `{"name":"TECH_TREE"}` albo `{"x","y"}` |
| `local_tech` | `POST /tech` |
| `local_diff` | `GET /diff` |
| `local_snapshot` | `POST /snapshot` |
| `local_back` / `local_confirm` / `local_calibrate` | jak nazwa |
| `local_step` | playbook fallback |

`GET /observe` (screen pixels, `hits_space: "screen"`):

- `units[]` — HP-bar, `id` `u0`… **sticky na turę** (lookup `u2` działa po ataku nawet gdy frame zgubi pasek)
- `cities` / `cities_own` / `cities_enemy` — Moonrise **frosted** nameplate (szary ~180, nie 215-white) + kolor plemienia. `id` **i** `city_id` = `c{gx}_{gy}`, pole `tile`. **Sticky na turę** (lookup po `city_id` działa po ataku nawet gdy frame zgubi miasto; druga komenda nie wraca `need x,y`). Disrof zostaje `cities_enemy` gdy widać miasto (garnizon / Bardur na porcie lub górze nie czyści listy). **12+ `cities_own` nie wycina Vengir** — Disrot / Rzgórst / Grugru na ciemnych kafelkach zostają z `tile_xy`; cap własnych miast nie zjada slotów enemy. Sąsiad Bardur 1 hex od Disrof **zostaje `cities_own`** (nie 2 przy pełnej mapie). `cities_own` = Bardur mimo złotej gwiazdy na prawej stronie plate. Vengir = fioletowy dach **albo złote lampy na budynku albo złote okna na frost plate** (`evidence: magenta_roof|gold_lamps`, wyższy `confidence`); frost-plate gold zostaje enemy (nie wymyśla own). Gołe frost FPs i OCR-ghost `Arch` odpadają i nie rosną w sesji (Disrof-class bez nazwy **zostają wszystkie**; dwa prawdziwe Vengir nie zlewają się przy 80 px). Jasny oumaji-looking na gorącej białej płytce odpada. `name` opcjonalne (OCR płytek wyłączone — `/attack` <8s). Lookup po `city_id`.
- `POST /move-to` — niebieskie marki po select `from_id`; cel = **stopa hexu** (~`plate_y` − 0.45 pitch). Garnizon enemy → `city_occupied` + `next:/attack` (bez kliku w blue). Gdy unit jest ~1 hex od `city_id` i clustered marki są puste, klika stopę (nie `no_move_marks`). Po kliku **czeka na animację chodzenia**, re-select jeśli unit zostaje sąsiadem, weryfikuje kafelek. `stood_on_city` po re-observe; `ok` true tylko wtedy. Negatyw: `no_mark_on_city_tile` / `out_of_range` / `not_stood_on_city`.
- `POST /attack` — po light select **czeka na czerwony hex** (0.18s często robi screenshot zanim hex się zapali). Klika **stopę kafelka**, nie pasek HP / dach. Jeśli HP nie spadło, re-select i drugi klik (`strike_retried`). `hp_dropped` aż `garrison_dead` / `next:move-to`. Raft → `naval_no_land` + `attacker_id` lądowego sąsiada. Brak czerwonego → szybki `no_mark`.
- `POST /capture` — gate ON tile. Gdy unit już stoi (`already_on_city`) a panel nie ma Capture, klika **stopę occupanta** aż pojawi się blob / OCR (nie dach miasta). `capture_soon` → `not_ready_until_next_turn`. Garnizon enemy → `city_occupied`. OCR Capture bez bloba klika panel, nie mapowe `do_it`. Blob w UNIT_CROP wystarczy nawet gdy OCR zgubi „Capture”. `captured` gdy `cities_enemy` straci ten `city_id`.
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
