# Polytopia local API

Nieoficjalna nakładka na okno **The Battle of Polytopia** (Unity): screenshot + OCR HUD + detekcja pikseli + `xdotool`. To **nie** jest API Hello Games i nie gada z ich serwerami.

## Wymagania

- Linux + X11 (`DISPLAY`, np. `:1`)
- `python3`, `numpy`, `Pillow`
- `ffmpeg`, `tesseract`, `xdotool`
- Gra może być **dowolny fullscreen** — koordy są w przestrzeni 1920×1200 i skalują się do okna. **1280×800** to ten sam 16:10 (mnożnik 2/3). Nadpis: `POLYTOPIA_SCREEN=1280x800`.

```bash
python3 -m polytopia_api.server   # http://127.0.0.1:8765
```

`GET /health` zwraca `layout.frame`, `layout.scale` i przeliczone przyciski (`end_turn`, `do_it`, …).

`POST /click` domyślnie bierze koordy **design** (1920×1200), tak jak playbook:

```bash
curl -s -X POST http://127.0.0.1:8765/click \
  -H 'Content-Type: application/json' \
  -d '{"x":1111,"y":1130}'
```

Pola z `/observe` (`move_marks`, `fruit`, bloby overlay) są już w pikselach okna (`hits_space: "screen"`). `think_turn` klika je tak:

```bash
curl -s -X POST http://127.0.0.1:8765/click \
  -H 'Content-Type: application/json' \
  -d '{"x":828,"y":400,"space":"screen"}'
```

Bez `"space":"screen"` serwer potraktuje liczby jako 1920×1200 i przeskaluje jeszcze raz.

## Endpointy

| Metoda | Ścieżka | Co robi |
|---|---|---|
| GET | `/health` | czy okno gry żyje |
| GET | `/hud` | tura, ★, dochód, score |
| GET | `/observe` | HUD + panel jednostki + pola ruchu + owoce + overlay |
| GET | `/screenshot` | PNG |
| POST | `/plan` | ta sama obserwacja, **bez** kliknięcia |
| POST | `/step` | jedna akcja playbooka, potem znowu HUD |
| POST | `/play` | `{"n":5}` — kilka kroków, stop na End Turn |
| POST | `/click` | `{"x":1111,"y":1130}` |
| POST | `/confirm` | DO IT / TRAIN |
| POST | `/end-turn` | podwójny klik |
| POST | `/back` | wyjście z Settings/tech (nie Escape) |

```bash
curl -s http://127.0.0.1:8765/observe
curl -s -X POST http://127.0.0.1:8765/step
```

## Playbook `POST /step`

1. Zamknij Settings / odrzuć Clear Forest  
2. Potwierdź harvest/train, jeśli modal już wisi  
3. Ruch tylko po wykrytych **niebieskich polach** (nie woda)  
4. Żniwa owoców przy ≥2★  
5. Odklik jednostki bez akcji  
6. End Turn; **blokada przy ≥20★**

Escape jest zablokowane (otwiera Settings).

## Testy

```bash
PYTHONPATH=. python3 polytopia_api/test_hud.py
```
