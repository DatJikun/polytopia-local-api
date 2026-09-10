# Polytopia local API

Nieoficjalna nakładka na okno **The Battle of Polytopia** (Unity): screenshot + OCR HUD + detekcja pikseli + `xdotool`. To **nie** jest API Hello Games i nie gada z ich serwerami.

## Wymagania

- Linux + X11 (`DISPLAY`, np. `:1`)
- `python3`, `numpy`, `Pillow`
- `ffmpeg`, `tesseract`, `xdotool`
- Gra odpalona fullscreen 1920×1200 (koordy UI są pod ten layout)

```bash
python3 -m polytopia_api.server   # http://127.0.0.1:8765
```

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
