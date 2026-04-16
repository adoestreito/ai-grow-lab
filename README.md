# AI Grow Lab

A small **home hydroponics lab stack**: an **ESP32-CAM** serves a live MJPEG stream and still captures, while a **Python web app** (FastAPI) saves snapshots on a schedule, proxies the stream for remote access (e.g. Tailscale), and runs **vision + chat** through **Ollama** or any **OpenAI-compatible** API (e.g. LM Studio). The UI is a cozy Stardew-inspired “farm shed” with a photo gallery and timelapse player.


---

## What’s in the repo

| Part | Role |
|------|------|
| `ESP32-BlindingLights-Camera/` | Arduino firmware (Espressif camera web server pattern): `/stream` on `port+1`, `/capture` and `/status` on the main port. |
| `diary-cam-server/` | FastAPI app: live stream proxy, snapshots, scheduler, journal (Markdown reports), gardener **chat** (with latest photo), gallery API. |
| `Diary/Photos/` & `Diary/Journal/` | Default folders for images and AI reports (contents are **gitignored**). |
| `scripts/` | Optional macOS `launchd` helper to pull a frame from the cam on an interval. |

---

## Quick start (web app)

```bash
cd diary-cam-server
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8765
```

Open **http://127.0.0.1:8765** (or your machine’s LAN / Tailscale IP).

- Configure **camera base URL** (e.g. `http://192.168.x.x` — stills on port 80, stream is proxied from port **81** automatically).
- Enable **“Proxy live stream through this Mac”** if you reach the UI over Tailscale but not your LAN.
- Set **Ollama** or **LM Studio** under *Garden AI report* for automatic reports and chat.

Config is stored in `~/Diary/.diary-cam-config.json` by default (overridable with `DIARY_CAM_CONFIG`). That path is outside the repo and is not committed.

---

## ESP32 firmware

1. Open `ESP32-BlindingLights-Camera/` in the Arduino IDE (board package + PSRAM-capable ESP32-CAM module as usual).
2. **Wi-Fi:** copy the example secrets file and edit **only on your machine**:
   ```bash
   cd ESP32-BlindingLights-Camera
   cp secrets.h.example secrets.h
   # Edit secrets.h with your SSID and password
   ```
3. Select the correct **camera model** in `board_config.h`.
4. Flash the board. In Serial Monitor, note the **IP address** — use `http://<ip>` as the camera base URL in the web app.

`secrets.h` is listed in `.gitignore` so credentials are never pushed.

---

## Features (high level)

- **Live stream** + manual **snapshot** into `Diary/Photos/`.
- **Scheduled** captures with optional **clock-aligned** start time.
- **AI garden reports** after each photo (Markdown in `Diary/Journal/`), with an optional **annotated pod map** image: place `grow_pod_reference.png` next to `garden_analysis.py` for richer prompts.
- **Chat with the gardener** — each reply includes your **latest saved** snapshot (vision models only).
- **Gallery & timelapse** at `/gallery.html` (slideshow oldest → newest).

---

## Privacy & what we don’t commit

- **Wi-Fi:** use `secrets.h` (from `secrets.h.example`); never paste real passwords into the `.ino`.
- **`Diary/Photos/` and `Diary/Journal/`:** actual grow photos and journals are **ignored**; only `.gitkeep` ships so the folders exist after clone.
- **Virtualenv** `diary-cam-server/.venv/` is ignored.
- **Local plist** paths: the example LaunchAgent uses a placeholder `/path/to/ai-grow-lab/…`.

---

## Optional: macOS scheduled fetch

See `scripts/com.blindinglights.diary-photo.plist.example` and `scripts/fetch-diary-photo.sh`. Adjust `ESP32_CAM_URL` and the script path, then install per the comments in the plist.

---

## Requirements

- Python **3.9+** (tested with 3.9+)
- **httpx**, **fastapi**, **uvicorn**, **markdown**
- A **vision-capable** local model if you use reports or chat with images

---

## License

Use and modify for your own grow setup. If you fork or publish derivatives, keep Wi-Fi and personal diary data out of git.
