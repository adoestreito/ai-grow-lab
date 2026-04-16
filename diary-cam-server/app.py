"""
Local web UI for ESP32-CAM: live stream proxy, manual snapshot, scheduled saves under this repo
(BlindingLights/Diary/Photos by default).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
import markdown
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from garden_analysis import build_report_markdown, run_gardener_chat, run_hydro_vision_report

logger = logging.getLogger("diary-cam")

_REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(
    __import__("os").environ.get(
        "DIARY_CAM_CONFIG",
        str(Path.home() / "Diary" / ".diary-cam-config.json"),
    )
)
DEFAULT_PHOTOS_DIR = _REPO_ROOT / "Diary" / "Photos"
DEFAULT_JOURNAL_DIR = _REPO_ROOT / "Diary" / "Journal"
_LEGACY_HOME_DIARY_PHOTOS = (Path.home() / "Diary" / "Photos").resolve()
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Same default boundary as ESP32 camera_web_server example (app_httpd.cpp)
DEFAULT_STREAM_TYPE = "multipart/x-mixed-replace; boundary=123456789000000000000987654321"


def _httpx_timeout(*, connect: float, read: float) -> httpx.Timeout:
    """httpx requires connect/read/write/pool all set (or a single default)."""
    return httpx.Timeout(connect=connect, read=read, write=connect, pool=connect)


def _camera_async_client(**kwargs: Any) -> httpx.AsyncClient:
    """
    Do not use system HTTP(S)_PROXY for the ESP32: proxies often break LAN IPs and
    long-lived MJPEG (502 from /api/stream) while short /status checks still look fine.
    """
    kwargs.pop("trust_env", None)
    return httpx.AsyncClient(trust_env=False, **kwargs)


def _seconds_until_next_clock_fire(
    now: datetime,
    interval_seconds: int,
    anchor_hour: int,
    anchor_minute: int,
) -> float:
    """
    Seconds until the next wall-clock slot when interval divides 86400 evenly.
    Slots: anchor time + k * interval (mod 24h), local time on this machine.
    """
    if interval_seconds <= 0 or 86400 % interval_seconds != 0:
        return float(max(interval_seconds, 1))
    n = 86400 // interval_seconds
    base = (anchor_hour * 3600 + anchor_minute * 60) % 86400
    slots = sorted({(base + k * interval_seconds) % 86400 for k in range(n)})
    t = now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6
    for s in slots:
        if s > t + 0.01:
            return max(1.0, s - t)
    return max(1.0, 86400 - t + slots[0])


_state: dict[str, Any] = {
    "camera_base_url": "http://192.168.1.100",
    "interval_seconds": 43200,  # 12h
    "photos_dir": str(DEFAULT_PHOTOS_DIR),
    "journal_dir": str(DEFAULT_JOURNAL_DIR),
    "scheduler_enabled": True,
    "schedule_start_hour": 8,
    "schedule_start_minute": 0,
    "ai_analysis_enabled": False,
    "ai_backend": "ollama",
    "ai_base_url": "http://127.0.0.1:11434",
    "ai_model": "gemma4:e2b",
    # When True, browser uses /api/stream (Mac proxies); needed for Tailscale/off-LAN.
    "stream_via_server": False,
}
_save_lock = asyncio.Lock()
_scheduler_task: asyncio.Task | None = None


def _photo_mime(path: Path) -> str:
    s = path.suffix.lower()
    if s in (".jpg", ".jpeg"):
        return "image/jpeg"
    if s == ".png":
        return "image/png"
    if s == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _latest_saved_photo(photos_dir: str) -> Path | None:
    """Most recently modified image in the diary photos folder (same rules as /api/photos)."""
    root = Path(photos_dir).expanduser().resolve()
    if not root.is_dir():
        return None
    best: Path | None = None
    best_mtime = -1.0
    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if mt > best_mtime:
            best_mtime = mt
            best = p
    return best


def _safe_photo_path(filename: str) -> Path:
    name = os.path.basename((filename or "").strip())
    if not name or name in (".", ".."):
        raise HTTPException(404, "Invalid photo name")
    root = Path(_state["photos_dir"]).expanduser().resolve()
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise HTTPException(404, "Invalid photo path") from e
    if not candidate.is_file():
        raise HTTPException(404, "Photo not found")
    return candidate


def _normalize_base(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        raise ValueError("Camera URL is required")
    if not re.match(r"^https?://", url, re.I):
        raise ValueError("Camera URL must start with http:// or https://")
    return url


def _esp32_stream_base(main_base: str) -> str:
    """
    Espressif camera_web_server starts a second httpd on server_port+1 for /stream only
    (app_httpd.cpp: config.server_port += 1 before stream_httpd). Default: UI :80, MJPEG :81.
    """
    main_base = _normalize_base(main_base)
    p = urlparse(main_base)
    host = p.hostname
    if not host:
        raise ValueError("Camera URL must include a hostname")
    scheme = (p.scheme or "http").lower()
    port = p.port
    if port is None:
        port = 443 if scheme == "https" else 80
    stream_port = port + 1
    netloc = f"{host}:{stream_port}"
    return urlunparse((scheme, netloc, "", "", "", ""))


def load_config() -> None:
    if not CONFIG_PATH.is_file():
        return
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read config %s: %s", CONFIG_PATH, e)
        return
    migrated_photos = False
    if "camera_base_url" in data:
        try:
            _state["camera_base_url"] = _normalize_base(str(data["camera_base_url"]))
        except ValueError:
            pass
    if "interval_seconds" in data:
        v = int(data["interval_seconds"])
        _state["interval_seconds"] = max(0, min(v, 86400 * 14))
    if "photos_dir" in data and data["photos_dir"]:
        loaded = Path(data["photos_dir"]).expanduser()
        try:
            if loaded.resolve() == _LEGACY_HOME_DIARY_PHOTOS:
                _state["photos_dir"] = str(DEFAULT_PHOTOS_DIR)
                migrated_photos = True
            else:
                _state["photos_dir"] = str(loaded)
        except OSError:
            _state["photos_dir"] = str(loaded)
    if "journal_dir" in data and data["journal_dir"]:
        _state["journal_dir"] = str(Path(data["journal_dir"]).expanduser())
    if "scheduler_enabled" in data:
        _state["scheduler_enabled"] = bool(data["scheduler_enabled"])
    if "schedule_start_hour" in data:
        _state["schedule_start_hour"] = max(0, min(23, int(data["schedule_start_hour"])))
    if "schedule_start_minute" in data:
        _state["schedule_start_minute"] = max(0, min(59, int(data["schedule_start_minute"])))
    if "ai_analysis_enabled" in data:
        _state["ai_analysis_enabled"] = bool(data["ai_analysis_enabled"])
    if "ai_backend" in data:
        b = str(data["ai_backend"]).lower().strip()
        if b in ("ollama", "lmstudio"):
            _state["ai_backend"] = b
    if "ai_base_url" in data and str(data["ai_base_url"] or "").strip():
        _state["ai_base_url"] = str(data["ai_base_url"]).strip().rstrip("/")
    if "ai_model" in data and str(data["ai_model"] or "").strip():
        _state["ai_model"] = str(data["ai_model"]).strip()
    if "stream_via_server" in data:
        _state["stream_via_server"] = bool(data["stream_via_server"])
    if migrated_photos:
        save_config()
        logger.info("Migrated photos_dir from ~/Diary/Photos to %s", DEFAULT_PHOTOS_DIR)


def save_config() -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "camera_base_url": _state["camera_base_url"],
        "interval_seconds": _state["interval_seconds"],
        "photos_dir": _state["photos_dir"],
        "journal_dir": _state["journal_dir"],
        "scheduler_enabled": _state["scheduler_enabled"],
        "schedule_start_hour": _state["schedule_start_hour"],
        "schedule_start_minute": _state["schedule_start_minute"],
        "ai_analysis_enabled": _state["ai_analysis_enabled"],
        "ai_backend": _state["ai_backend"],
        "ai_base_url": _state["ai_base_url"],
        "ai_model": _state["ai_model"],
        "stream_via_server": _state["stream_via_server"],
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def save_snapshot_from_camera(*, reason: str) -> dict[str, Any]:
    base = _normalize_base(_state["camera_base_url"])
    url = f"{base}/capture"
    dest_dir = Path(_state["photos_dir"]).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    dest = dest_dir / f"diary_{stamp}.jpg"

    async with _camera_async_client(timeout=_httpx_timeout(connect=15.0, read=120.0)) as client:
        r = await client.get(url)
        r.raise_for_status()
        body = r.content

    async with _save_lock:
        dest.write_bytes(body)

    logger.info("Saved snapshot (%s) -> %s", reason, dest)
    out: dict[str, Any] = {"ok": True, "path": str(dest), "reason": reason}
    if _state.get("ai_analysis_enabled"):
        asyncio.create_task(_analyze_photo_report_task(dest, reason))
        out["analysis_queued"] = True
    else:
        out["analysis_queued"] = False
    return out


async def _analyze_photo_report_task(photo: Path, reason: str) -> None:
    if not _state.get("ai_analysis_enabled"):
        return
    backend = str(_state.get("ai_backend") or "ollama").lower()
    if backend not in ("ollama", "lmstudio"):
        backend = "ollama"
    base = str(_state.get("ai_base_url") or "http://127.0.0.1:11434").rstrip("/")
    model = str(_state.get("ai_model") or "gemma4:e2b").strip()
    journal_dir = Path(_state.get("journal_dir") or str(DEFAULT_JOURNAL_DIR)).expanduser()
    journal_dir.mkdir(parents=True, exist_ok=True)
    report_path = journal_dir / f"{photo.stem}_garden_report.md"
    err_path = journal_dir / f"{photo.stem}_garden_report_error.txt"
    timeout = _httpx_timeout(connect=30.0, read=600.0)
    try:
        body = await run_hydro_vision_report(
            image_path=photo,
            backend=backend,  # type: ignore[arg-type]
            base_url=base,
            model=model,
            timeout=timeout,
        )
        md = build_report_markdown(
            photo_path=photo,
            reason=reason,
            backend=backend,
            model=model,
            body=body,
        )
        report_path.write_text(md, encoding="utf-8")
        if err_path.is_file():
            err_path.unlink(missing_ok=True)
        logger.info("Garden report -> %s", report_path)
    except Exception as e:
        logger.exception("Garden AI report failed for %s: %s", photo, e)
        try:
            err_path.write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
        except OSError:
            pass


async def _scheduler_loop() -> None:
    while True:
        enabled = _state["scheduler_enabled"]
        interval = int(_state["interval_seconds"])
        if not enabled or interval <= 0:
            await asyncio.sleep(1.0)
            continue
        if interval > 0 and 86400 % interval == 0:
            delay = _seconds_until_next_clock_fire(
                datetime.now(),
                interval,
                int(_state["schedule_start_hour"]),
                int(_state["schedule_start_minute"]),
            )
            await asyncio.sleep(delay)
        else:
            await asyncio.sleep(float(interval))
        if not _state["scheduler_enabled"] or int(_state["interval_seconds"]) <= 0:
            continue
        try:
            await save_snapshot_from_camera(reason="scheduled")
        except Exception as e:
            logger.exception("Scheduled snapshot failed: %s", e)


app = FastAPI(title="Diary Cam Server", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    logging.basicConfig(level=logging.INFO)
    load_config()
    global _scheduler_task
    _scheduler_task = asyncio.create_task(_scheduler_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _scheduler_task:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass


@app.get("/api/settings")
async def api_get_settings() -> JSONResponse:
    try:
        stream_base = _esp32_stream_base(_state["camera_base_url"])
    except ValueError:
        stream_base = ""
    iv = int(_state["interval_seconds"])
    clock_aligned = iv > 0 and 86400 % iv == 0
    return JSONResponse(
        {
            "camera_base_url": _state["camera_base_url"],
            "camera_stream_base": stream_base,
            "interval_seconds": _state["interval_seconds"],
            "photos_dir": _state["photos_dir"],
            "journal_dir": _state["journal_dir"],
            "scheduler_enabled": _state["scheduler_enabled"],
            "schedule_start_hour": _state["schedule_start_hour"],
            "schedule_start_minute": _state["schedule_start_minute"],
            "schedule_clock_aligned": clock_aligned,
            "schedule_timezone_note": "local computer clock",
            "ai_analysis_enabled": _state["ai_analysis_enabled"],
            "ai_backend": _state["ai_backend"],
            "ai_base_url": _state["ai_base_url"],
            "ai_model": _state["ai_model"],
            "stream_via_server": _state["stream_via_server"],
        }
    )


@app.post("/api/settings")
async def api_post_settings(body: dict[str, Any] = Body(...)) -> JSONResponse:
    if "camera_base_url" in body:
        try:
            _state["camera_base_url"] = _normalize_base(str(body["camera_base_url"]))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    if "interval_seconds" in body:
        v = int(body["interval_seconds"])
        if v < 0 or v > 86400 * 14:
            raise HTTPException(400, "interval_seconds must be between 0 and 1209600")
        _state["interval_seconds"] = v
    if "photos_dir" in body:
        p = str(body["photos_dir"] or "").strip()
        if not p:
            raise HTTPException(400, "photos_dir must not be empty")
        _state["photos_dir"] = str(Path(p).expanduser())
    if "journal_dir" in body:
        j = str(body["journal_dir"] or "").strip()
        if not j:
            raise HTTPException(400, "journal_dir must not be empty")
        _state["journal_dir"] = str(Path(j).expanduser())
    if "scheduler_enabled" in body:
        _state["scheduler_enabled"] = bool(body["scheduler_enabled"])
    if "schedule_start_hour" in body:
        h = int(body["schedule_start_hour"])
        if h < 0 or h > 23:
            raise HTTPException(400, "schedule_start_hour must be 0–23")
        _state["schedule_start_hour"] = h
    if "schedule_start_minute" in body:
        m = int(body["schedule_start_minute"])
        if m < 0 or m > 59:
            raise HTTPException(400, "schedule_start_minute must be 0–59")
        _state["schedule_start_minute"] = m
    if "ai_analysis_enabled" in body:
        _state["ai_analysis_enabled"] = bool(body["ai_analysis_enabled"])
    if "ai_backend" in body:
        b = str(body["ai_backend"]).lower().strip()
        if b not in ("ollama", "lmstudio"):
            raise HTTPException(400, "ai_backend must be ollama or lmstudio")
        _state["ai_backend"] = b
    if "ai_base_url" in body:
        u = str(body["ai_base_url"] or "").strip().rstrip("/")
        if not u:
            raise HTTPException(400, "ai_base_url must not be empty")
        if not re.match(r"^https?://", u, re.I):
            raise HTTPException(400, "ai_base_url must start with http:// or https://")
        _state["ai_base_url"] = u
    if "ai_model" in body:
        m = str(body["ai_model"] or "").strip()
        if not m:
            raise HTTPException(400, "ai_model must not be empty")
        _state["ai_model"] = m
    if "stream_via_server" in body:
        _state["stream_via_server"] = bool(body["stream_via_server"])
    save_config()
    return await api_get_settings()


@app.get("/api/journal/latest")
async def api_journal_latest() -> JSONResponse:
    root = Path(_state.get("journal_dir") or str(DEFAULT_JOURNAL_DIR)).expanduser()
    if not root.is_dir():
        return JSONResponse(
            {
                "found": False,
                "message": "Journal folder does not exist yet.",
                "journal_dir": str(root),
            }
        )
    reports = list(root.glob("*_garden_report.md"))
    if not reports:
        return JSONResponse(
            {
                "found": False,
                "message": "No garden reports yet.",
                "journal_dir": str(root),
            }
        )
    latest = max(reports, key=lambda p: p.stat().st_mtime)
    text = latest.read_text(encoding="utf-8")
    html = markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "nl2br"],
    )
    st = latest.stat()
    return JSONResponse(
        {
            "found": True,
            "path": str(latest),
            "modified": st.st_mtime,
            "markdown": text,
            "html": html,
            "journal_dir": str(root),
        }
    )


@app.get("/api/health")
async def api_health() -> JSONResponse:
    """Reachability check from this machine to the camera (not from the browser)."""
    ok = False
    try:
        base = _normalize_base(_state["camera_base_url"])
        async with _camera_async_client(timeout=_httpx_timeout(connect=5.0, read=10.0)) as client:
            r = await client.get(f"{base}/status")
            ok = r.status_code == 200
    except (ValueError, httpx.RequestError) as e:
        logger.info("Health check failed: %s", e)
        ok = False
    except Exception as e:
        logger.exception("Health check unexpected error: %s", e)
        ok = False
    return JSONResponse({"camera_reachable": ok, "camera_base_url": _state["camera_base_url"]})


@app.get("/api/stream")
async def api_stream() -> StreamingResponse:
    stream_origin = _esp32_stream_base(_state["camera_base_url"])
    stream_url = f"{stream_origin}/stream"
    limits = httpx.Limits(max_keepalive_connections=0, max_connections=8)
    client = _camera_async_client(
        timeout=_httpx_timeout(connect=30.0, read=86400.0),
        limits=limits,
        headers={"Connection": "close"},
    )
    try:
        request = client.build_request("GET", stream_url)
        response = await client.send(request, stream=True)
        response.raise_for_status()
    except httpx.RequestError as e:
        await client.aclose()
        logger.warning("Camera stream failed (%s): %s", stream_url, e)
        raise HTTPException(502, f"Camera stream failed: {e}") from e
    except httpx.HTTPStatusError as e:
        await client.aclose()
        logger.warning("Camera stream HTTP error (%s): %s", stream_url, e)
        raise HTTPException(502, f"Camera stream failed: {e}") from e
    except Exception:
        await client.aclose()
        raise

    ct = response.headers.get("content-type", DEFAULT_STREAM_TYPE)

    async def gen():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(gen(), media_type=ct)


@app.get("/api/capture")
async def api_capture_proxy() -> Response:
    """Single-frame JPEG from the camera (proxy)."""
    base = _normalize_base(_state["camera_base_url"])
    try:
        async with _camera_async_client(timeout=_httpx_timeout(connect=15.0, read=120.0)) as client:
            r = await client.get(f"{base}/capture")
            r.raise_for_status()
            body = r.content
    except httpx.RequestError as e:
        raise HTTPException(502, f"Camera capture failed: {e}") from e
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Camera capture failed: {e}") from e
    return Response(content=body, media_type="image/jpeg")


@app.get("/api/photos")
async def api_photos_list() -> JSONResponse:
    root = Path(_state["photos_dir"]).expanduser().resolve()
    items: list[dict[str, Any]] = []
    if root.is_dir():
        for p in root.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            st = p.stat()
            items.append({"name": p.name, "mtime": st.st_mtime, "size": st.st_size})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return JSONResponse({"photos_dir": str(root), "count": len(items), "items": items})


@app.get("/api/photo/{filename}")
async def api_photo_file(filename: str) -> FileResponse:
    path = _safe_photo_path(filename)
    return FileResponse(path, media_type=_photo_mime(path))


@app.post("/api/chat")
async def api_chat(body: dict[str, Any] = Body(...)) -> JSONResponse:
    raw = body.get("messages")
    if not isinstance(raw, list):
        raise HTTPException(400, "messages must be a list")
    clean: list[dict[str, str]] = []
    for m in raw[-50:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        c = content.strip()
        if not c:
            continue
        clean.append({"role": role, "content": c[:48_000]})
    if not clean or clean[-1]["role"] != "user":
        raise HTTPException(400, "Send a non-empty user message as the last turn")
    model = str(_state.get("ai_model") or "").strip()
    base = str(_state.get("ai_base_url") or "").strip()
    if not model or not base:
        raise HTTPException(400, "Configure AI model and API base URL in Options first")
    latest = _latest_saved_photo(str(_state["photos_dir"]))
    if latest is None:
        raise HTTPException(
            400,
            "No saved photos yet. Take a snapshot first so the gardener can see your latest grow photo.",
        )
    backend = "lmstudio" if _state.get("ai_backend") == "lmstudio" else "ollama"
    timeout = _httpx_timeout(connect=15.0, read=300.0)
    try:
        reply = await run_gardener_chat(
            backend=backend,
            base_url=base,
            model=model,
            messages=clean,
            timeout=timeout,
            latest_photo=latest,
        )
    except ValueError as e:
        raise HTTPException(502, str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(502, f"AI request failed: {e}") from e
    return JSONResponse({"reply": reply, "context_photo": latest.name})


@app.post("/api/snapshot")
async def api_snapshot() -> JSONResponse:
    try:
        result = await save_snapshot_from_camera(reason="manual")
    except httpx.RequestError as e:
        raise HTTPException(502, f"Camera capture failed: {e}") from e
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Camera capture failed: {e}") from e
    except OSError as e:
        raise HTTPException(502, f"Could not save photo: {e}") from e
    return JSONResponse(result)


# Mount UI last so /api/* wins
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:

    @app.get("/")
    async def _missing_static() -> JSONResponse:
        return JSONResponse(
            {"error": f"static folder missing: {STATIC_DIR}"},
            status_code=500,
        )
