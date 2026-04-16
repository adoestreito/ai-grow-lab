"""Vision LLM hydroponic review: Ollama or OpenAI-compatible (e.g. LM Studio)."""
from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx

logger = logging.getLogger("diary-cam")

# Text map always sent (backup if reference image missing). Matches grow_pod_reference.png.
GROW_POD_MAP = """## Pod map for this 12-pod unit (top-down)
Rows: **Back** (toward the light pole / back of unit), **Middle**, **Front** (toward the Gennup logo / front edge). Columns **1–4** from **left** to **right**.

| Crop | Pod(s) |
|------|--------|
| Bell pepper | Back row, pod2 |
| Radish | Middle row pod 2 **and** Front row pod 3 |
| Cherry tomato | Middle row pod 3 |
| Lettuce | **All pods not listed above:** Back 1, 3, 4; Middle 1, 4; Front 1, 2, 4. **Any pod without a specific crop in this table is lettuce.** |

**Orientation cues:** Water level tube with green float on the **left**. Front-left (Front 1) often has a **white flip lid**. Middle-right (Middle 4) may show a **green sprout-shaped plug**. **Silver light pole** sits center-back between the two middle pods of the back row. **GENNUP** branding on the front of the base.

When commenting, name pods as **Row + number** (e.g. "Middle 3" = cherry tomato)."""

GARDENER_CHAT_SYSTEM = f"""{GROW_POD_MAP}

You are the resident expert hydroponic gardener for this indoor Diary Cam system. Answer questions about care, nutrients, troubleshooting, scheduling, and what you can see in the grow.

When the user message includes an **attached snapshot**, that image is the **most recently saved** top-down photo from the Diary Cam—use it as the main visual context. If something is unclear in the photo, say so briefly.

Be concise, practical, and friendly. Use Markdown (short bullets, ## headings) when it helps."""

_CHAT_IMAGE_PREAMBLE = (
    "The image attached is the **latest saved top-down photo** from the Diary Cam. "
    "Ground your answer in what you see when relevant.\n\n"
)

HYDRO_SYSTEM = f"""{GROW_POD_MAP}

You are an expert hydroponic gardener reviewing photographs of this system.

Use only what is reasonably visible. If the image is unclear or something cannot be judged, say so briefly.

Reply in Markdown with exactly these sections (use ## headings):
## Observations
What you notice by pod where possible (tie to the map above), plus overall: stage, color, spacing, medium, water level, algae, biofilm, lighting, hardware.

## Assessment
Short interpretation: overall vigor, possible stress or deficiency *hypotheses* (not diagnoses), hygiene risks.

## Suggestions
Concrete, prioritized actions (nutrients, light distance/duration, airflow, cleaning, pH/EC checks if relevant). Keep bullets practical."""

BackendName = Literal["ollama", "lmstudio"]

MODULE_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE = MODULE_DIR / "grow_pod_reference.png"


def _ollama_chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/chat"


def _openai_chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def _mime_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _b64_file(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _build_gardener_ollama_messages(
    messages: list[dict[str, str]],
    latest_photo: Path | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": GARDENER_CHAT_SYSTEM}]
    if not messages:
        return out
    if messages[-1]["role"] != "user":
        raise ValueError("Last turn must be a user message")
    for m in messages[:-1]:
        out.append({"role": m["role"], "content": m["content"]})
    last = messages[-1]
    if latest_photo is not None and latest_photo.is_file():
        out.append(
            {
                "role": "user",
                "content": _CHAT_IMAGE_PREAMBLE + last["content"],
                "images": [_b64_file(latest_photo)],
            }
        )
    else:
        out.append({"role": "user", "content": last["content"]})
    return out


def _build_gardener_openai_messages(
    messages: list[dict[str, str]],
    latest_photo: Path | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": GARDENER_CHAT_SYSTEM}]
    if not messages:
        return out
    if messages[-1]["role"] != "user":
        raise ValueError("Last turn must be a user message")
    for m in messages[:-1]:
        out.append({"role": m["role"], "content": m["content"]})
    last = messages[-1]
    if latest_photo is not None and latest_photo.is_file():
        mime = _mime_for_path(latest_photo)
        b64 = _b64_file(latest_photo)
        out.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _CHAT_IMAGE_PREAMBLE + last["content"]},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        )
    else:
        out.append({"role": "user", "content": last["content"]})
    return out


async def run_gardener_chat(
    *,
    backend: BackendName,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: httpx.Timeout,
    latest_photo: Path | None = None,
) -> str:
    """Chat with optional vision: `latest_photo` is sent only on the current user turn."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("Missing API base URL")
    if not (model or "").strip():
        raise ValueError("Missing model name")
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Conversation must end with a user message")
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        if backend == "ollama":
            url = _ollama_chat_url(base)
            full_messages = _build_gardener_ollama_messages(messages, latest_photo)
            payload: dict[str, Any] = {
                "model": model.strip(),
                "messages": full_messages,
                "stream": False,
            }
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data.get("message") or {}
            text = (msg.get("content") or "").strip()
            if not text:
                raise ValueError(f"Ollama returned empty content: {data!r:.500}")
            return text

        url = _openai_chat_url(base)
        full_messages = _build_gardener_openai_messages(messages, latest_photo)
        payload = {
            "model": model.strip(),
            "messages": full_messages,
            "max_tokens": 4096,
            "temperature": 0.55,
        }
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"No choices in response: {data!r:.500}")
        msg = choices[0].get("message") or {}
        c = msg.get("content")
        if isinstance(c, list):
            parts = []
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text") or "")
            text = "\n".join(parts).strip()
        else:
            text = (c or "").strip()
        if not text:
            raise ValueError(f"LM Studio / OpenAI response had no text: {data!r:.500}")
        return text


async def run_hydro_vision_report(
    *,
    image_path: Path,
    backend: BackendName,
    base_url: str,
    model: str,
    timeout: httpx.Timeout,
) -> str:
    live_b64 = _b64_file(image_path)
    live_mime = _mime_for_path(image_path)

    use_ref = REFERENCE_IMAGE.is_file()
    if use_ref:
        ref_b64 = _b64_file(REFERENCE_IMAGE)
        ref_mime = _mime_for_path(REFERENCE_IMAGE)
        user_text = (
            "The FIRST image is the fixed **annotated pod layout reference** (arrows/labels) for this grower. "
            "The SECOND image is **today's live top-down photo** from the camera. "
            "Align orientation using the cues in the system message. "
            "Any pod not given a specific crop in the table is **lettuce**. "
            "Write the report sections."
        )
        logger.info("Vision report: reference diagram + live frame (%s)", image_path.name)
    else:
        ref_b64 = ""
        ref_mime = ""
        user_text = (
            "Review this **live grow photo** and write the report sections. "
            "Use the pod map in the system message; unlisted pods are lettuce."
        )
        logger.warning("Missing %s — text-only pod map sent", REFERENCE_IMAGE.name)

    async with httpx.AsyncClient(timeout=timeout) as client:
        if backend == "ollama":
            url = _ollama_chat_url(base_url)
            images: list[str] = [ref_b64, live_b64] if use_ref else [live_b64]
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": HYDRO_SYSTEM},
                    {
                        "role": "user",
                        "content": user_text,
                        "images": images,
                    },
                ],
                "stream": False,
            }
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data.get("message") or {}
            text = (msg.get("content") or "").strip()
            if not text:
                raise ValueError(f"Ollama returned empty content: {data!r:.500}")
            return text

        url = _openai_chat_url(base_url)
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if use_ref:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{ref_mime};base64,{ref_b64}"},
                }
            )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{live_mime};base64,{live_b64}"},
            }
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": HYDRO_SYSTEM},
                {"role": "user", "content": content},
            ],
            "max_tokens": 4096,
            "temperature": 0.4,
        }
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"No choices in response: {data!r:.500}")
        msg = choices[0].get("message") or {}
        c = msg.get("content")
        if isinstance(c, list):
            parts = []
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text") or "")
            text = "\n".join(parts).strip()
        else:
            text = (c or "").strip()
        if not text:
            raise ValueError(f"LM Studio / OpenAI response had no text: {data!r:.500}")
        return text


def build_report_markdown(
    *,
    photo_path: Path,
    reason: str,
    backend: str,
    model: str,
    body: str,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join(
        [
            "# Hydroponic grow report",
            "",
            f"- **Photo:** `{photo_path}`",
            f"- **Generated:** {ts}",
            f"- **Trigger:** {reason}",
            f"- **Backend:** {backend}",
            f"- **Model:** {model}",
            "",
            "---",
            "",
            body,
            "",
        ]
    )
