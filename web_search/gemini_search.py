"""Stage 2: Reverse-image search and visual identity analysis via Google Gemini API.

Uses Google Gemini (e.g. gemini-3.8-flash) from Google AI Studio / Google Labs.
Does not require Google Cloud billing or paid cloud credentials.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import requests

from errors import ConfigError, SearchError
from web_search.vision_search import SearchResult, is_social_url

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.8-flash"
_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "45"))


def resolve_gemini_key() -> str | None:
    """Return GEMINI_API_KEY from environment, trimmed."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


def resolve_gemini_model() -> str:
    """Return configured GEMINI_MODEL or default."""
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def credential_summary() -> str:
    """Masked summary of active Gemini credentials for display."""
    key = resolve_gemini_key()
    model = resolve_gemini_model()
    if not key:
        return "none"
    if len(key) <= 8:
        masked = "***"
    else:
        masked = f"{key[:4]}...{key[-4:]}"
    return f"Gemini ({model}, key {masked})"


def parse_gemini_response(payload: Mapping[str, Any]) -> list[SearchResult]:
    """Parse Gemini's JSON candidates into a list of SearchResult objects."""
    candidates_raw = payload.get("candidates", [])
    if not candidates_raw:
        return []

    content = candidates_raw[0].get("content", {})
    parts = content.get("parts", [])
    if not parts:
        return []

    text = parts[0].get("text", "").strip()
    if not text:
        return []

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    matches = data.get("matches", [])
    results: list[SearchResult] = []

    for item in matches:
        url = str(item.get("url", "")).strip()
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            continue

        kind = str(item.get("kind", "page_full_match")).strip()
        if kind not in (
            "page_full_match",
            "page_partial_match",
            "full_match_image",
            "partial_match_image",
            "visually_similar_image",
        ):
            kind = "page_full_match"

        try:
            score = float(item.get("confidence", 0.85))
        except (ValueError, TypeError):
            score = 0.85

        title = item.get("title") or item.get("page_title")
        image_urls_raw = item.get("image_urls") or []
        image_urls = [
            str(u).strip()
            for u in image_urls_raw
            if str(u).strip().startswith("http")
        ]

        # If it is an image kind and image_urls is empty, include the url itself
        if kind.endswith("_image") and url not in image_urls:
            image_urls.append(url)

        results.append(
            SearchResult(
                url=url,
                kind=kind,
                score=score,
                page_title=str(title) if title else None,
                image_urls=image_urls,
            )
        )

    return results


def search_image(
    image_path: str | Path,
    max_results: int = 10,
    save_raw: str | None = None,
) -> list[SearchResult]:
    """Analyze and reverse-search an image using Gemini API."""
    key = resolve_gemini_key()
    if not key:
        raise ConfigError(
            "Gemini API key is not configured.",
            hint=(
                "Set GEMINI_API_KEY in your .env file with your key from "
                "Google AI Studio (https://aistudio.google.com/)."
            ),
        )

    path = Path(image_path)
    if not path.is_file():
        raise SearchError(f"Image not found: {image_path}")

    image_bytes = path.read_bytes()
    if not image_bytes:
        raise SearchError("Image file is empty.")

    ext = path.suffix.lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"
    b64_img = base64.b64encode(image_bytes).decode("utf-8")

    model = resolve_gemini_model()
    endpoint = f"{GEMINI_API_BASE}/{model}:generateContent?key={key}"

    prompt = (
        "You are an expert reverse-image researcher and OSINT facial analyst.\n"
        "Analyze this photo of a person.\n"
        "1. Identify who this person is if publicly known, or describe their distinctive features.\n"
        "2. Identify probable online matching pages, public profiles, or publication URLs where "
        "this person or image appears (e.g., LinkedIn, Instagram, Twitter/X, YouTube, Wikimedia, "
        "GitHub, Behance, or news/blog sites).\n"
        f"3. Return up to {max_results} candidate matches as a JSON object formatted strictly as:\n"
        "{\n"
        '  "matches": [\n'
        '    {\n'
        '      "url": "https://example.com/profile/path",\n'
        '      "title": "Page or post title",\n'
        '      "kind": "page_full_match",\n'
        '      "confidence": 0.90,\n'
        '      "image_urls": ["https://example.com/avatar.jpg"]\n'
        "    }\n"
        "  ],\n"
        '  "person_name": "Name if known, else Unknown",\n'
        '  "summary": "Brief summary of image analysis"\n'
        "}\n"
        "If the image is a private personal photo with no known public web matches, return an "
        'empty "matches" list.'
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_img,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json"
        },
    }

    models_to_try = [model]
    if model != "gemini-3.5-flash":
        models_to_try.append("gemini-3.5-flash")
    if "gemini-3.1-flash-lite" not in models_to_try:
        models_to_try.append("gemini-3.1-flash-lite")

    import time
    last_response = None
    data = None

    for m in models_to_try:
        endpoint = f"{GEMINI_API_BASE}/{m}:generateContent?key={key}"
        for attempt in range(2):
            try:
                response = requests.post(
                    endpoint,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=_TIMEOUT,
                )
                last_response = response
            except requests.RequestException as exc:
                if attempt == 0:
                    time.sleep(2)
                    continue
                raise SearchError(
                    f"Gemini API network call failed: {exc}",
                    hint="Check your internet connection and proxy settings.",
                ) from exc

            if response.status_code == 200:
                try:
                    data = response.json()
                    break
                except ValueError as exc:
                    raise SearchError("Gemini response is not valid JSON.") from exc

            if response.status_code == 503:
                # Spikes in demand: wait briefly and retry
                time.sleep(2.5)
                continue
            if response.status_code == 429:
                time.sleep(2)
                continue
            break

        if data is not None:
            break

    if data is None:
        if last_response is not None:
            if last_response.status_code == 429:
                raise SearchError(
                    "Gemini API rate limit or quota exceeded.",
                    hint="Wait a minute or check your quota at https://aistudio.google.com/.",
                )
            raise SearchError(
                f"Gemini API returned HTTP {last_response.status_code}: {last_response.text}"
            )
        raise SearchError("No response received from Gemini API.")

    if save_raw:
        try:
            Path(save_raw).write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise SearchError(f"Failed to save raw search JSON to {save_raw}: {exc}") from exc

    return parse_gemini_response(data)
