"""Stage 2: live reverse-image search via Google Cloud Vision (Web Detection).

    THIS MODULE ALWAYS TALKS TO THE REAL API. There is no demo mode, no cached
    sample response, and no fallback that invents a match. If credentials are
    missing it raises; if the API errors it raises; if the API returns nothing it
    returns an empty list and the pipeline reports "no match" honestly.
    (Task constraint 1.2.) The only canned Vision JSON in this repo lives in
    ``tests/test_search_parser.py`` and is handed directly to
    :func:`parse_web_detection`, which does no I/O.

Two credential styles are supported, checked in this order:

1. ``GOOGLE_VISION_API_KEY`` -- a plain API key, appended as ``?key=...``.
   Simplest to create; restrict it to the Vision API in the Cloud console.
2. ``GOOGLE_APPLICATION_CREDENTIALS`` -- path to a service-account JSON file.
   Exchanged for a short-lived OAuth2 bearer token via ``google-auth``.

Cost: Web Detection is billed per image under Vision's free tier of 1,000
units/month. One pipeline run = one unit.

API reference: https://cloud.google.com/vision/docs/detecting-web
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import requests

from errors import ConfigError, SearchError

VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

#: OAuth2 scope required for the service-account path.
_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

_TIMEOUT = float(os.getenv("VISION_TIMEOUT", "30"))
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # Vision rejects payloads above ~20 MB.


#: Hosts treated as social media. Override or extend with the SOCIAL_DOMAINS env
#: var (comma-separated). Matching is on the host suffix, so "instagram.com"
#: also matches "www.instagram.com".
DEFAULT_SOCIAL_DOMAINS: tuple[str, ...] = (
    "instagram.com",
    "facebook.com",
    "fb.com",
    "fb.watch",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "reddit.com",
    "redd.it",
    "pinterest.com",
    "pin.it",
    "tumblr.com",
    "threads.net",
    "threads.com",
    "snapchat.com",
    "bsky.app",
    "mastodon.social",
    "flickr.com",
    "vk.com",
    "weibo.com",
    "t.me",
    "telegram.me",
    "medium.com",
    "quora.com",
    "twitch.tv",
    "vimeo.com",
    "deviantart.com",
    "behance.net",
    "dribbble.com",
    "500px.com",
)


def _load_social_domains() -> tuple[str, ...]:
    override = os.getenv("SOCIAL_DOMAINS", "").strip()
    if not override:
        return DEFAULT_SOCIAL_DOMAINS
    domains = tuple(
        d.strip().lower().removeprefix("www.") for d in override.split(",") if d.strip()
    )
    return domains or DEFAULT_SOCIAL_DOMAINS


SOCIAL_DOMAINS = _load_social_domains()


#: Result kinds, best evidence first. ``visually_similar_image`` is deliberately
#: absent: Vision returns look-alike photos there, not copies of *this* photo, so
#: treating one as a match would overstate the finding.
MATCH_PRIORITY: tuple[str, ...] = (
    "page_full_match",
    "page_partial_match",
    "full_match_image",
    "partial_match_image",
)

_ALL_KINDS = MATCH_PRIORITY + ("visually_similar_image",)


@dataclass
class SearchResult:
    """One normalised hit from Vision's Web Detection response."""

    #: Page URL (for ``page_*`` kinds) or direct image URL (for image kinds).
    url: str
    #: One of :data:`_ALL_KINDS`.
    kind: str
    #: Vision's confidence, often 0.0 for page results -- do not rank on it alone.
    score: float = 0.0
    #: ``pageTitle`` when Vision supplied one.
    page_title: str | None = None
    #: For page hits: the matching image URLs Vision found *on* that page. Stage 4
    #: downloads one of these to re-encode the face.
    image_urls: list[str] = field(default_factory=list)

    @property
    def host(self) -> str:
        return (urlparse(self.url).hostname or "").lower()

    @property
    def is_social(self) -> bool:
        return is_social_url(self.url)

    @property
    def priority(self) -> int:
        try:
            return MATCH_PRIORITY.index(self.kind)
        except ValueError:
            return len(MATCH_PRIORITY) + 1

    def best_image_url(self) -> str | None:
        """The image URL to download for re-encoding, if there is one."""
        if self.image_urls:
            return self.image_urls[0]
        if self.kind.endswith("_image"):
            return self.url
        return None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(host=self.host, is_social=self.is_social)
        return payload



def is_social_url(url: str, domains: Sequence[str] = SOCIAL_DOMAINS) -> bool:
    """True if ``url``'s host is (a subdomain of) a known social-media domain."""
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False

    host = (parsed.hostname or "").lower().removeprefix("www.")
    if not host:
        return False

    return any(host == domain or host.endswith("." + domain) for domain in domains)


# --------------------------------------------------------------------------- #
# Response parsing -- pure, no I/O, unit-tested with canned JSON
# --------------------------------------------------------------------------- #

def parse_web_detection(payload: Mapping[str, Any]) -> list[SearchResult]:
    """Normalise a Vision ``images:annotate`` response into search results.

    Accepts either the full response (``{"responses": [{...}]}``), a single
    annotation (``{"webDetection": {...}}``), or a bare ``webDetection`` object,
    because all three shapes show up in the wild (REST, client libraries, docs).

    Raises :class:`SearchError` if the response carries an ``error`` block.
    """
    if not isinstance(payload, Mapping):
        raise SearchError(f"Vision response was {type(payload).__name__}, expected an object.")

    annotation: Mapping[str, Any] = payload
    responses = payload.get("responses")
    if isinstance(responses, list):
        if not responses:
            return []
        first = responses[0]
        if not isinstance(first, Mapping):
            raise SearchError("Vision response['responses'][0] is not an object.")
        annotation = first

    error = annotation.get("error") or payload.get("error")
    if isinstance(error, Mapping) and (error.get("message") or error.get("code")):
        raise SearchError(
            f"Vision API returned an error: {error.get('message', error)} "
            f"(code {error.get('code', '?')})",
            hint="Check that the Cloud Vision API is enabled for the project.",
        )

    web = annotation.get("webDetection", annotation)
    if not isinstance(web, Mapping):
        raise SearchError("Vision response contained no usable 'webDetection' object.")

    return _collect_results(web)



def _image_urls(entries: Any) -> list[str]:
    """Pull ``url`` out of a list of Vision image objects, preserving order."""
    urls: list[str] = []
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping):
                url = entry.get("url")
                if isinstance(url, str) and url:
                    urls.append(url)
    return urls


def _collect_results(web: Mapping[str, Any]) -> list[SearchResult]:
    """Flatten a ``webDetection`` object into de-duplicated results."""
    results: list[SearchResult] = []

    for page in web.get("pagesWithMatchingImages") or []:
        if not isinstance(page, Mapping):
            continue
        url = page.get("url")
        if not isinstance(url, str) or not url:
            continue

        full = _image_urls(page.get("fullMatchingImages"))
        partial = _image_urls(page.get("partialMatchingImages"))
        results.append(
            SearchResult(
                url=url,
                kind="page_full_match" if full else "page_partial_match",
                score=_as_float(page.get("score")),
                page_title=_as_text(page.get("pageTitle")),
                image_urls=full + partial,
            )
        )

    for key, kind in (
        ("fullMatchingImages", "full_match_image"),
        ("partialMatchingImages", "partial_match_image"),
        ("visuallySimilarImages", "visually_similar_image"),
    ):
        for entry in web.get(key) or []:
            if not isinstance(entry, Mapping):
                continue
            url = entry.get("url")
            if isinstance(url, str) and url:
                results.append(
                    SearchResult(url=url, kind=kind, score=_as_float(entry.get("score")))
                )

    return _dedupe(results)



def _as_float(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _as_text(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _dedupe(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Keep the strongest entry per URL, then sort by evidence strength.

    Vision often lists the same URL as both a page hit and an image hit; keeping
    the higher-priority one avoids reporting a duplicate "match".
    """
    best: dict[str, SearchResult] = {}
    for result in results:
        existing = best.get(result.url)
        if existing is None or result.priority < existing.priority:
            if existing is not None and not result.image_urls:
                result.image_urls = existing.image_urls
            best[result.url] = result
        elif not existing.image_urls and result.image_urls:
            existing.image_urls = result.image_urls

    ordered = sorted(
        best.values(),
        key=lambda r: (r.priority, -r.score, r.url),
    )
    return ordered


def social_candidates(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Eligible social-media matches, strongest evidence first.

    Filters to social hosts *and* drops ``visually_similar_image`` -- a look-alike
    photo on a social profile is not evidence that this photo was posted there.
    """
    eligible = [
        r for r in results if r.is_social and r.kind in MATCH_PRIORITY
    ]
    return sorted(eligible, key=lambda r: (r.priority, -r.score, r.url))



def find_social_match_record(results: Iterable[SearchResult]) -> SearchResult | None:
    """Best social-media match, or ``None`` when there is none.

    ``None`` is a legitimate outcome: the photo may simply not be indexed by
    Google, or may be indexed only on non-social sites.
    """
    candidates = social_candidates(results)
    return candidates[0] if candidates else None


def find_social_match(results: Iterable[SearchResult]) -> str | None:
    """The URL of the best social-media match, or ``None``.

    Thin wrapper over :func:`find_social_match_record` for callers that only want
    the URL (the signature named in the task brief).
    """
    match = find_social_match_record(results)
    return match.url if match else None


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #

_SETUP_HINT = (
    "No Google Cloud Vision credential found. Set ONE of:\n"
    "  GOOGLE_VISION_API_KEY=<api key>                  (simplest)\n"
    "  GOOGLE_APPLICATION_CREDENTIALS=<path to sa.json>  (service account)\n"
    "Copy .env.example to .env and fill it in. Full walkthrough: README.md "
    "-> 'Setup checklist (one-time, human)'. The free tier covers 1,000 "
    "Web Detection calls/month and one pipeline run costs 1."
)


def credential_summary() -> str:
    """Which credential style is configured -- for the CLI banner. Never the value."""
    if os.getenv("GOOGLE_VISION_API_KEY", "").strip():
        return "GOOGLE_VISION_API_KEY (api key)"
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if path:
        return f"GOOGLE_APPLICATION_CREDENTIALS ({path})"
    return "none"



def _bearer_token_from_service_account(path: str) -> str:
    """Exchange a service-account JSON file for a short-lived OAuth2 token."""
    key_file = Path(path)
    if not key_file.exists():
        raise ConfigError(
            f"GOOGLE_APPLICATION_CREDENTIALS points at {key_file}, which does not exist.",
            hint=_SETUP_HINT,
        )

    try:
        from google.auth.transport.requests import Request  # type: ignore[import-not-found]
        from google.oauth2 import service_account  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigError(
            f"The service-account credential path needs google-auth ({exc}).",
            hint="pip install -r requirements.txt, or use GOOGLE_VISION_API_KEY instead.",
        ) from exc

    try:
        credentials = service_account.Credentials.from_service_account_file(
            str(key_file), scopes=[_SCOPE]
        )
        credentials.refresh(Request())
    except Exception as exc:  # noqa: BLE001 - google-auth raises many types
        raise ConfigError(
            f"Could not obtain an access token from {key_file}: {exc}",
            hint=(
                "Confirm the file is a service-account key (it has "
                '"type": "service_account"), that the key is not disabled, and '
                "that the Cloud Vision API is enabled for its project."
            ),
        ) from exc

    if not credentials.token:
        raise ConfigError(f"google-auth returned an empty token for {key_file}.")
    return str(credentials.token)


def _request_target() -> tuple[str, dict[str, str]]:
    """Resolve the endpoint URL and auth headers, or raise :class:`ConfigError`."""
    api_key = os.getenv("GOOGLE_VISION_API_KEY", "").strip()
    if api_key:
        return f"{VISION_ENDPOINT}?key={api_key}", {}

    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if sa_path:
        token = _bearer_token_from_service_account(sa_path)
        return VISION_ENDPOINT, {"Authorization": f"Bearer {token}"}

    raise ConfigError("Google Cloud Vision credentials are not configured.", hint=_SETUP_HINT)



def search_image(
    image_path: str | Path,
    max_results: int = 50,
    save_raw_to: str | Path | None = None,
) -> list[SearchResult]:
    """Run Web Detection on ``image_path`` against the live Vision API.

    Args:
        image_path: Local image file to search for.
        max_results: ``maxResults`` passed to Vision (its cap on each result list).
        save_raw_to: Optional path to dump the raw JSON response for auditing.

    Returns:
        Normalised :class:`SearchResult` objects, strongest evidence first. An
        empty list means Vision genuinely found nothing -- not an error.

    Raises:
        ConfigError: no usable credential (see constraint 1.1).
        SearchError: the request failed, or Vision reported an error.
    """
    path = Path(image_path)
    if not path.exists():
        raise SearchError(f"Cannot search for {path}: file not found.")

    content = path.read_bytes()
    if not content:
        raise SearchError(f"Cannot search for {path}: file is empty.")
    if len(content) > _MAX_IMAGE_BYTES:
        raise SearchError(
            f"{path} is {len(content) / 1e6:.1f} MB; the Vision API rejects "
            f"payloads over {_MAX_IMAGE_BYTES / 1e6:.0f} MB.",
            hint="Downscale the photo before searching.",
        )

    url, headers = _request_target()
    body = {
        "requests": [
            {
                "image": {"content": base64.b64encode(content).decode("ascii")},
                "features": [{"type": "WEB_DETECTION", "maxResults": int(max_results)}],
                "imageContext": {"webDetectionParams": {"includeGeoResults": False}},
            }
        ]
    }

    payload = _post(url, headers, body)

    if save_raw_to:
        target = Path(save_raw_to)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return parse_web_detection(payload)



_STATUS_HINTS = {
    400: "Malformed request -- usually an unsupported image format.",
    401: "The credential was rejected. Regenerate the key / service-account file.",
    403: (
        "Access denied. Enable the Cloud Vision API for the project, confirm "
        "billing is attached (required even on the free tier), and check any "
        "API-key restrictions."
    ),
    404: "Endpoint not found -- check VISION_ENDPOINT has not been overridden.",
    429: "Quota exhausted. The free tier is 1,000 Web Detection units/month.",
    500: "Google-side error. Retry in a moment.",
    503: "Vision API temporarily unavailable. Retry in a moment.",
}


def _post(url: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> dict[str, Any]:
    """POST to Vision and return the decoded JSON, mapping failures to SearchError."""
    request_headers = {"Content-Type": "application/json; charset=utf-8", **headers}

    try:
        response = requests.post(
            url, headers=request_headers, json=body, timeout=_TIMEOUT
        )
    except requests.Timeout as exc:
        raise SearchError(
            f"The Vision API did not respond within {_TIMEOUT:.0f}s.",
            hint="Raise VISION_TIMEOUT or check your network connection.",
        ) from exc
    except requests.RequestException as exc:
        raise SearchError(f"Could not reach the Vision API: {exc}") from exc

    if response.status_code != 200:
        detail = _error_detail(response)
        raise SearchError(
            f"Vision API returned HTTP {response.status_code}: {detail}",
            hint=_STATUS_HINTS.get(response.status_code),
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise SearchError(
            f"Vision API returned non-JSON content: {response.text[:200]!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise SearchError(f"Vision API returned {type(payload).__name__}, expected an object.")
    return payload


def _error_detail(response: "requests.Response") -> str:
    """Best-effort extraction of Google's error message from a failed response."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:300] or "<empty body>"
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, Mapping):
            return str(error.get("message") or error)
        if isinstance(error, str):
            return error
    return json.dumps(data)[:300]



__all__ = [
    "DEFAULT_SOCIAL_DOMAINS",
    "MATCH_PRIORITY",
    "SOCIAL_DOMAINS",
    "VISION_ENDPOINT",
    "SearchResult",
    "credential_summary",
    "find_social_match",
    "find_social_match_record",
    "is_social_url",
    "parse_web_detection",
    "search_image",
    "social_candidates",
]


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m web_search.vision_search --image photo.jpg`` -- stage 2 alone."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Google Vision Web Detection on one image (live API call).",
    )
    parser.add_argument("--image", required=True, help="Path to the image to search for.")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--save-raw", help="Write the raw Vision JSON response here.")
    parser.add_argument(
        "--all", action="store_true", help="List every hit, not just social matches."
    )
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from errors import PipelineError

    try:
        results = search_image(args.image, args.max_results, args.save_raw)
    except PipelineError as exc:
        print(f"ERROR: {exc.message}")
        if exc.hint:
            print(f"\n{exc.hint}")
        return exc.exit_code

    print(f"{len(results)} result(s) from Vision Web Detection.")
    for result in results if args.all else social_candidates(results):
        flag = "SOCIAL" if result.is_social else "      "
        print(f"  [{flag}] {result.kind:<22} {result.url}")

    match = find_social_match(results)
    print(f"\nBest social match: {match or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
