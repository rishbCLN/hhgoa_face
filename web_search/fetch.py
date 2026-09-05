"""Download the matched image so stage 4 can re-encode the face in it.

The URL comes from Google's response rather than from the operator, so the
fetcher is deliberately defensive: HTTPS/HTTP only, a hard byte cap enforced
while streaming (not just trusting ``Content-Length``), a short timeout, a
redirect limit, an ``image/*`` content-type requirement, and a refusal to fetch
anything that resolves to a private, loopback or link-local address.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any
from urllib.parse import urlparse

import requests

from errors import SearchError

#: Cap on the downloaded body. Streaming stops the moment this is exceeded.
MAX_BYTES = int(os.getenv("DOWNLOAD_MAX_BYTES", str(20 * 1024 * 1024)))

TIMEOUT = float(os.getenv("DOWNLOAD_TIMEOUT", "30"))

#: Some social CDNs 403 an empty User-Agent.
USER_AGENT = os.getenv(
    "DOWNLOAD_USER_AGENT",
    "Mozilla/5.0 (compatible; hhgoa-task3-face-verify/1.0; +local research script)",
)

#: Set DOWNLOAD_ALLOW_PRIVATE_HOSTS=1 to disable the private-address guard
#: (only useful when pointing the pipeline at a local fixture server).
_ALLOW_PRIVATE = os.getenv("DOWNLOAD_ALLOW_PRIVATE_HOSTS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _assert_public_host(url: str) -> None:
    """Reject URLs whose host resolves into a non-public address range."""
    if _ALLOW_PRIVATE:
        return

    host = urlparse(url).hostname
    if not host:
        raise SearchError(f"Cannot download {url!r}: no hostname in the URL.")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise SearchError(f"Could not resolve {host}: {exc}") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            raise SearchError(
                f"Refusing to download from {host} -- it resolves to the "
                f"non-public address {address}.",
                hint="Set DOWNLOAD_ALLOW_PRIVATE_HOSTS=1 only for local test fixtures.",
            )


def download_image(url: str, max_bytes: int = MAX_BYTES) -> bytes:
    """Fetch ``url`` and return its bytes, or raise :class:`SearchError`.

    Args:
        url: An ``http``/``https`` image URL, normally from Vision's response.
        max_bytes: Hard cap; the stream is abandoned once it is passed.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SearchError(
            f"Refusing to download {url!r}: only http/https URLs are allowed."
        )

    _assert_public_host(url)

    headers = {"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"}
    try:
        with requests.get(
            url, headers=headers, timeout=TIMEOUT, stream=True, allow_redirects=True
        ) as response:
            return _read_body(response, url, max_bytes)
    except requests.TooManyRedirects as exc:
        raise SearchError(f"Too many redirects while downloading {url}") from exc
    except requests.Timeout as exc:
        raise SearchError(
            f"Timed out after {TIMEOUT:.0f}s downloading {url}",
            hint="Raise DOWNLOAD_TIMEOUT, or try another candidate URL.",
        ) from exc
    except requests.RequestException as exc:
        raise SearchError(f"Could not download {url}: {exc}") from exc



def _read_body(response: Any, url: str, max_bytes: int) -> bytes:
    """Validate the response and stream its body under the byte cap."""
    if response.status_code != 200:
        raise SearchError(
            f"Downloading {url} returned HTTP {response.status_code}.",
            hint=(
                "Social networks often block direct image fetches from scripts. "
                "The match itself is still valid -- only the re-encode step needs "
                "the bytes. Try a different candidate, or save the image manually."
            ),
        )

    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type and not content_type.startswith("image/"):
        raise SearchError(
            f"{url} served Content-Type {content_type!r}, not an image.",
            hint="The URL probably points at an HTML page rather than the image file.",
        )

    declared = response.headers.get("Content-Length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise SearchError(
            f"{url} declares {int(declared) / 1e6:.1f} MB, over the "
            f"{max_bytes / 1e6:.0f} MB cap."
        )

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise SearchError(
                f"{url} exceeded the {max_bytes / 1e6:.0f} MB download cap.",
                hint="Raise DOWNLOAD_MAX_BYTES if the image really is that large.",
            )
        chunks.append(chunk)

    body = b"".join(chunks)
    if not body:
        raise SearchError(f"{url} returned an empty body.")
    return body


__all__ = ["MAX_BYTES", "TIMEOUT", "download_image"]
