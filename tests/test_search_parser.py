"""Tests for parsing Google Vision Web Detection responses.

    THE JSON IN THIS FILE IS A TEST FIXTURE. It exists only to exercise
    ``parse_web_detection`` / ``find_social_match``, which are pure functions over
    a response object. It is defined in this test module, is never imported by
    ``pipeline.py`` or ``web_search``, and there is no code path from the pipeline
    that could reach it. Task constraint 1.2: the pipeline always performs a real
    API call and never substitutes a canned match.

The last two tests actively enforce that: with no credential configured,
``search_image`` raises; with the network broken, it raises. Neither invents a
result.
"""

from __future__ import annotations

import json

import pytest

import web_search.vision_search as vs
from errors import ConfigError, SearchError

# --------------------------------------------------------------------------- #
# Fixture: the shape Vision returns for images:annotate with WEB_DETECTION
# --------------------------------------------------------------------------- #

FAKE_VISION_RESPONSE = {
    "responses": [
        {
            "webDetection": {
                "webEntities": [
                    {"entityId": "/m/0dzct", "score": 0.71, "description": "Face"}
                ],
                "fullMatchingImages": [
                    {"url": "https://scontent.cdninstagram.com/v/t51/full.jpg", "score": 0.9},
                    {"url": "https://cdn.example-blog.com/img/hero.jpg", "score": 0.4},
                ],
                "partialMatchingImages": [
                    {"url": "https://pbs.twimg.com/media/partial.jpg", "score": 0.3}
                ],
                "pagesWithMatchingImages": [
                    {
                        "url": "https://www.instagram.com/p/CxAmPl3/",
                        "pageTitle": "Sample Person on Instagram",
                        "score": 0.0,
                        "fullMatchingImages": [
                            {"url": "https://scontent.cdninstagram.com/v/t51/full.jpg"}
                        ],
                    },
                    {
                        "url": "https://example-blog.com/2026/03/some-article",
                        "pageTitle": "Some article",
                        "partialMatchingImages": [
                            {"url": "https://cdn.example-blog.com/img/hero.jpg"}
                        ],
                    },
                    {
                        "url": "https://twitter.com/sampleperson/status/1234567890",
                        "pageTitle": "Sample Person on X",
                        "partialMatchingImages": [
                            {"url": "https://pbs.twimg.com/media/partial.jpg"}
                        ],
                    },
                ],
                "visuallySimilarImages": [
                    {"url": "https://www.instagram.com/p/L00KALIKE/", "score": 0.2},
                    {"url": "https://images.example.net/similar.jpg", "score": 0.1},
                ],
                "bestGuessLabels": [{"label": "person", "languageCode": "en"}],
            }
        }
    ]
}


def test_parses_every_result_kind():
    results = vs.parse_web_detection(FAKE_VISION_RESPONSE)
    by_url = {r.url: r for r in results}

    assert by_url["https://www.instagram.com/p/CxAmPl3/"].kind == "page_full_match"
    assert by_url["https://twitter.com/sampleperson/status/1234567890"].kind == (
        "page_partial_match"
    )
    assert by_url["https://example-blog.com/2026/03/some-article"].kind == (
        "page_partial_match"
    )
    assert by_url["https://pbs.twimg.com/media/partial.jpg"].kind == "partial_match_image"
    assert by_url["https://images.example.net/similar.jpg"].kind == (
        "visually_similar_image"
    )


def test_accepts_all_three_response_shapes():
    """Full response, single annotation, and a bare webDetection object."""
    full = vs.parse_web_detection(FAKE_VISION_RESPONSE)
    annotation = vs.parse_web_detection(FAKE_VISION_RESPONSE["responses"][0])
    bare = vs.parse_web_detection(FAKE_VISION_RESPONSE["responses"][0]["webDetection"])

    assert [r.url for r in full] == [r.url for r in annotation] == [r.url for r in bare]


def test_page_results_carry_their_image_urls():
    results = vs.parse_web_detection(FAKE_VISION_RESPONSE)
    page = next(r for r in results if r.url == "https://www.instagram.com/p/CxAmPl3/")

    assert page.image_urls == ["https://scontent.cdninstagram.com/v/t51/full.jpg"]
    assert page.best_image_url() == "https://scontent.cdninstagram.com/v/t51/full.jpg"
    assert page.page_title == "Sample Person on Instagram"


def test_results_are_ordered_strongest_evidence_first():
    results = vs.parse_web_detection(FAKE_VISION_RESPONSE)
    priorities = [r.priority for r in results]
    assert priorities == sorted(priorities)
    assert results[0].kind == "page_full_match"


def test_duplicate_urls_are_collapsed_to_the_strongest_kind():
    """The same URL listed as both a page and an image must appear once."""
    payload = {
        "webDetection": {
            "fullMatchingImages": [{"url": "https://www.instagram.com/p/DUPE/"}],
            "pagesWithMatchingImages": [
                {
                    "url": "https://www.instagram.com/p/DUPE/",
                    "fullMatchingImages": [{"url": "https://cdn.example.com/a.jpg"}],
                }
            ],
        }
    }
    results = vs.parse_web_detection(payload)

    assert len(results) == 1
    assert results[0].kind == "page_full_match"
    assert results[0].image_urls == ["https://cdn.example.com/a.jpg"]



def test_empty_and_missing_web_detection_yield_no_results():
    assert vs.parse_web_detection({"responses": []}) == []
    assert vs.parse_web_detection({"responses": [{}]}) == []
    assert vs.parse_web_detection({"webDetection": {}}) == []


def test_malformed_entries_are_skipped_not_fatal():
    payload = {
        "webDetection": {
            "fullMatchingImages": [
                "not-an-object",
                {"no_url_key": 1},
                {"url": ""},
                {"url": "https://ok.example.com/a.jpg"},
            ],
            "pagesWithMatchingImages": [None, {"url": None}],
        }
    }
    results = vs.parse_web_detection(payload)
    assert [r.url for r in results] == ["https://ok.example.com/a.jpg"]


def test_api_error_block_raises_search_error():
    payload = {
        "responses": [
            {"error": {"code": 7, "message": "This API method requires billing."}}
        ]
    }
    with pytest.raises(SearchError, match="billing"):
        vs.parse_web_detection(payload)


def test_non_object_payload_raises():
    with pytest.raises(SearchError):
        vs.parse_web_detection(["not", "a", "dict"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Social-domain filtering
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.instagram.com/p/abc/", True),
        ("https://instagram.com/p/abc/", True),
        ("https://m.facebook.com/photo?fbid=1", True),
        ("https://x.com/user/status/1", True),
        ("https://twitter.com/user/status/1", True),
        ("https://uk.linkedin.com/in/someone", True),
        ("https://t.me/channel/12", True),
        ("https://example.com/blog", False),
        ("https://notinstagram.com/p/abc/", False),
        ("https://instagram.com.evil.example/p/abc/", False),
        ("ftp://instagram.com/p/abc/", False),
        ("not a url", False),
        ("", False),
    ],
)
def test_is_social_url(url, expected):
    assert vs.is_social_url(url) is expected



def test_find_social_match_prefers_the_full_match_page():
    results = vs.parse_web_detection(FAKE_VISION_RESPONSE)
    assert vs.find_social_match(results) == "https://www.instagram.com/p/CxAmPl3/"


def test_find_social_match_record_exposes_the_image_to_download():
    results = vs.parse_web_detection(FAKE_VISION_RESPONSE)
    match = vs.find_social_match_record(results)

    assert match is not None
    assert match.kind == "page_full_match"
    assert match.best_image_url() == "https://scontent.cdninstagram.com/v/t51/full.jpg"


def test_non_social_pages_are_not_selected():
    payload = {
        "webDetection": {
            "pagesWithMatchingImages": [
                {
                    "url": "https://example-blog.com/post",
                    "fullMatchingImages": [{"url": "https://cdn.example.com/a.jpg"}],
                }
            ]
        }
    }
    results = vs.parse_web_detection(payload)
    assert results  # the hit is reported ...
    assert vs.find_social_match(results) is None  # ... but is not a social match


def test_visually_similar_social_urls_are_never_a_match():
    """A look-alike photo on a social profile is not evidence of *this* photo."""
    payload = {
        "webDetection": {
            "visuallySimilarImages": [
                {"url": "https://www.instagram.com/p/L00KALIKE/", "score": 0.99}
            ]
        }
    }
    results = vs.parse_web_detection(payload)

    assert len(results) == 1
    assert results[0].is_social is True
    assert vs.social_candidates(results) == []
    assert vs.find_social_match(results) is None


def test_no_results_means_no_match():
    assert vs.find_social_match([]) is None
    assert vs.find_social_match_record([]) is None


def test_social_domains_can_be_extended_by_env(monkeypatch):
    monkeypatch.setenv("SOCIAL_DOMAINS", "mycommunity.example")
    domains = vs._load_social_domains()

    assert vs.is_social_url("https://forum.mycommunity.example/u/me", domains) is True
    assert vs.is_social_url("https://instagram.com/p/a/", domains) is False



# --------------------------------------------------------------------------- #
# Constraint 1.2 enforcement: the live path never substitutes a canned result
# --------------------------------------------------------------------------- #

def test_search_image_raises_without_credentials(monkeypatch, tmp_path):
    """No credential -> ConfigError. Not an empty list, not a fixture."""
    monkeypatch.delenv("GOOGLE_VISION_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")  # smallest thing that is not empty

    with pytest.raises(ConfigError) as excinfo:
        vs.search_image(image)

    assert "GOOGLE_VISION_API_KEY" in (excinfo.value.hint or "")
    assert "README" in (excinfo.value.hint or "")


def test_search_image_raises_when_the_network_fails(monkeypatch, tmp_path):
    """A broken API call surfaces as an error rather than a made-up match."""
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "test-key-not-real")

    def explode(*args, **kwargs):
        raise vs.requests.ConnectionError("simulated network failure")

    monkeypatch.setattr(vs.requests, "post", explode)

    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")

    with pytest.raises(SearchError, match="simulated network failure"):
        vs.search_image(image)


def test_search_image_actually_posts_to_the_vision_endpoint(monkeypatch, tmp_path):
    """Confirm the live path is a real HTTP POST carrying the image bytes."""
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "test-key-not-real")
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"responses": [{"webDetection": {}}]}

    def capture(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(vs.requests, "post", capture)

    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")
    assert vs.search_image(image) == []

    assert captured["url"].startswith(vs.VISION_ENDPOINT)
    assert "key=test-key-not-real" in captured["url"]
    request = captured["body"]["requests"][0]
    assert request["features"] == [{"type": "WEB_DETECTION", "maxResults": 50}]
    assert request["image"]["content"]  # base64 of the file we just wrote


def test_no_module_in_the_pipeline_path_hardcodes_a_match():
    """Static guard: no shipped module contains a canned social post URL.

    The fixture above lives in this test file only. If a URL like it ever appears
    in the pipeline path, this fails.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    shipped = [
        root / "pipeline.py",
        root / "record.py",
        root / "errors.py",
        *(root / "web_search").glob("*.py"),
        *(root / "face_id").glob("*.py"),
        *(root / "verify").glob("*.py"),
        *(root / "blockchain").glob("*.py"),
    ]
    post_url = re.compile(
        r"https?://(?:www\.|m\.)?"
        r"(?:instagram\.com/p/|twitter\.com/\w+/status/|x\.com/\w+/status/|"
        r"facebook\.com/photo|tiktok\.com/@)",
        re.IGNORECASE,
    )

    offenders = [
        path.name
        for path in shipped
        if post_url.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"hardcoded post URL found in: {offenders}"
