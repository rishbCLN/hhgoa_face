"""Unit tests for the Google Gemini search module."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from errors import ConfigError, SearchError
import web_search.gemini_search as gs


def test_credential_summary_none(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert gs.credential_summary() == "none"


def test_credential_summary_masked(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.Ab8RN6JHYuA_-JxkQRAigFLCYgdCMfuFmUvePSIF1WEvwaK1wg")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    summary = gs.credential_summary()
    assert "gemini-3.8-flash" in summary
    assert "AQ.A...K1wg" in summary


def test_parse_gemini_response_valid():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "matches": [
                                        {
                                            "url": "https://example.com/profile/jane",
                                            "title": "Jane Doe Profile",
                                            "kind": "page_full_match",
                                            "confidence": 0.95,
                                            "image_urls": ["https://example.com/avatar.jpg"],
                                        }
                                    ]
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }
    results = gs.parse_gemini_response(payload)
    assert len(results) == 1
    assert results[0].url == "https://example.com/profile/jane"
    assert results[0].page_title == "Jane Doe Profile"
    assert results[0].kind == "page_full_match"
    assert results[0].score == 0.95
    assert results[0].image_urls == ["https://example.com/avatar.jpg"]


def test_parse_gemini_response_markdown_fences():
    inner_json = json.dumps(
        {
            "matches": [
                {
                    "url": "https://example.org/person",
                    "title": "Example Person",
                    "kind": "page_partial_match",
                    "confidence": 0.88,
                }
            ]
        }
    )
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": f"```json\n{inner_json}\n```"}]
                }
            }
        ]
    }
    results = gs.parse_gemini_response(payload)
    assert len(results) == 1
    assert results[0].url == "https://example.org/person"


def test_parse_gemini_response_empty_or_malformed():
    assert gs.parse_gemini_response({}) == []
    assert gs.parse_gemini_response({"candidates": []}) == []
    assert gs.parse_gemini_response({"candidates": [{"content": {"parts": [{"text": "not-json"}]}}]}) == []


def test_search_image_raises_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    img = tmp_path / "test.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    with pytest.raises(ConfigError) as excinfo:
        gs.search_image(img)
    assert "GEMINI_API_KEY" in str(excinfo.value.hint or "")


def test_search_image_raises_missing_file(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    with pytest.raises(SearchError, match="Image not found"):
        gs.search_image("non_existent_file_path_12345.jpg")
