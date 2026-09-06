"""Stage 2: live reverse-image search (Google Gemini API or Google Cloud Vision)."""

from web_search import gemini_search, vision_search
from web_search.vision_search import (  # noqa: F401
    DEFAULT_SOCIAL_DOMAINS,
    MATCH_PRIORITY,
    SOCIAL_DOMAINS,
    SearchResult,
    credential_summary,
    find_social_match,
    find_social_match_record,
    is_social_url,
    parse_web_detection,
    search_image,
    social_candidates,
)

__all__ = [
    "DEFAULT_SOCIAL_DOMAINS",
    "MATCH_PRIORITY",
    "SOCIAL_DOMAINS",
    "SearchResult",
    "credential_summary",
    "find_social_match",
    "find_social_match_record",
    "is_social_url",
    "parse_web_detection",
    "search_image",
    "social_candidates",
    "gemini_search",
    "vision_search",
]
