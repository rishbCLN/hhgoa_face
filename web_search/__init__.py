"""Stage 2: live reverse-image search (Google Cloud Vision, Web Detection)."""

from web_search.vision_search import (  # noqa: F401
    SOCIAL_DOMAINS,
    SearchResult,
    credential_summary,
    find_social_match,
    find_social_match_record,
    is_social_url,
    parse_web_detection,
    search_image,
)

__all__ = [
    "SOCIAL_DOMAINS",
    "SearchResult",
    "credential_summary",
    "find_social_match",
    "find_social_match_record",
    "is_social_url",
    "parse_web_detection",
    "search_image",
]
