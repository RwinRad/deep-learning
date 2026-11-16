"""Lightweight, tweet-aware text normalization and tokenization."""

from __future__ import annotations

import html
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_USER_RE = re.compile(r"(?<!\w)@\w+")
_TOKEN_RE = re.compile(r"<url>|<user>|#[\w]+|[\w]+|[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_tweet(text: str) -> str:
    """Normalize a tweet without discarding potentially useful punctuation or hashtags."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = html.unescape(text).lower()
    text = _URL_RE.sub(" <url> ", text)
    text = _USER_RE.sub(" <user> ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text, preserving URL/user markers and hashtags."""
    return _TOKEN_RE.findall(normalize_tweet(text))
