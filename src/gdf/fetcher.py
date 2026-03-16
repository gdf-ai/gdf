"""Fetch text content from URLs and extract readable text from HTML."""

from __future__ import annotations

import re
import urllib.request
import urllib.error


def fetch_url(url: str, timeout: int = 30) -> str:
    """Fetch a URL and return extracted text content."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "gdf/0.1 (text-fetcher)",
        "Accept": "text/html,text/plain,*/*",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read()

    # Detect encoding
    encoding = "utf-8"
    if "charset=" in content_type:
        encoding = content_type.split("charset=")[-1].split(";")[0].strip()

    text = raw.decode(encoding, errors="replace")

    # If HTML, extract text
    if "html" in content_type.lower() or text.strip().startswith("<!") or "<html" in text[:500].lower():
        text = html_to_text(text)

    # Clean up
    text = _clean_text(text)
    return text


def html_to_text(html: str) -> str:
    """Strip HTML to plain text."""
    # Remove script/style blocks
    html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<header[^>]*>.*?</header>', ' ', html, flags=re.DOTALL | re.IGNORECASE)

    # Convert common block elements to newlines
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</(p|div|h[1-6]|li|tr|blockquote|article|section)>', '\n', html, flags=re.IGNORECASE)

    # Remove all remaining tags
    html = re.sub(r'<[^>]+>', ' ', html)

    # Decode common HTML entities
    html = html.replace('&nbsp;', ' ')
    html = html.replace('&amp;', '&')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&quot;', '"')
    html = html.replace('&#39;', "'")
    html = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), html)
    html = re.sub(r'&\w+;', ' ', html)

    return html


def _clean_text(text: str) -> str:
    """Normalize whitespace, remove junk lines."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that are mostly non-alphanumeric (nav junk, symbols)
        alpha_count = sum(1 for c in line if c.isalpha())
        if len(line) > 5 and alpha_count < len(line) * 0.3:
            continue
        # Skip very short lines (likely UI elements)
        if len(line) < 3:
            continue
        cleaned.append(line)

    return '\n'.join(cleaned)


def is_url(s: str) -> bool:
    """Check if a string looks like a URL."""
    return s.startswith(("http://", "https://", "www."))
