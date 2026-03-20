"""Fetch text content from URLs and extract readable text from HTML."""

from __future__ import annotations

import re
import urllib.request
import urllib.error
import urllib.parse


def fetch_url_raw(url: str, timeout: int = 30) -> tuple[str, str]:
    """Fetch a URL and return (raw_html, extracted_text).

    Useful when the caller needs the raw HTML (e.g. for link extraction)
    AND the cleaned text for training.
    """
    req = urllib.request.Request(url, headers={
        "User-Agent": "gdf/0.1 (text-fetcher)",
        "Accept": "text/html,text/plain,*/*",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw_bytes = resp.read()

    # Detect encoding
    encoding = "utf-8"
    if "charset=" in content_type:
        encoding = content_type.split("charset=")[-1].split(";")[0].strip()

    raw_html = raw_bytes.decode(encoding, errors="replace")

    # If HTML, extract text
    is_html = ("html" in content_type.lower()
               or raw_html.strip().startswith("<!")
               or "<html" in raw_html[:500].lower())
    if is_html:
        text = _clean_text(html_to_text(raw_html))
    else:
        text = _clean_text(raw_html)

    return raw_html, text


def fetch_url(url: str, timeout: int = 30) -> str:
    """Fetch a URL and return extracted text content."""
    _, text = fetch_url_raw(url, timeout=timeout)
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


# Extensions to skip when extracting links (non-content resources)
_SKIP_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".mp3", ".mp4", ".avi", ".mkv", ".wav", ".ogg",
    ".exe", ".msi", ".dmg", ".deb", ".rpm",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
}


def extract_links(html: str, base_url: str, same_domain_only: bool = True) -> list[str]:
    """Extract links from HTML.

    Args:
        html: Raw HTML string.
        base_url: The URL of the page (used to resolve relative links).
        same_domain_only: If True (default), only return links on the same domain.
            If False, return all http/https links.

    Returns:
        Deduplicated list of absolute URLs.
    """
    parsed_base = urllib.parse.urlparse(base_url)
    base_domain = parsed_base.netloc.lower()

    # Find all href attributes
    hrefs = re.findall(r'<a\s[^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE)

    seen: set[str] = set()
    links: list[str] = []

    for href in hrefs:
        # Skip anchors, javascript, mailto
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        # Resolve relative URLs
        absolute = urllib.parse.urljoin(base_url, href)

        # Strip fragment
        absolute = urllib.parse.urldefrag(absolute)[0]

        parsed = urllib.parse.urlparse(absolute)

        # Must be http/https
        if parsed.scheme not in ("http", "https"):
            continue

        # Same domain filter
        if same_domain_only and parsed.netloc.lower() != base_domain:
            continue

        # Skip non-content extensions
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
            continue

        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)

    return links


def is_url(s: str) -> bool:
    """Check if a string looks like a URL."""
    return s.startswith(("http://", "https://", "www."))
