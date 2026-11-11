"""Rewrite image references in saved HTML to local relative paths."""

from typing import List, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


def _is_same_host(base_url: str, url: str) -> bool:
    return urlparse(url).netloc == urlparse(base_url).netloc


def extract_images(html: str, page_url: str, base_url: str) -> List[str]:
    """Return absolute URLs of same-host images referenced by the page."""
    urls = []
    seen = set()
    for img in BeautifulSoup(html, "lxml").find_all("img", src=True):
        url = urljoin(page_url, img["src"])
        if url not in seen and _is_same_host(base_url, url):
            seen.add(url)
            urls.append(url)
    return urls


def localize_images(html: str, page_url: str, base_url: str) -> Tuple[str, List[str]]:
    """Rewrite same-host ``<img src>`` to bare filenames (same-dir mirror).

    Images are stored beside the HTML file under their URL basename, so the
    rewritten src is just the basename. Returns the rewritten HTML and the
    list of image URLs that still need downloading.
    """
    soup = BeautifulSoup(html, "lxml")
    urls = []
    seen = set()
    for img in soup.find_all("img", src=True):
        url = urljoin(page_url, img["src"])
        if not _is_same_host(base_url, url):
            continue
        name = urlparse(url).path.rsplit("/", 1)[-1]
        img["src"] = name
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return str(soup), urls
