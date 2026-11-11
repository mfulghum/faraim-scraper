"""HTTP session with retries, rate limiting and resumable caching."""

import threading
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import DEFAULT_RETRIES, DEFAULT_TIMEOUT, DEFAULT_RATE_LIMIT, USER_AGENT


class RateLimiter:
    """Global minimum-interval limiter shared by all worker threads."""

    def __init__(self, requests_per_second: float):
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                sleep_for = self._next_time - now
                self._next_time += self.min_interval
            else:
                sleep_for = 0.0
                self._next_time = now + self.min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)


class Http:
    """Thin wrapper over requests.Session with retries and rate limiting."""

    def __init__(
        self,
        base_url: str,
        retries: int = DEFAULT_RETRIES,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.limiter = RateLimiter(rate_limit)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        retry = Retry(
            total=retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get(self, url: str) -> requests.Response:
        """GET a URL with rate limiting; raises on final HTTP error."""
        self.limiter.wait()
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp

    def get_text(self, url: str) -> str:
        """GET a URL and decode it, defaulting to UTF-8 when no charset is sent."""
        resp = self.get(url)
        if "charset" not in resp.headers.get("content-type", "").lower():
            resp.encoding = "utf-8"
        return resp.text

    def cached_get(self, url: str, dest: Path, force: bool = False) -> bool:
        """Download url to dest, skipping when a non-empty cache exists.

        Returns True if the file was (re)downloaded, False if it was skipped.
        """
        if not force and dest.is_file() and dest.stat().st_size > 0:
            return False
        resp = self.get(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)
        return True
