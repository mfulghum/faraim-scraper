"""Site URLs, entry points and scrape defaults."""

DEFAULT_BASE_URL = "https://faraim.org"

# Entry points
FAR_PILOT_INDEX_PATH = "/faa/far/pilot-index.html"
FAR_FULL_INDEX_PATH = "/faa/far/index.html"
AIM_INDEX_PATH = "/faa/aim/index.html"

# Request politeness defaults
DEFAULT_CONCURRENCY = 5
DEFAULT_RATE_LIMIT = 3.0  # requests per second, globally
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 30  # seconds

USER_AGENT = (
    "faraim-scraper/0.1 (research dataset builder; contact: local user) "
    "based on requests"
)

from pathlib import Path

# The packaged dataset ships in its own Hugging Face dataset repo
# (https://huggingface.co/datasets/mfulghum/far-aim), cloned here.
HF_DATASET_DIR = Path.home() / "huggingface" / "far-aim"


def default_dataset_dir() -> Path:
    return HF_DATASET_DIR if HF_DATASET_DIR.is_dir() else Path("dataset")

# Patterns of section hrefs on the site
import re

FAR_SECTION_RE = re.compile(r"^section-[\d.]+\.html$")
FAR_PART_DIR_RE = re.compile(r"(?:^|/)cfr/(title-\d+)/(part-\d+)/index(?:-all)?\.html$")
