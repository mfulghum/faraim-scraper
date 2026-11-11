# faraim-scraper

Scraper for the **FAR** (Federal Aviation Regulations) and **AIM** (Aeronautical
Information Manual) documents published on [faraim.org](https://faraim.org).
It mirrors the section HTML pages, their clean `.txt` versions, and all
attached images into a local tree, plus a `manifest.jsonl` designed to drive
Hugging Face dataset creation.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
```

## Usage

```bash
# Full FAR pilot index (84 ASA-style parts) + AIM, into data/
python -m faraim_scraper

# Just the AIM
python -m faraim_scraper --documents aim

# Complete FAR index (all parts, not just the pilot ones)
python -m faraim_scraper --full-far

# Try it out on a few sections first
python -m faraim_scraper --limit 3
```

Common options:

| Flag | Default | Meaning |
|---|---|---|
| `--out DIR` | `data` | output directory (git-ignored) |
| `--documents far,aim` | `far,aim` | which documents to scrape |
| `--full-far` | off | use the full FAR index instead of the pilot index |
| `--no-txt` | off | skip the `.txt` version of each section |
| `--concurrency N` | 5 | parallel downloads |
| `--rate-limit N` | 3.0 | max requests/second (0 disables) — please stay polite |
| `--retries N` | 3 | retries per request with exponential backoff |
| `--limit N` | – | stop after N sections per document (testing) |
| `--force` | off | re-download files even if cached |

Runs are **resumable**: existing files are skipped unless `--force` is given.
Re-running after an interruption only fetches what is missing.

## Output layout

```
data/
├── far/title-14/part-91/
│   ├── index files skipped (we store sections)
│   ├── section-91.113.html      # localized HTML (img src rewritten to local files)
│   ├── section-91.113.txt       # clean plain text
│   └── section-91.113-img-1.png # images stored beside their page
├── aim/chapter-1/
│   ├── section-1-1-9.html
│   ├── section-1-1-9.txt
│   └── section-1-1-9-img-1.png
├── aim/appendix-1/appendix-1.{html,txt}
├── aim/pilot-controller-glossary/glossary.{html,txt}
├── manifest.jsonl               # one JSON record per section
└── scrape_info.json             # provenance: date, source PDF, counts, failures
```

Each `manifest.jsonl` record:

```json
{"doc": "aim", "section_id": "section-1-1-9", "part": null, "chapter": "chapter-1",
 "title": "Instrument Landing System (ILS)", "url": "https://faraim.org/...",
 "html_path": "aim/chapter-1/section-1-1-9.html",
 "txt_path": "aim/chapter-1/section-1-1-9.txt",
 "images": ["aim/chapter-1/section-1-1-9-img-1.png"], "sha256": "..."}
```

## Building a Hugging Face dataset

The manifest is dataset-ready: load it with `datasets.load_dataset("json", ...)`
and join in the text by reading `txt_path` (preferred for embedding/chunking —
it's the site's own clean text extraction). `sha256` lets you detect content
changes between scrapes, and `scrape_info.json` gives you citation metadata
for the dataset card (the AIM index page records the source PDF name/date).

## Tests

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

Tests run fully offline against fixture pages in `tests/fixtures/`.

## Politeness notes

- Default ~3 requests/second globally, 5 parallel workers, proper retries with
  backoff, and a descriptive User-Agent.
- Only same-host `<img>` assets are fetched; CSS, fonts and analytics scripts
  are left alone.
- The full pilot-FAR + AIM run is ~5,000 pages and ~12,000 requests. Please do
  not hammer the site with high `--rate-limit` values.

## Provenance

All content originates from faraim.org (which itself mirrors public FAA
publications). Keep `scrape_info.json` alongside any published dataset and
credit the source.
