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

## ACS documents (faa.gov)

The FAA Airman Certification Standards are PDFs, not HTML — a dedicated module
scrapes and structures them, so training data can be tied to ACS test keys
(e.g. `PA.II.A.K4`):

```bash
python -m faraim_scraper.acs          # download + parse all ACS PDFs -> data/acs/
python -m faraim_scraper.acs_links    # crosswalk ACS codes <-> FAR/AIM sections
```

Outputs in `data/acs/`:

- `pdf/` — all ACS PDFs (faa.gov requires a browser User-Agent)
- `acs_elements.jsonl` — every Knowledge/Risk/Skill element with its official
  code, area/task titles, doc id (FAA-S-ACS-x), and text
- `acs_tasks.jsonl` — task objectives + Reference lines (14 CFR parts, ACs,
  FAA-H-8083-x handbooks, POH/AFM)
- `acs_links.jsonl` — for each ACS code, linked FAR/AIM sections with match
  evidence and strength (`3` = explicit citation, `2` = TF-IDF semantic,
  including CFR-part boosts)
- `acs_keys_by_section.jsonl` — reverse index: FAR/AIM section → relevant ACS codes

## FAA handbooks (PHAK, AFH, AMT, weather…)

```bash
python -m faraim_scraper.handbooks    # download + parse -> data/handbooks/
```

Scrapes the faa.gov aviation handbooks catalog (index + subpages), keeps one
edition per handbook (full volumes preferred over chapter/front-matter splits),
and chunks each book by chapter with page provenance → `handbook_chunks.jsonl`.
~60 books / ~5,500 chunks, including every FAA-H-8083-x the ACSs reference.
The ACS crosswalk resolves handbook References to specific handbook chunks.
Handbook chunks also feed QA synthesis: `python -m faraim_scraper.synthesize
--handbooks --only hbk: --out work/qa_handbooks.jsonl` generates the same four
validated pair types for every handbook (pilot and mechanic volumes alike).

## ACS -> governing-text links (teacher-in-the-loop)

```bash
python -m faraim_scraper.acs_link_teacher            # plan/link/verify per task group
python -m faraim_scraper.acs_link_teacher --merge    # fold into data/acs/acs_links.jsonl
```

No cosine ranking for scoped records: for every ACS task group the reader
model (1) writes the expected **answer key** from the table of contents of the
documents named in the task's References line, (2) loads those passages in
full and emits per-code links with **verbatim-quoted** evidence, (3) an
inspector pass marks every (code, requirement) covered/uncovered, and (4)
requirements that NAME a section (e.g. `14 CFR 91.205(c)`) are enforced
mechanically — the quote is demanded from that exact section, falling back to
the named paragraph's own text (`llm gap-fill (auto-quote)`). Checkpointed in
`work/acs_links_llm.jsonl`, resumable. TF-IDF remains only as the fallback
for records whose References name nothing we scraped.

## Summary tree (semantic navigation)

```bash
python -m faraim_scraper.tree_summaries   # -> work/summaries.jsonl
```

Hierarchical catalog: part/chapter -> section -> subsection for FAR, AIM and
every handbook, summarized bottom-up by the teacher (parents condense their
children's summaries). The linker's TOC prompts show these catalog blurbs, so
search descends general -> specific before loading full text. Ships as the
`summaries` dataset config.

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
