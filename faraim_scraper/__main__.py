"""Command line entry point for the faraim.org scraper."""

import argparse
import sys
from pathlib import Path

from .config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_RATE_LIMIT,
    DEFAULT_RETRIES,
    DEFAULT_BASE_URL,
    FAR_FULL_INDEX_PATH,
    FAR_PILOT_INDEX_PATH,
)
from .crawl import Crawler
from .discover import discover_aim, discover_far
from .http import Http
from .manifest import write_manifest, write_scrape_info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="faraim-scraper",
        description="Scrape FAR and AIM HTML pages, text and images from faraim.org.",
    )
    p.add_argument("--out", type=Path, default=Path("data"),
                   help="output directory (default: data/)")
    p.add_argument("--documents", default="far,aim",
                   help="comma separated list: far,aim (default: far,aim)")
    p.add_argument("--full-far", action="store_true",
                   help="use the full FAR index instead of the 84-part pilot index")
    p.add_argument("--no-txt", action="store_true",
                   help="skip the .txt version of each section")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help=f"parallel downloads (default: {DEFAULT_CONCURRENCY})")
    p.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT,
                   help=f"max requests per second, 0 disables (default: {DEFAULT_RATE_LIMIT})")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                   help=f"retries per request (default: {DEFAULT_RETRIES})")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="stop after N sections per document (for testing)")
    p.add_argument("--force", action="store_true",
                   help="re-download files even if already cached")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"site base URL (default: {DEFAULT_BASE_URL})")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    docs = [d.strip() for d in args.documents.split(",") if d.strip()]
    unknown = set(docs) - {"far", "aim"}
    if unknown:
        print(f"error: unknown document(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    http = Http(args.base_url, retries=args.retries, rate_limit=args.rate_limit)
    crawler = Crawler(http, args.out, include_txt=not args.no_txt, force=args.force)

    all_sections = []
    far_index = None
    aim_source = None
    if "far" in docs:
        far_index = FAR_FULL_INDEX_PATH if args.full_far else FAR_PILOT_INDEX_PATH
        sections = discover_far(http, full=args.full_far)
        print(f"far: discovered {len(sections)} sections via {far_index}")
        all_sections.extend(sections[: args.limit] if args.limit else sections)
    if "aim" in docs:
        sections, aim_source = discover_aim(http)
        print(f"aim: discovered {len(sections)} sections"
              + (f" (source: {aim_source})" if aim_source else ""))
        all_sections.extend(sections[: args.limit] if args.limit else sections)

    if not all_sections:
        print("error: nothing to scrape", file=sys.stderr)
        return 2

    records = crawler.crawl(all_sections, concurrency=args.concurrency)
    manifest_path = write_manifest(args.out, records)
    write_scrape_info(
        args.out, http.base_url, records, crawler.stats.failed,
        far_index=far_index, aim_source=aim_source,
    )

    n_images = sum(len(r["images"]) for r in records)
    print(f"\ndownloaded: {crawler.stats.downloaded} files, "
          f"skipped (cached): {crawler.stats.skipped}, "
          f"bytes: {crawler.stats.bytes:,}")
    print(f"sections: {len(records)}, images: {n_images}")
    if crawler.stats.failed:
        print(f"failed: {len(crawler.stats.failed)} (see scrape_info.json)")
        for line in crawler.stats.failed[:10]:
            print(f"  {line}")
    print(f"manifest: {manifest_path}")
    return 1 if crawler.stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
