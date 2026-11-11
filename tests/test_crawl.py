import json
from pathlib import Path

from faraim_scraper.crawl import Crawler, local_rel_path
from faraim_scraper.discover import Section
from faraim_scraper.manifest import write_manifest, write_scrape_info

BASE = "https://faraim.org"
FIXTURES = Path(__file__).parent / "fixtures"


class FakeHttp:
    """In-memory stand-in for Http, keyed by URL."""

    base_url = BASE

    def __init__(self, pages):
        self.pages = pages
        self.requested = []

    def cached_get(self, url, dest, force=False):
        self.requested.append(url)
        if not force and dest.is_file() and dest.stat().st_size > 0:
            return False
        if url not in self.pages:
            raise RuntimeError(f"404 {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.pages[url])
        return True


def make_http():
    aim_page = (FIXTURES / "aim_section.html").read_text(encoding="utf-8")
    return FakeHttp({
        BASE + "/faa/aim/chapter-1/section-1-1-9.html": aim_page.encode(),
        BASE + "/faa/aim/chapter-1/section-1-1-9.txt": b"1-1-9. ILS\n",
        BASE + "/faa/aim/chapter-1/section-1-1-9-img-1.png": b"\x89PNG fake1",
        BASE + "/faa/aim/chapter-1/section-1-1-9-img-2.png": b"\x89PNG fake2",
    })


def make_section():
    return Section(
        doc="aim",
        url=BASE + "/faa/aim/chapter-1/section-1-1-9.html",
        section_id="section-1-1-9",
        chapter="chapter-1",
        title="Instrument Landing System (ILS)",
    )


def test_local_rel_path():
    assert local_rel_path(make_section()).as_posix() == "aim/chapter-1/section-1-1-9.html"
    far = Section(doc="far", url=BASE + "/faa/far/cfr/title-14/part-91/section-91.113.html",
                  section_id="91.113", part="91")
    assert local_rel_path(far).as_posix() == "far/title-14/part-91/section-91.113.html"


def test_crawl_section_writes_files_and_record(tmp_path):
    http = make_http()
    crawler = Crawler(http, tmp_path)
    [record] = crawler.crawl([make_section()])

    html = (tmp_path / "aim/chapter-1/section-1-1-9.html").read_text()
    assert 'src="section-1-1-9-img-1.png"' in html
    assert (tmp_path / "aim/chapter-1/section-1-1-9-img-1.png").is_file()
    assert (tmp_path / "aim/chapter-1/section-1-1-9.txt").is_file()

    assert record["title"] == "Instrument Landing System (ILS)"
    assert record["html_path"] == "aim/chapter-1/section-1-1-9.html"
    assert record["txt_path"] == "aim/chapter-1/section-1-1-9.txt"
    assert record["images"] == [
        "aim/chapter-1/section-1-1-9-img-1.png",
        "aim/chapter-1/section-1-1-9-img-2.png",
    ]
    assert len(record["sha256"]) == 64


def test_crawl_is_resumable(tmp_path):
    http = make_http()
    Crawler(http, tmp_path).crawl([make_section()])
    n_before = len(http.requested)

    crawler2 = Crawler(http, tmp_path)
    crawler2.crawl([make_section()])
    assert crawler2.stats.downloaded == 0
    assert crawler2.stats.skipped == len(http.requested) - n_before


def test_crawl_missing_txt_keeps_record(tmp_path):
    http = make_http()
    del http.pages[BASE + "/faa/aim/chapter-1/section-1-1-9.txt"]
    crawler = Crawler(http, tmp_path)
    [record] = crawler.crawl([make_section()])
    assert record["txt_path"] is None
    assert not crawler.stats.failed


def test_manifest_files(tmp_path):
    http = make_http()
    crawler = Crawler(http, tmp_path)
    records = crawler.crawl([make_section()])
    manifest = write_manifest(tmp_path, records)
    lines = manifest.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["section_id"] == "section-1-1-9"

    info_path = write_scrape_info(tmp_path, BASE, records, [], aim_source="AIM pdf")
    info = json.loads(info_path.read_text())
    assert info["documents"]["aim"]["sections"] == 1
    assert info["documents"]["aim"]["images"] == 2
    assert info["aim_source"] == "AIM pdf"
