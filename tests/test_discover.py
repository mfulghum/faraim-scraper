from pathlib import Path

from faraim_scraper.discover import (
    parse_aim_index,
    parse_aim_source,
    parse_far_part_index,
    parse_far_part_links,
)

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "https://faraim.org"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_far_part_links():
    parts = parse_far_part_links(
        read("far_pilot_index.html"), BASE + "/faa/far/pilot-index.html"
    )
    assert [(t, p) for t, p, _ in parts] == [
        ("title-14", "part-1"),
        ("title-14", "part-91"),
        ("title-49", "part-830"),
    ]
    assert parts[1][2] == BASE + "/faa/far/cfr/title-14/part-91/index-all.html"


def test_parse_far_part_index_dedupes_sections():
    urls = parse_far_part_index(
        read("far_part_index_all.html"),
        BASE + "/faa/far/cfr/title-14/part-91/index-all.html",
    )
    assert urls == [
        BASE + "/faa/far/cfr/title-14/part-91/section-91.101.html",
        BASE + "/faa/far/cfr/title-14/part-91/section-91.103.html",
        BASE + "/faa/far/cfr/title-14/part-91/section-91.113.html",
    ]


def test_parse_aim_index_sections_and_titles():
    rows = parse_aim_index(read("aim_index.html"))
    ids = [(chapter, sid) for href, chapter, sid, _ in rows]
    assert ids == [
        ("chapter-1", "section-1-1-9"),
        ("chapter-1", "section-1-1-10"),
        ("appendix-1", "appendix-1"),
        ("pilot-controller-glossary", "glossary"),
    ]
    titles = {sid: title for _, _, sid, title in rows}
    assert titles["section-1-1-9"] == "Instrument Landing System (ILS)"
    assert titles["appendix-1"] == "Appendix 1. Major International Airports"
    assert titles["glossary"] == "Pilot/Controller Glossary"


def test_parse_aim_index_ignores_format_duplicates():
    # The HTML/TXT/PDF format links repeat the .html href; it must appear once.
    hrefs = [href for href, _, _, _ in parse_aim_index(read("aim_index.html"))]
    assert hrefs.count("chapter-1/section-1-1-9.html") == 1


def test_parse_aim_source():
    source = parse_aim_source(read("aim_index.html"))
    assert source is not None
    assert "AIM_Basic_w_Chg_1_dtd_7-9-26_FINAL.pdf" in source
