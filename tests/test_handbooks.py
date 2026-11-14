import json
from pathlib import Path

import pymupdf
import pytest

from faraim_scraper.handbooks import (
    _detect_chapters,
    _page1_doc_id,
    parse_handbook,
    select_pdfs,
)


def _make_pdf(path: Path, pages: list, first_lines=None):
    doc = pymupdf.open()
    for i, text in enumerate(pages):
        page = doc.new_page()
        y = 60
        for line in text.splitlines():
            page.insert_text((50, y), line, fontsize=9)
            y += 14
    doc.save(str(path))
    doc.close()
    return path


def test_detect_chapters_page_labels(tmp_path):
    pages = ["cover"] * 12
    for ch in range(1, 4):
        pages.append(f"{ch}-1\nChapter body start for chapter {ch}")
        pages += [f"{ch}-{i} body text continues here" for i in range(2, 6)]
    pdf = _make_pdf(tmp_path / "bk.pdf", pages)
    doc = pymupdf.open(str(pdf))
    chapters = _detect_chapters(doc)
    assert chapters is not None
    assert [c["title"] for c in chapters] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    # figure labels like '2-1' mid-chapter must not create boundaries
    pages2 = ["cover"] * 12 + [f"body {i}" for i in range(30)]
    pages2[20] = "2-1\nthis is a figure caption, not a chapter start"
    pdf2 = _make_pdf(tmp_path / "bk2.pdf", pages2)
    doc2 = pymupdf.open(str(pdf2))
    assert _detect_chapters(doc2) is None


def test_parse_handbook_chunks(tmp_path):
    pages = ["cover"] * 12
    for ch in (1, 2):
        pages.append(f"{ch}-1\nChapter start")
        pages += [f"{ch}-{i} " + "filler words for the chapter body. " * 30
                  for i in range(2, 8)]
    pdf = _make_pdf(tmp_path / "faa-h-8083-99.pdf", pages)
    chunks = parse_handbook(pdf, max_tokens=200)
    assert chunks
    assert all(c["handbook"] == "FAA-H-8083-99" for c in chunks)
    assert {c["chapter"] for c in chunks} == {1, 2}
    assert all(c["split"] == "train" for c in chunks)


def test_page1_doc_id_prefers_filename(tmp_path):
    pdf = _make_pdf(tmp_path / "faa-h-8083-25c.pdf",
                    ["random text citing FAA-H-8083-2 elsewhere"])
    doc = pymupdf.open(str(pdf))
    assert _page1_doc_id(pdf, doc) == "FAA-H-8083-25C"
    doc.close()
    # no filename code, no prominent page-1 code -> stem
    pdf2 = _make_pdf(tmp_path / "some_manual.pdf", ["nothing relevant here"])
    doc2 = pymupdf.open(str(pdf2))
    assert _page1_doc_id(pdf2, doc2) == "some_manual"
    doc2.close()


def test_select_pdfs_keeps_full_over_fragments(tmp_path):
    def touch(name):
        p = tmp_path / name
        p.write_bytes(b"%PDF-1.4 fake")
        return p
    files = [touch("helicopter_flying_handbook.pdf"), touch("faa-h-8083-21.pdf"),
             touch("hfh_ch01.pdf"), touch("hfh_front.pdf"),
             touch("unrelated_guide.pdf")]
    # select_pdfs opens real PDFs; our fakes fail to open -> treated as own family
    selected = select_pdfs(files)
    assert tmp_path / "unrelated_guide.pdf" in selected


def test_select_pdfs_real_pdfs_dedupes(tmp_path):
    # one full volume + a chapter fragment of the same family
    full = _make_pdf(tmp_path / "helicopter_flying_handbook.pdf",
                     ["cover"] * 20)
    frag = _make_pdf(tmp_path / "hfh_ch03.pdf", ["cover"] * 5)
    selected = select_pdfs([full, frag])
    assert selected == [full]


def test_curated_title():
    from faraim_scraper.handbooks import HB_EXCLUDE, curated_title
    # the cover-page extractor used to call 8083-21 'Flight Standards Service'
    assert curated_title('FAA-H-8083-21') == 'Rotorcraft Flying Handbook'
    assert curated_title('FAA-H-8083-25C') == \
        "Pilot's Handbook of Aeronautical Knowledge"
    assert curated_title('FAA-H-8083-19A') == \
        'Plane Sense: General Aviation Information'
    assert curated_title('bfh_chapter_3') == 'Balloon Flying Handbook'
    assert curated_title('03_aih_chapter_1') == "Aviation Instructor's Handbook"
    assert curated_title('some_unknown_doc') is None
    assert '00-80T-80' in HB_EXCLUDE


def test_build_handbook_chunks_skips_excluded(tmp_path):
    from faraim_scraper.dataset import build_handbook_chunks
    hb = tmp_path / 'handbooks'
    hb.mkdir()
    rec = {"id": "hbk:00-80T-80:ch1", "handbook": "00-80T-80",
           "handbook_title": "x", "chapter": 1, "chapter_title": None,
           "pages": "1-2", "text": "supply requisitioning junk here",
           "n_tokens": 10}
    good = dict(rec, id="hbk:FAA-H-8083-21:ch1", handbook="FAA-H-8083-21",
                handbook_title="Rotorcraft Flying Handbook",
                text="rotorcraft airfoil lift and drag fundamentals " * 4)
    (hb / 'handbook_chunks.jsonl').write_text(
        "\n".join(json.dumps(r) for r in (rec, good)), encoding='utf-8')
    chunks = build_handbook_chunks(tmp_path)
    assert [c.id for c in chunks] == ["hbk:FAA-H-8083-21:ch1"]
