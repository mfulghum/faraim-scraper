import json

from faraim_scraper.emit_dataset import build_acs_gold


def _fixture(tmp_path):
    acs_dir = tmp_path / "acs"
    acs_dir.mkdir()
    (acs_dir / "acs_elements.jsonl").write_text(json.dumps({
        "code": "IH.I.A.K1", "text": "Certification requirements, recency of "
                                     "experience, and recordkeeping.",
        "elem_type": "K", "pdf": "instrument_helicopter_acs_14.pdf",
        "doc_id": "FAA-S-ACS-14"}), encoding="utf-8")
    (acs_dir / "acs_tasks.jsonl").write_text("", encoding="utf-8")
    llm = tmp_path / "acs_links_llm.jsonl"
    llm.write_text("\n".join(json.dumps({
        "group": "g", "links": [
            {"code": "IH.I.A.K1", "type": "far", "section_id": "61.57",
             "html_path": "far/61.57.html", "quote": "six instrument approaches",
             "evidence": "llm gap-fill (auto-quote)"},
            {"code": "IH.I.A.K1", "type": "far", "section_id": "61.51",
             "html_path": "far/61.51.html", "quote": "document and record",
             "evidence": "llm full-text"},
            {"code": "IH.I.A.K1", "type": "far", "section_id": "61.57",
             "html_path": "far/61.57.html", "quote": "dupe",
             "evidence": "llm full-text"},
            {"code": "ORPHAN.K1", "type": "far", "section_id": "91.1",
             "html_path": "far/91.1.html", "quote": "x",
             "evidence": "llm full-text"}]})
        for _ in range(2)), encoding="utf-8")
    return llm, acs_dir


from pathlib import Path

import pytest

from faraim_scraper.eval_gold import score_acs_gold
from faraim_scraper.citations import SectionIndex


@pytest.fixture
def index(tmp_path):
    rec = {"doc": "far", "section_id": "61.57", "title": "§ 61.57 Recent flight "
           "experience: Pilot in command.", "url": "u",
           "txt_path": "far/61.57.txt"}
    (tmp_path / "manifest.jsonl").write_text(json.dumps(rec) + "\n",
                                             encoding="utf-8")
    (tmp_path / "far").mkdir()
    (tmp_path / "far" / "61.57.txt").write_text("recent experience text",
                                                encoding="utf-8")
    return SectionIndex(tmp_path)


def test_score_acs_gold(index):
    item = {"code": "IH.I.A.K1", "acs_text": "recency of experience",
            "gold_sections": [{"section_id": "61.57", "type": "far",
                               "quote": "q", "evidence": "llm full-text"},
                              {"section_id": "61.51", "type": "far",
                               "quote": "q2", "evidence": "llm full-text"}]}
    hit, cov = score_acs_gold(item, "Per 14 CFR § 61.57 you need...", index)
    assert hit and abs(cov - 0.5) < 1e-9
    hit, cov = score_acs_gold(item, "no citations here", index)
    assert not hit and cov == 0.0


def test_build_acs_gold(tmp_path):
    llm, acs_dir = _fixture(tmp_path)
    rows = build_acs_gold(llm, acs_dir)
    assert len(rows) == 1                       # orphan code excluded
    row = rows[0]
    assert row["code"] == "IH.I.A.K1"
    assert row["split"] == "test"               # ACS-14 is a held-out test doc
    assert row["acs_text"].startswith("Certification requirements")
    # deduped by html_path: 61.57 once, 61.51 once
    assert [g["section_id"] for g in row["gold_sections"]] == ["61.57", "61.51"]
    assert row["gold_sections"][0]["evidence"] == "llm gap-fill (auto-quote)"


def test_build_acs_gold_missing_file(tmp_path):
    assert build_acs_gold(tmp_path / "none.jsonl", tmp_path) == []
