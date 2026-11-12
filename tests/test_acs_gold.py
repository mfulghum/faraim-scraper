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
