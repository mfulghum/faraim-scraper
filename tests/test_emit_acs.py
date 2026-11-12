import json
from pathlib import Path

from faraim_scraper.citations import SectionIndex
from faraim_scraper.dataset import Chunk
from faraim_scraper.emit_dataset import (
    build_acs_records,
    build_closed_book_records,
    build_sft_records,
)


def make_data(tmp_path: Path):
    rec = {"doc": "far", "section_id": "91.113", "part": "91", "chapter": None,
           "title": "§ 91.113 Right-of-way rules: Except water operations.",
           "url": "https://faraim.org/faa/far/cfr/title-14/part-91/section-91.113.html",
           "html_path": "far/title-14/part-91/section-91.113.html",
           "txt_path": "far/title-14/part-91/section-91.113.txt", "images": []}
    (tmp_path / "manifest.jsonl").write_text(json.dumps(rec), encoding="utf-8")
    (tmp_path / "far/title-14/part-91").mkdir(parents=True)
    (tmp_path / "far/title-14/part-91/section-91.113.txt").write_text(
        "vigilance shall be maintained", encoding="utf-8")
    # ACS crosswalk reverse index
    (tmp_path / "acs").mkdir()
    (tmp_path / "acs/acs_keys_by_section.jsonl").write_text(json.dumps({
        "html_path": "far/title-14/part-91/section-91.113.html",
        "section_id": "91.113", "doc": "far",
        "acs_keys": ["PA.IX.A.K1", "PA.IX.A.S1"]}), encoding="utf-8")
    return rec


def make_chunks():
    return {
        "far:91:91.113": Chunk(
            id="far:91:91.113", text="vigilance shall be maintained", doc="far",
            part="91", chapter=None, section_id="91.113",
            title="§ 91.113 Right-of-way rules",
            url="https://faraim.org/faa/far/cfr/title-14/part-91/section-91.113.html",
            n_tokens=6, split="train"),
    }


def make_pair_rec():
    return {"chunk_id": "far:91:91.113", "type": "factual", "problems": [],
            "pair": {"question": "Must I keep lookout?",
                     "answer": "Yes per 14 CFR § 91.113.",
                     "quotes": [{"section_id": "91.113",
                                 "quote": "vigilance shall be maintained"}],
                     "citations": ["91.113"]}}


def test_sft_records_carry_acs_keys(tmp_path):
    make_data(tmp_path)
    from faraim_scraper.dataset import load_acs_keys_by_section
    acs_map = load_acs_keys_by_section(tmp_path)
    index = SectionIndex(tmp_path)
    chunks = make_chunks()
    rec = make_pair_rec()
    sft = build_sft_records([rec], chunks, index, acs_map)
    assert sft[0]["acs_keys"] == ["PA.IX.A.K1", "PA.IX.A.S1"]
    closed = build_closed_book_records([rec], chunks, acs_map)
    assert closed[0]["acs_keys"] == ["PA.IX.A.K1", "PA.IX.A.S1"]


def test_no_acs_file_yields_empty_keys(tmp_path):
    (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")
    from faraim_scraper.dataset import load_acs_keys_by_section
    assert load_acs_keys_by_section(tmp_path) == {}


def test_build_acs_records_and_splits(tmp_path):
    acs_dir = tmp_path / "acs"
    acs_dir.mkdir()
    (acs_dir / "acs_elements.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"code": "PA.I.A.K1", "text": "Certification requirements.", "category": "PA",
         "elem_type": "K", "pdf": "private_airplane_acs_6.pdf", "doc_id": "FAA-S-ACS-6C",
         "doc_title": "Private Pilot for Airplane Category",
         "area_title": "Preflight Preparation", "task_title": "Pilot Qualifications"},
        {"code": "IH.I.A.K1", "text": "IFR equipment.", "category": "IH",
         "elem_type": "K", "pdf": "instrument_helicopter_acs_14.pdf",
         "doc_id": "FAA-S-ACS-14", "doc_title": "Instrument Rating Helicopter",
         "area_title": "Preflight Preparation", "task_title": "Equipment"},
    ]), encoding="utf-8")
    (acs_dir / "acs_tasks.jsonl").write_text(json.dumps({
        "code": "PA.I.A", "task_title": "Pilot Qualifications",
        "area_title": "Preflight Preparation",
        "objective": "To determine knowledge...", "refs": ["14 CFR part 61"],
        "pdf": "private_airplane_acs_6.pdf"}), encoding="utf-8")
    records = build_acs_records(acs_dir)
    assert len(records) == 3
    by_id = {r["id"]: r for r in records}
    assert by_id["acs:PA.I.A.K1"]["split"] == "train"
    assert by_id["acs:IH.I.A.K1"]["split"] == "test"      # held-out ACS doc
    assert by_id["acs:PA.I.A (task)"]["text"].startswith("To determine")
    assert "14 CFR part 61" in by_id["acs:PA.I.A (task)"]["text"]
    assert by_id["acs:PA.I.A.K1"]["text"].startswith("Preflight Preparation")
