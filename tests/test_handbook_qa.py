import json

from faraim_scraper.citations import (
    SectionIndex, canonical_citation, canonical_hb_citation, citation_variants)
from faraim_scraper.dataset import Chunk, build_handbook_chunks


def _write_handbooks(data_dir, chunks):
    hb_dir = data_dir / "handbooks"
    hb_dir.mkdir(parents=True, exist_ok=True)
    with (hb_dir / "handbook_chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")


def _handbook_chunk(pid="FAA-H-8083-31B", chapter=3, title="Airframe Systems",
                    pages="3-1-3-30"):
    return {"id": f"hbk:{pid}:ch{chapter}", "handbook": pid,
            "handbook_title": "Technician Handbook—Airframe",
            "chapter": chapter, "chapter_title": title, "pages": pages,
            "text": "The alternate air door opens automatically when the filter "
                    "becomes restricted. Preflight inspection must include the "
                    "static wick condition.", "n_tokens": 30}


def _index(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rec = {"doc": "far", "section_id": "43.13", "title": "§ 43.13 Performance rules",
           "txt_path": "far/43.13.txt", "url": "https://faraim.org/faa/far/title-14/"}
    (data_dir / "manifest.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (data_dir / "far").mkdir()
    (data_dir / "far" / "43.13.txt").write_text(
        "Each person maintaining an aircraft shall use techniques and materials "
        "specified by the manufacturer.", encoding="utf-8")
    _write_handbooks(data_dir, [_handbook_chunk()])
    return SectionIndex(data_dir)


def test_canonical_hb_citation(tmp_path):
    c = _handbook_chunk()
    cit = canonical_hb_citation(c)
    assert cit.startswith("FAA-H-8083-31B — Technician Handbook—Airframe: "
                          "Chapter 3, Airframe Systems")
    assert "pp. 3-1-3-30" in cit
    # via the record dispatcher
    assert canonical_citation({"doc": "handbook", "chunk": c}) == cit


def test_index_resolves_handbook_chunks(tmp_path):
    index = _index(tmp_path)
    rec = index.get("hbk:FAA-H-8083-31B:ch3")
    assert rec is not None and rec["doc"] == "handbook"
    assert "alternate air door" in index.text(rec)
    variants = citation_variants(rec)
    assert variants[0] == canonical_hb_citation(_handbook_chunk())
    assert "FAA-H-8083-31B, Chapter 3" in variants


def test_validate_pair_handbook_verbatim(tmp_path):
    index = _index(tmp_path)
    pair = {
        "question": "When does the alternate air door open on its own?",
        "answer": ("According to FAA-H-8083-31B — Technician Handbook—Airframe: "
                   "Chapter 3, Airframe Systems (pp. 3-1-3-30), "
                   "\"the alternate air door opens automatically when the filter "
                   "becomes restricted.\""),
        "quotes": [{"section_id": "hbk:FAA-H-8083-31B:ch3",
                    "quote": "the alternate air door opens automatically when "
                             "the filter becomes restricted"}],
        "citations": ["hbk:FAA-H-8083-31B:ch3"],
    }
    assert index.validate_pair(pair) == []
    # a fabricated quote must fail
    bad = json.loads(json.dumps(pair))
    bad["quotes"][0]["quote"] = "the alternate air door never opens on its own"
    assert any("not verbatim" in p for p in index.validate_pair(bad))
    # missing citation string in answer must fail
    bad2 = json.loads(json.dumps(pair))
    bad2["answer"] = "It opens when the filter gets restricted."
    assert any("canonical citation" in p for p in index.validate_pair(bad2))


def test_build_handbook_chunks_shape(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_handbooks(data_dir, [_handbook_chunk(),
                                _handbook_chunk(chapter=4, title="Structures")])
    chunks = build_handbook_chunks(data_dir)
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) and c.doc == "handbook" for c in chunks)
    assert chunks[0].part == "FAA-H-8083-31B"
    assert chunks[0].section_id == chunks[0].id == "hbk:FAA-H-8083-31B:ch3"
    assert all(c.split == "train" for c in chunks)


def test_build_handbook_chunks_missing_file(tmp_path):
    assert build_handbook_chunks(tmp_path) == []
