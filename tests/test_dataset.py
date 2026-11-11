import json
from pathlib import Path

from faraim_scraper.dataset import build_chunks, write_corpus, assign_split, count_tokens

LONG_TEXT = "Intro paragraph.\n\n" + "\n\n".join(
    f"(a{i}) Filler paragraph number {i} with enough words to consume tokens steadily."
    for i in range(400)
)


def make_data(tmp_path: Path):
    recs = [
        {"doc": "far", "section_id": "91.113", "part": "91", "chapter": None,
         "title": "§ 91.113 Right-of-way rules: Except water operations.",
         "url": "https://faraim.org/faa/far/cfr/title-14/part-91/section-91.113.html",
         "html_path": "far/title-14/part-91/section-91.113.html",
         "txt_path": "far/title-14/part-91/section-91.113.txt", "images": []},
        {"doc": "far", "section_id": "107.19", "part": "107", "chapter": None,
         "title": "§ 107.19 Crewmember",
         "url": "https://faraim.org/faa/far/cfr/title-14/part-107/section-107.19.html",
         "html_path": "far/title-14/part-107/section-107.19.html",
         "txt_path": "far/title-14/part-107/section-107.19.txt", "images": []},
        {"doc": "aim", "section_id": "section-1-1-9", "part": None, "chapter": "chapter-1",
         "title": "Instrument Landing System (ILS)",
         "url": "https://faraim.org/faa/aim/chapter-1/section-1-1-9.html",
         "html_path": "aim/chapter-1/section-1-1-9.html",
         "txt_path": "aim/chapter-1/section-1-1-9.txt", "images": []},
        {"doc": "aim", "section_id": "appendix-1", "part": None, "chapter": "appendix-1",
         "title": "Appendix 1. Major International Airports",
         "url": "https://faraim.org/faa/aim/appendix-1/appendix-1.html",
         "html_path": "aim/appendix-1/appendix-1.html",
         "txt_path": "aim/appendix-1/appendix-1.txt", "images": []},
        {"doc": "far", "section_id": "91.411", "part": "91", "chapter": None,
         "title": "§ 91.411",
         "url": "https://faraim.org/faa/far/cfr/title-14/part-91/section-91.411.html",
         "html_path": "far/title-14/part-91/section-91.411.html",
         "txt_path": "far/title-14/part-91/section-91.411.txt", "images": []},
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    texts = {
        recs[0]["txt_path"]: "(b) General. Vigilance shall be maintained.",
        recs[1]["txt_path"]: "(a) No person may manipulate the controls of a small unmanned aircraft...",
        recs[2]["txt_path"]: "1-1-9. Instrument Landing System (ILS)\na. General\n1. The ILS provides an approach path.",
        recs[3]["txt_path"]: "Appendix 1 lists major international airports.",
        recs[4]["txt_path"]: LONG_TEXT,
    }
    for rel, text in texts.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return recs


def test_build_chunks_basic(tmp_path):
    make_data(tmp_path)
    chunks = build_chunks(tmp_path)
    ids = [c.id for c in chunks]
    assert "far:91:91.113" in ids
    assert "aim:chapter-1:section-1-1-9" in ids
    by_id = {c.id: c for c in chunks}
    assert by_id["far:91:91.113"].title == "§ 91.113 Right-of-way rules: Except water operations."
    assert by_id["far:91:91.113"].n_tokens == count_tokens(by_id["far:91:91.113"].text)


def test_split_oversized_section(tmp_path):
    make_data(tmp_path)
    chunks = [c for c in build_chunks(tmp_path, max_tokens=300) if c.section_id == "91.411"]
    assert len(chunks) > 1
    assert all(c.n_tokens <= 400 for c in chunks)  # small overflow allowed by merge fallback
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_total == len(chunks) for c in chunks)
    joined = "\n\n".join(c.text for c in chunks)
    assert "Intro paragraph." in joined and "(a399)" in joined


def test_holdout_splits_are_whole_units(tmp_path):
    make_data(tmp_path)
    chunks = build_chunks(tmp_path)
    by_id = {c.id: c for c in chunks}
    assert by_id["far:107:107.19"].split == "test"          # whole Part 107 held out
    assert by_id["far:91:91.113"].split == "train"           # Part 91 trains
    assert by_id["aim:appendix-1:appendix-1"].split == "test"
    assert by_id["aim:chapter-1:section-1-1-9"].split == "train"
    assert assign_split({"doc": "far", "part": "103"}) == "val"
    assert assign_split({"doc": "aim", "chapter": "chapter-3"}) == "test"


def test_write_corpus_parquet(tmp_path):
    make_data(tmp_path)
    chunks = build_chunks(tmp_path)
    counts = write_corpus(chunks, tmp_path / "ds")
    assert sum(counts.values()) == len(chunks)
    import pyarrow.parquet as pq
    train = pq.read_table(tmp_path / "ds" / "corpus" / "train.parquet").to_pylist()
    assert any(r["section_id"] == "91.113" for r in train)
    assert (tmp_path / "ds" / "corpus" / "test.parquet").is_file()
