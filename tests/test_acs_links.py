import json

from faraim_scraper.acs_links import AcsLinker, parse_refs


def test_parse_refs():
    text = ("See 14 CFR § 91.113 and AIM 3-1-4; refs: 14 CFR parts 61, 91; "
            "Part 68 BasicMed; FAA-H-8083-25")
    parsed = parse_refs(text)
    assert parsed["far_sections"] == ["91.113"]
    assert parsed["aim_paras"] == ["3-1-4"]
    assert parsed["parts"] == ["61", "68", "91"]


def make_index(tmp_path):
    recs = [
        {"doc": "far", "section_id": "91.113", "part": "91",
         "title": "§ 91.113 Right-of-way rules: Except water operations.",
         "url": "https://faraim.org/faa/far/cfr/title-14/part-91/section-91.113.html",
         "html_path": "far/title-14/part-91/section-91.113.html",
         "txt_path": "far/title-14/part-91/section-91.113.txt"},
        {"doc": "aim", "section_id": "section-3-1-4", "chapter": "chapter-3",
         "title": "Basic VFR Weather Minimums",
         "url": "https://faraim.org/faa/aim/chapter-3/section-3-1-4.html",
         "html_path": "aim/chapter-3/section-3-1-4.html",
         "txt_path": "aim/chapter-3/section-3-1-4.txt"},
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    (tmp_path / "far/title-14/part-91").mkdir(parents=True)
    (tmp_path / "far/title-14/part-91/section-91.113.txt").write_text(
        "Right of way.", encoding="utf-8")
    (tmp_path / "aim/chapter-3").mkdir(parents=True)
    (tmp_path / "aim/chapter-3/section-3-1-4.txt").write_text("VFR minimums.",
                                                               encoding="utf-8")
    return tmp_path


def test_explicit_citation_links(tmp_path):
    linker = AcsLinker(make_index(tmp_path), semantic=False)
    result = linker.links_for("PA.IX.A.K1",
                              "Under 14 CFR § 91.113 and AIM 3-1-4 see weather minimums",
                              ["14 CFR part 91"])
    ids = {(l["type"], l["section_id"], l["strength"]) for l in result["links"]}
    assert ("far", "91.113", 3) in ids
    assert ("aim", "section-3-1-4", 3) in ids
    assert result["parts"] == ["91"]


def test_reverse_index_shape(tmp_path):
    from faraim_scraper.acs_links import build_links
    acs_dir = tmp_path / "acs"
    acs_dir.mkdir()
    (acs_dir / "acs_elements.jsonl").write_text(json.dumps({
        "code": "PA.IX.A.K1", "text": "Under § 91.113 give way", "elem_type": "K",
        "pdf": "x.pdf", "doc_id": "FAA-S-ACS-6C", "doc_title": "Private"}),
        encoding="utf-8")
    (acs_dir / "acs_tasks.jsonl").write_text("", encoding="utf-8")
    stats = build_links(acs_dir, make_index(tmp_path))
    assert stats["with_strong_links"] == 1
    reverse = [json.loads(l) for l in
               (acs_dir / "acs_keys_by_section.jsonl").read_text().splitlines()]
    assert reverse[0]["section_id"] == "91.113"
    assert reverse[0]["acs_keys"] == ["PA.IX.A.K1"]
