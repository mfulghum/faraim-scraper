import json

from faraim_scraper.citations import (
    SectionIndex,
    canonical_citation,
    citation_variants,
    normalize,
)


def make_index(tmp_path):
    recs = [
        {"doc": "far", "section_id": "91.113", "part": "91",
         "title": "§ 91.113 Right-of-way rules: Except water operations.",
         "url": "https://faraim.org/faa/far/cfr/title-14/part-91/section-91.113.html",
         "txt_path": "far/title-14/part-91/section-91.113.txt"},
        {"doc": "far", "section_id": "1544.3", "part": "1544",
         "title": "§ 1544.3 Requirements for security programs.",
         "url": "https://faraim.org/faa/far/cfr/title-49/part-1544/section-1544.3.html",
         "txt_path": "far/title-49/part-1544/section-1544.3.txt"},
        {"doc": "aim", "section_id": "section-1-1-9", "chapter": "chapter-1",
         "title": "Instrument Landing System (ILS)",
         "url": "https://faraim.org/faa/aim/chapter-1/section-1-1-9.html",
         "txt_path": "aim/chapter-1/section-1-1-9.txt"},
        {"doc": "aim", "section_id": "glossary", "chapter": "pilot-controller-glossary",
         "title": "Pilot/Controller Glossary",
         "url": "https://faraim.org/faa/aim/pilot-controller-glossary/glossary.html",
         "txt_path": "aim/pilot-controller-glossary/glossary.txt"},
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    (tmp_path / "far/title-14/part-91").mkdir(parents=True)
    (tmp_path / "far/title-14/part-91/section-91.113.txt").write_text(
        "(b) General. When weather conditions permit, vigilance shall be maintained.",
        encoding="utf-8")
    (tmp_path / "far/title-49/part-1544").mkdir(parents=True)
    (tmp_path / "far/title-49/part-1544/section-1544.3.txt").write_text(
        "Each certificate holder must have a security program.", encoding="utf-8")
    (tmp_path / "aim/chapter-1").mkdir(parents=True)
    (tmp_path / "aim/chapter-1/section-1-1-9.txt").write_text(
        "The ILS provides \u201cexact alignment\u201d on final approach.", encoding="utf-8")
    (tmp_path / "aim/pilot-controller-glossary").mkdir(parents=True)
    (tmp_path / "aim/pilot-controller-glossary/glossary.txt").write_text(
        "VFR: Visual flight rules.", encoding="utf-8")
    return SectionIndex(tmp_path)


def test_canonical_citations(tmp_path):
    idx = make_index(tmp_path)
    far = idx.get("91.113")
    assert canonical_citation(far) == (
        "14 CFR § 91.113 (Right-of-way rules: Except water operations.)")
    t49 = idx.get("1544.3")
    assert canonical_citation(t49).startswith("49 CFR § 1544.3")
    aim = idx.get("section-1-1-9")
    assert canonical_citation(aim) == "AIM ¶ 1-1-9 (Instrument Landing System (ILS))"
    gloss = idx.get("glossary")
    assert canonical_citation(gloss) == "AIM Pilot/Controller Glossary"


def test_citation_variants_present_in_answer(tmp_path):
    idx = make_index(tmp_path)
    assert idx.answer_cites(
        "Yes, per 14 CFR § 91.113 you must maintain vigilance.", idx.get("91.113"))
    assert idx.answer_cites(
        "See AIM 1-1-9 for the localizer details.", idx.get("section-1-1-9"))
    assert not idx.answer_cites(
        "You should always be careful.", idx.get("91.113"))


def test_quote_verbatim_with_normalization(tmp_path):
    idx = make_index(tmp_path)
    # curly quotes normalize to straight, whitespace collapses
    assert idx.quote_in_section(
        'The ILS provides "exact alignment" on final approach.', idx.get("section-1-1-9"))
    assert not idx.quote_in_section("The ILS provides rough alignment.", idx.get("section-1-1-9"))


def test_validate_pair_accepts_good_pair(tmp_path):
    idx = make_index(tmp_path)
    pair = {
        "question": "What must pilots do regarding other aircraft?",
        "answer": "14 CFR § 91.113 requires vigilance: vigilance shall be maintained.",
        "quotes": [{"section_id": "91.113",
                    "quote": "vigilance shall be maintained"}],
        "citations": ["91.113"],
    }
    assert idx.validate_pair(pair) == []


def test_validate_pair_rejects_hallucinated_quote(tmp_path):
    idx = make_index(tmp_path)
    pair = {
        "question": "q",
        "answer": "Under 14 CFR § 91.113 pilots must do a loop.",
        "quotes": [{"section_id": "91.113", "quote": "pilots must do a loop"}],
        "citations": ["91.113"],
    }
    problems = idx.validate_pair(pair)
    assert any("not verbatim" in p for p in problems)


def test_validate_pair_rejects_missing_citation(tmp_path):
    idx = make_index(tmp_path)
    pair = {
        "question": "q", "answer": "Just vigilance, no reference.",
        "quotes": [{"section_id": "91.113", "quote": "vigilance shall be maintained"}],
        "citations": ["91.113"],
    }
    assert any("canonical citation" in p for p in idx.validate_pair(pair))


def test_validate_pair_rejects_unknown_section(tmp_path):
    idx = make_index(tmp_path)
    pair = {"question": "q", "answer": "42 CFR § 9.9 says hi",
            "quotes": [{"section_id": "9.9", "quote": "x"}], "citations": ["9.9"]}
    assert any("unknown section_id" in p for p in idx.validate_pair(pair))


def test_normalize_handles_dashes_and_spaces():
    assert normalize("1\u22121\u22129.  General") == normalize("1-1-9. General")
