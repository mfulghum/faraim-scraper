import json

from faraim_scraper.citations import SectionIndex
from faraim_scraper.filters import dedupe_queries, filter_file, gate


def make_index(tmp_path):
    recs = [
        {"doc": "far", "section_id": "91.113", "part": "91",
         "title": "§ 91.113 Right-of-way rules: Except water operations.",
         "url": "https://faraim.org/faa/far/cfr/title-14/part-91/section-91.113.html",
         "txt_path": "far/title-14/part-91/section-91.113.txt"},
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    (tmp_path / "far/title-14/part-91").mkdir(parents=True)
    (tmp_path / "far/title-14/part-91/section-91.113.txt").write_text(
        "(b) General. When weather conditions permit, regardless of whether an "
        "operation is conducted under instrument flight rules or visual flight "
        "rules, vigilance shall be maintained by each person operating an aircraft "
        "so as to see and avoid other aircraft.", encoding="utf-8")
    return SectionIndex(tmp_path)


def good_pair(question, answer, quote="vigilance shall be maintained"):
    return {"chunk_id": "far:91:91.113", "type": "factual", "problems": [],
            "pair": {"question": question, "answer": answer,
                     "quotes": [{"section_id": "91.113", "quote": quote}],
                     "citations": ["91.113"]}}


def test_gate_rejects_short_and_refusal(tmp_path):
    idx = make_index(tmp_path)
    assert "answer too short" in gate(good_pair("Must I maintain vigilance under 14 CFR § 91.113?",
                                                "Yes, 14 CFR § 91.113."), idx, "factual")
    assert "question too short" in gate(good_pair("huh?", "x" * 100), idx, "factual")
    refusal = good_pair("Must I maintain vigilance per 14 CFR § 91.113?",
                        "I cannot answer that question because the excerpt is unclear here.")
    assert "refusal answer" in gate(refusal, idx, "factual")


def test_gate_allows_good_pair(tmp_path):
    idx = make_index(tmp_path)
    rec = good_pair(
        "What does 14 CFR § 91.113 require of pilots regarding other aircraft?",
        "Pilots must keep lookout: 14 CFR § 91.113 states vigilance shall be "
        "maintained by each person operating an aircraft so as to see and avoid "
        "other aircraft.")
    assert gate(rec, idx, "factual") == []


def test_dedupe_keeps_longest_answer():
    a = good_pair("What does the rule require for seeing other aircraft?", "short " * 20)
    b = good_pair("What does the rule require for seeing other aircrafts?", "longer answer " * 30)
    kept, dropped = dedupe_queries([a, b])
    assert dropped == 1
    assert len(kept) == 1
    assert kept[0]["pair"]["answer"].startswith("longer")


def test_filter_file_end_to_end(tmp_path):
    idx = make_index(tmp_path)
    records = [
        good_pair("What does 14 CFR § 91.113 require of pilots regarding other aircraft?",
                  "Pilots must keep lookout: 14 CFR § 91.113 requires that vigilance "
                  "shall be maintained by each person operating an aircraft so as to "
                  "see and avoid other aircraft."),
        good_pair("What does 14 CFR § 91.113 require of pilots regarding other aircrafts?",
                  "Vigilance: 14 CFR § 91.113 says vigilance shall be maintained by "
                  "each person operating an aircraft so as to see and avoid others."),
        {"chunk_id": "far:91:91.113", "type": "factual", "pair": None,
         "problems": ["api error"]},
    ]
    in_path = tmp_path / "raw.jsonl"
    in_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    stats = filter_file(in_path, tmp_path / "clean.jsonl", idx)
    assert stats["raw_valid"] == 2
    assert stats["after_dedupe"] == 1
    cleaned = [json.loads(l) for l in (tmp_path / "clean.jsonl").read_text().splitlines()]
    assert len(cleaned) == 1
