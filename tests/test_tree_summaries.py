import json

from faraim_scraper.tree_summaries import (
    build_plan, leaf_prompt, load_done, parent_prompt, subsection_blocks)

CFR_TEXT = """
(a) Except as provided in paragraph (b) of this section and § 91.113, no person may operate an aircraft so close to another aircraft as to create a collision hazard. Pilots must maintain vigilance to see and avoid other aircraft at all times during operation of the aircraft under these rules described in this section.

(b) No person may operate a civil or military aircraft without showing the right of way to other aircraft as prescribed in this section. Each person operating an aircraft shall maintain careful vigilance in order to see and avoid other aircraft whenever operating in the conditions described herein.

(c) Right-of-way given to another aircraft by this section shall not be construed as relieving the remote pilot in command from the duty to give way to another aircraft. This obligation continues until the other aircraft has clearly passed and is no longer a hazard to the give-way aircraft under these rules.

(d) An aircraft towing or refueling other aircraft has the right of way over all other engine-driven aircraft operating in the same general vicinity of the tow or refueling operation described in this paragraph. This right of way applies whether the towing operation is conducted for compensation or hire or for personal purposes under the rules of this part.
""".strip()


def test_subsection_blocks():
    blocks = subsection_blocks(CFR_TEXT)
    assert [b["marker"] for b in blocks] == ["(a)", "(b)", "(c)", "(d)"]
    assert blocks[0]["title"].startswith("(a) Except as provided")
    # too few top-level markers -> no split
    assert subsection_blocks("just one (a) paragraph here") == []


def make_data(tmp_path):
    recs = [
        {"doc": "far", "section_id": "91.205", "part": "91",
         "title": "§ 91.205 Powered civil aircraft: Instrument and equipment.",
         "url": "u", "html_path": "h1", "txt_path": "far/91.205.txt"},
        {"doc": "aim", "section_id": "section-4-3-24", "chapter": "chapter-4",
         "title": "Lights", "url": "u", "html_path": "h2",
         "txt_path": "aim/4-3-24.txt"},
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    (tmp_path / "far").mkdir()
    (tmp_path / "far" / "91.205.txt").write_text("filler. " * 400 + CFR_TEXT,
                                                 encoding="utf-8")
    (tmp_path / "aim").mkdir()
    (tmp_path / "aim" / "4-3-24.txt").write_text("position lights",
                                                 encoding="utf-8")
    return tmp_path


def test_build_plan_hierarchy(tmp_path):
    plan = build_plan(make_data(tmp_path))
    by_id = {n["id"]: n for n in plan}
    assert "sum:far:91:91.205" in by_id
    assert "sum:far:91" in by_id
    assert "sum:aim:chapter-4" in by_id
    assert any(i.startswith("sum:far:91:91.205#(") for i in by_id)
    sub = next(n for n in plan if n["level"] == "subsection")
    assert sub["parent_id"] == "sum:far:91:91.205"
    assert by_id["sum:far:91"]["parent_id"] == "far"
    assert {c["id"] for c in by_id["sum:far:91"]["children"]} == {"sum:far:91:91.205"}


def test_build_plan_includes_handbooks(tmp_path):
    data = make_data(tmp_path)
    hb = data / "handbooks"
    hb.mkdir()
    (hb / "handbook_chunks.jsonl").write_text("\n".join(json.dumps({
        "id": f"hbk:amtg_handbook:ch3#{i}", "handbook": "amtg_handbook",
        "handbook_title": "Technician Handbook—General", "chapter": 3,
        "chapter_title": "DC Circuits", "pages": "3-1-3-40",
        "text": "Voltage is the force that pushes current. " * 20,
        "n_tokens": 120}) for i in (1, 2)), encoding="utf-8")
    plan = build_plan(data)
    sec = next(n for n in plan if n["id"] == "sum:hbk:amtg_handbook:ch3#1")
    assert sec["level"] == "section" and sec["doc"] == "handbook"
    assert sec["parent_id"] == "sum:hbk:amtg_handbook:ch3"
    ch = next(n for n in plan if n["id"] == "sum:hbk:amtg_handbook:ch3")
    assert ch["level"] == "part" and len(ch["children"]) == 2


def test_prompts_and_resume(tmp_path):
    node = {"title": "§ 91.205", "text": "night equipment list here"}
    lp = leaf_prompt(node, "far part 91")
    assert "night equipment list" in lp and "far part 91" in lp
    pp = parent_prompt({"title": "Part 91"}, ["- 91.205: equipment rules"],
                       "far overview")
    assert "equipment rules" in pp
    (tmp_path / "s.jsonl").write_text(
        json.dumps({"id": "x", "summary": "s"}) + "\n", encoding="utf-8")
    done = load_done(tmp_path / "s.jsonl")
    assert done["x"]["summary"] == "s"
