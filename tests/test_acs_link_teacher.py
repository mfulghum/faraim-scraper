import json

from faraim_scraper.acs_link_teacher import (
    CandidatePool, build_groups, links_prompt, parse_scope, toc_prompt,
    validate_links, _toc_lines)


def _data(tmp_path):
    recs = [
        {"doc": "far", "section_id": "91.209", "part": "91",
         "title": "§ 91.209 Aircraft lights.",
         "url": "u", "html_path": "far/title-14/part-91/section-91.209.html",
         "txt_path": "far/title-14/part-91/section-91.209.txt"},
        {"doc": "aim", "section_id": "section-4-3-24", "chapter": "chapter-4",
         "title": "Lights", "url": "u",
         "html_path": "aim/chapter-4/section-4-3-24.html",
         "txt_path": "aim/chapter-4/section-4-3-24.txt"},
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    (tmp_path / "far/title-14/part-91").mkdir(parents=True)
    (tmp_path / "far/title-14/part-91/section-91.209.txt").write_text(
        "No person may operate an aircraft during the period from sunset to "
        "sunrise unless it has lighted position lights.", encoding="utf-8")
    (tmp_path / "aim/chapter-4").mkdir(parents=True)
    (tmp_path / "aim/chapter-4/section-4-3-24.txt").write_text(
        "Aircraft position lights are required to be lighted.", encoding="utf-8")
    hb = tmp_path / "handbooks"
    hb.mkdir()
    (hb / "handbook_chunks.jsonl").write_text(json.dumps({
        "id": "hbk:FAA-H-8083-25C:ch8#1", "handbook": "FAA-H-8083-25C",
        "handbook_title": "Pilot's Handbook of Aeronautical Knowledge",
        "chapter": 8, "chapter_title": "Flight Controls", "pages": "8-1-8-20",
        "text": "Position lights red and green mark the aircraft.",
        "n_tokens": 10}) + "\n", encoding="utf-8")
    return tmp_path


def test_parse_scope():
    s = parse_scope(["14 CFR part 91", "AC 68-1", "FAA-H-8083-25", "AIM 4-3-24"])
    assert s["parts"] == ["91"]
    assert s["hb_codes"] == ["FAA-H-8083-25"]
    assert s["aim_paras"] == ["4-3-24"]
    assert s["aim_bare"] is True
    assert parse_scope([]) == {"parts": [], "hb_codes": [],
                               "aim_paras": [], "aim_bare": False}


def test_pool_rows_and_load(tmp_path):
    pool = CandidatePool(_data(tmp_path))
    rows = pool.rows(parse_scope(["14 CFR part 91", "FAA-H-8083-25", "AIM"]))
    kinds = [r["kind"] for r in rows]
    assert kinds.count("far") == 1 and "handbook" in kinds and "aim" in kinds
    passages, loaded = [], set()
    pool.load_selected(rows, [1, 2, 3, 99], loaded, passages)
    assert {p["section_id"] for p in passages} == {
        "91.209", "section-4-3-24", "hbk:FAA-H-8083-25C:ch8#1"}
    # second round is idempotent and additive
    pool.load_selected(rows, [1, 2], loaded, passages)
    assert len(passages) == 3


def test_validate_links_verbatim():
    passages = [{"cite": "14 CFR § 91.209", "kind": "far", "section_id": "91.209",
                 "text": "No person may operate an aircraft during the period "
                         "from sunset to sunrise unless it has lighted position "
                         "lights.", "html_path": "h", "url": "", "title": "t"}]
    codes = ["HI.II.M.K3", "HI.II.M (task)"]
    good = [{"code": "HI.II.M.K3", "target_id": 1,
             "quote": "lighted position lights"}]
    out = validate_links(good, passages, codes)
    assert len(out) == 1 and out[0]["strength"] == 3
    assert out[0]["evidence"] == "llm full-text"
    # fabricated quote, bad code, bad target all dropped
    bad = [{"code": "HI.II.M.K3", "target_id": 1, "quote": "must be lighted at night"},
           {"code": "BOGUS.K1", "target_id": 1, "quote": "position lights"},
           {"code": "HI.II.M.K3", "target_id": 7, "quote": "position lights"}]
    assert validate_links(bad, passages, codes) == []


def test_prompts_and_toc_marks(tmp_path):
    pool = CandidatePool(_data(tmp_path))
    rows = pool.rows(parse_scope(["14 CFR part 91", "AIM"]))
    passages, loaded = [], set()
    pool.load_selected(rows, [1], loaded, passages)
    g = {"task_code": "HI.II.M", "objective": "Night operations",
         "elements": [("HI.II.M.K3", "Helicopter equipment and lighting "
                                      "requirements for night operations.")],
         "refs": ["14 CFR part 91"], "doc_title": "CFI Helicopter"}
    from faraim_scraper.acs_link_teacher import query_block
    tp = toc_prompt(rows, query_block(g), g["doc_title"])
    assert "HI.II.M.K3" in tp and "[1] 14 CFR § 91.209" in tp
    lp = links_prompt(passages, query_block(g), g["doc_title"],
                      _toc_lines(rows, loaded))
    assert "sunset to sunrise" in lp        # full text loaded
    assert "Aircraft lights. *" in lp       # TOC line marked as loaded
    assert "load_more" in lp


def test_load_section_by_id_outside_scope(tmp_path):
    pool = CandidatePool(_data(tmp_path))
    passages: list = []
    assert pool.load_section_by_id("91.209", passages)
    assert passages[0]["section_id"] == "91.209"
    assert "sunset to sunrise" in passages[0]["text"]
    assert passages[0]["cite"] == "14 CFR § 91.209 Aircraft lights."
    assert pool.load_section_by_id("99.99", passages) is False
    assert len(passages) == 1


def test_verify_prompt_flags_missing_links():
    from faraim_scraper.acs_link_teacher import verify_prompt, _plan_requirement_lines
    codes = ["HI.II.M.K3", "HI.II.M.K4"]
    plan_map = {"HI.II.M.K3": {"requirements": [
        "night VFR instrument and equipment list, 14 CFR 91.205(c)",
        "position lights, 14 CFR 91.209"]}}
    links = [{"code": "HI.II.M.K3", "section_id": "91.209",
              "quote": "lighted position lights"}]
    vp = verify_prompt(codes, plan_map, links, ["[617] 14 CFR § 91.205"])
    assert "91.205(c)" in vp                      # required point listed
    assert "link: (none)" in vp                   # K4 failing
    assert "[617] 14 CFR § 91.205" in vp          # contents shown
    req = _plan_requirement_lines(plan_map, codes)
    assert "HI.II.M.K3 requires:" in req and "HI.II.M.K4" not in req


def test_requirement_sections_named_by_answer_key():
    from faraim_scraper.acs_link_teacher import requirement_sections
    assert requirement_sections([
        "14 CFR 91.205(c) — night VFR equipment",
        "position lights, FAR 91.209; AIM 8-1-5; see § 61.57"]) == {
        "91.205", "91.209", "61.57"}


def test_merge_folds_teacher_links(tmp_path):
    import json as _json
    from faraim_scraper.acs_link_teacher import merge
    acs_dir = tmp_path / "acs"
    acs_dir.mkdir()
    def w(path, recs):
        (acs_dir / path).write_text(
            "\n".join(_json.dumps(r) for r in recs), encoding="utf-8")
    w("acs_links.jsonl", [
        {"code": "IH.I.A.K1", "pdf": "x.pdf", "kind": "element",
         "links": [
             {"type": "far", "section_id": "61.57", "title": "t",
              "html_path": "far/61.57.html", "url": "u",
              "evidence": "semantic (cosine=0.2)", "strength": 2},
             {"type": "far", "section_id": "61.3", "title": "t",
              "html_path": "far/61.3.html", "url": "u",
              "evidence": "§ 61.3", "strength": 3},
             {"type": "handbook", "section_id": "FAA-H-8083-25", "title": None,
              "html_path": "handbooks/FAA-H-8083-25", "url": "",
              "evidence": "references FAA-H-8083-25", "strength": 3}]},
        {"code": "ZZ.Z.Z.K1", "pdf": "y.pdf", "kind": "element",
         "links": [{"type": "far", "section_id": "91.1", "title": "t",
                    "html_path": "far/91.1.html", "url": "u",
                    "evidence": "semantic (cosine=0.2)", "strength": 2}]},
    ])
    llm = tmp_path / "llm.jsonl"
    llm.write_text(_json.dumps({"group": "x|IH.I.A", "links": [
        {"code": "IH.I.A.K1", "type": "far", "section_id": "61.57",
         "title": "Recent flight experience", "html_path": "far/61.57.html",
         "url": "u", "quote": "six instrument approaches",
         "evidence": "llm gap-fill (auto-quote)", "strength": 3}]}) + "\n",
        encoding="utf-8")
    stats = merge(acs_dir, llm)
    assert stats["teacher_merged_records"] == 1
    forward = [_json.loads(l) for l in
               (acs_dir / "acs_links.jsonl").read_text().splitlines()]
    k1 = next(r for r in forward if r["code"] == "IH.I.A.K1")
    ids = {l["section_id"] for l in k1["links"]}
    assert ids == {"61.3", "FAA-H-8083-25", "61.57"}   # cosine dropped, explicit kept
    assert next(l for l in k1["links"] if l["section_id"] == "61.57")["evidence"] \
        == "llm gap-fill (auto-quote)"
    zz = next(r for r in forward if r["code"] == "ZZ.Z.Z.K1")
    assert zz["links"][0]["strength"] == 2             # untouched, no teacher links
    reverse = [_json.loads(l) for l in
               (acs_dir / "acs_keys_by_section.jsonl").read_text().splitlines()]
    entry = next(r for r in reverse if r["html_path"] == "far/61.57.html")
    assert "IH.I.A.K1" in entry["acs_keys"]


def test_build_groups_document_prior_refs(tmp_path):
    acs = tmp_path / "acs"
    acs.mkdir()
    (acs / "acs_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in [
        {"code": "PA.I.A", "area": "I", "task": "A", "objective": "o1",
         "pdf": "x.pdf", "refs": ["14 CFR part 61", "FAA-H-8083-25"]},
        {"code": "PA.I.B", "area": "I", "task": "B", "objective": "o2",
         "pdf": "x.pdf", "refs": []},
    ]), encoding="utf-8")
    (acs / "acs_elements.jsonl").write_text("", encoding="utf-8")
    by_code = {g["task_code"]: g for g in build_groups(acs)}
    assert by_code["PA.I.B"]["refs"] == ["14 CFR part 61", "FAA-H-8083-25"]
    assert by_code["PA.I.B"]["refs_source"] == "document"
    assert "refs_source" not in by_code["PA.I.A"]


def test_build_groups(tmp_path):
    acs = tmp_path / "acs"
    acs.mkdir()
    (acs / "acs_tasks.jsonl").write_text(json.dumps({
        "code": "HI.II.M", "area": "II", "task": "M", "task_title": "Night",
        "objective": "Night operations", "pdf": "cfi_helicopter_acs_29.pdf",
        "refs": ["14 CFR part 91", "AIM"]}), encoding="utf-8")
    (acs / "acs_elements.jsonl").write_text("\n".join(json.dumps({
        "code": f"HI.II.M.K{i}", "text": f"element {i}", "elem_type": "K",
        "pdf": "cfi_helicopter_acs_29.pdf", "doc_title": "CFI Helicopter"})
        for i in (1, 2)), encoding="utf-8")
    groups = build_groups(acs)
    assert len(groups) == 1
    g = groups[0]
    assert g["refs"] == ["14 CFR part 91", "AIM"]  # inherited by elements
    assert len(g["elements"]) == 2
