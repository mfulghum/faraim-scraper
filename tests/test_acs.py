from pathlib import Path

import pymupdf
import pytest

from faraim_scraper.acs import parse_acs

BODY = """Area of Operation I.  Preflight Preparation
Task A.  Pilot Qualifications
References:
14 CFR parts 61, 68, 91; AC 68-1; FAA-H-8083-2, FAA-H-8083-25
Objective:
To determine the applicant exhibits satisfactory knowledge, risk
management, and skills associated with airman and medical certificates.
Knowledge:
The applicant demonstrates understanding of:
PA.I.A.K1
Certification requirements, recent flight experience, and recordkeeping.
PA.I.A.K2
Privileges and limitations.
Risk Management:
The applicant is able to identify, assess, and mitigate risk associated with:
PA.I.A.R1
Proficiency versus currency.
Skills:
The applicant exhibits the skill to:
PA.I.A.S1
Apply requirements to act as pilot-in-command under VFR.
"""

FIRST = """Private Pilot for Airplane Category
Airman Certification Standards
FAA-S-ACS-6C
Flight Standards Service
Washington, DC 20591
November 2023
"""


@pytest.fixture
def acs_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    for text in (FIRST, BODY):
        page = doc.new_page()
        y = 60
        for line in text.splitlines():
            page.insert_text((50, y), line, fontsize=9)
            y += 14
    path = tmp_path / "private_airplane_acs_6.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_parse_acs_structure(acs_pdf):
    parsed = parse_acs(acs_pdf)
    assert parsed["doc_id"] == "FAA-S-ACS-6C"
    assert parsed["title"] == "Private Pilot for Airplane Category"
    assert parsed["category"] == "PA"
    codes = [e["code"] for e in parsed["elements"]]
    assert codes == ["PA.I.A.K1", "PA.I.A.K2", "PA.I.A.R1", "PA.I.A.S1"]
    k1 = parsed["elements"][0]
    assert k1["text"].startswith("Certification requirements")
    assert k1["area"] == "I" and k1["task"] == "A" and k1["elem_type"] == "K"
    # label-terminator: the R1 element must not absorb the Skills intro
    r1 = next(e for e in parsed["elements"] if e["code"] == "PA.I.A.R1")
    assert "Skills" not in r1["text"] and "exhibits the skill" not in r1["text"]


def test_parse_acs_task_objective_and_refs(acs_pdf):
    parsed = parse_acs(acs_pdf)
    assert len(parsed["tasks"]) == 1
    task = parsed["tasks"][0]
    assert task["code"] == "PA.I.A"
    assert task["task_title"] == "Pilot Qualifications"
    assert task["objective"].startswith("To determine")
    assert "14 CFR part 61" in task["refs"]
    assert "AC 68-1" in task["refs"]
    assert "FAA-H-8083-25" in task["refs"]
