"""Glossary entry parsing and pair construction."""
from faraim_scraper.citations import SectionIndex
from faraim_scraper.glossary_qa import (
    _aim_pair, _CAPTION_RE, parse_aim_entries, parse_hb_entries)


def test_parse_aim_entries_basic_and_wraps():
    text = (
        "PILOT/CONTROLLER\nGLOSSARY\n"
        "EXPEDITE- Used by ATC when prompt compliance is required to avoid "
        "the development of an imminent situation.\n"
        "LOW FREQUENCY (LF)- The frequency band between 30 and 300 kHz.\n"
        "ACCELERATE-STOP DISTANCE AVAILABLE- The runway plus stopway length "
        "declared\navailable and suitable for the accelerate-stop maneuver.\n"
        "SHORTIE- Too short.\n"
        "GIZMO- See SOMETHING ELSE.\n")
    entries = dict(parse_aim_entries(text))
    assert entries["EXPEDITE"].startswith("Used by ATC when prompt")
    assert "LOW FREQUENCY (LF)" in entries
    assert entries["ACCELERATE-STOP DISTANCE AVAILABLE"].startswith("The runway")
    assert "SHORTIE" not in entries          # definition < 20 chars
    assert "GIZMO" not in entries            # bare 'See X' cross-reference


def test_parse_aim_entries_no_space_dash():
    text = ("100-HOUR INSPECTION\u2014An inspection required by 14 CFR 91.409 "
            "for FAA-certificated aircraft that are operated for hire.\n"
            "ABUSE\u2014The improper maintenance or use of equipment or "
            "materials by anyone.\n")
    entries = dict(parse_aim_entries(text))
    assert any(t.startswith("100-HOUR INSPECTION") for t in entries)
    assert "ABUSE" in entries
    assert not any(t == "100" for t in entries)   # term did not split at hyphen


def test_parse_hb_entries_line_start_and_wrap():
    text = (
        "Glossary\n"
        "Airworthiness Directive. A regulatory notice sent out by the FAA to "
        "the registered owner of an aircraft.\n"
        "Best angle of climb (VX). The speed at which the aircraft will "
        "produce the best\nangle of climb.\n"
        "continuing text that wraps mid sentence and should not start an\n"
        "entry. This is not a term because the previous line ends "
        "mid-sentence.\n")
    entries = dict(parse_hb_entries(text))
    assert "Airworthiness Directive" in entries
    assert "Best angle of climb (VX)" in entries
    assert not any(t.startswith("entry") for t in entries)


def test_caption_guard():
    assert _CAPTION_RE.match("Figure 5-10")
    assert _CAPTION_RE.match("WARNING")
    assert not _CAPTION_RE.match("Wake turbulence")


def test_pair_shape_and_citation():
    p = _aim_pair("EXPEDITE", "Used by ATC when prompt compliance is required.")
    assert p["quotes"][0]["section_id"] == "glossary"
    assert "AIM Pilot/Controller Glossary" in p["answer"]
    assert "Expedite" in p["question"]
