"""Link ACS elements/tasks to the scraped FAR/AIM/handbook sections.

Matching evidence, strongest first:
  1. explicit citations in ACS text: 14 CFR § 91.103 / AIM ¶ 5-1-4 / "AIM 4-3-3"
  2. reference-scoped semantic hits: the task's References line
     ("14 CFR part 61; FAA-H-8083-25...") CONSTRAINS the candidate pool to the
     sections of those CFR parts and the chunks of those handbooks; elements
     inherit their parent task's References
  3. book-level links for every handbook cited in References
  4. global semantic fallback for records whose References name no scraped doc

Outputs (in the ACS output dir):
  acs_links.jsonl        one line per ACS code: links + evidence
  acs_keys_by_section.jsonl
                         one line per section/handbook chunk: relevant ACS codes
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

FAR_SECTION_REF_RE = re.compile(r"§\s*(\d+\.\d+)")
AIM_PARA_REF_RE = re.compile(r"\b(\d+-\d{1,2}-\d{1,3})\b")
FAR_PART_REF_RE = re.compile(r"14 CFR parts?\s+([\d,\sand]+)")
FAR_PART_WORD_RE = re.compile(r"\b[Pp]art\s+(\d{1,4})\b")

# ACS References cite handbook codes that don't match our file-derived ids.
HB_CODE_ALIASES = {
    "FAA-H-8083-30": ["amtg_handbook"],                      # AMT General
    "FAA-H-8083-31": ["amt_powerplant_handbook",             # AMT Powerplant...
                      "FAA-H-8083-31B"],                      # ...and Airframe
    "FAA-H-8083-24": ["FAA-H-8083-9B"],                       # Aviation Instructor
}


def _hb_key(code: str):
    """Canonical join key for a handbook code: ('8083','25') or lowercase stem."""
    m = re.match(r"FAA-H-(\d+)-(\d+)", code.upper())
    return (m.group(1), m.group(2)) if m else code.lower()


_TOKEN_RE = re.compile(r"[a-z0-9§¶\-]+")
# ACS prose and regulation prose use different words for the same concept.
_SYNONYMS = {
    "recency": "recent", "currency": "current", "logbook": "record",
    "rating": "certificate", "proficiency": "competency",
}


def _stem(token: str) -> str:
    """Light suffix stripping so 'experiences'~'experience', 'certifying'~'certify'."""
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize_stem(text: str) -> List[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]


def _expand(query: str) -> str:
    """Append regulation-side vocabulary for ACS-side terms (post-stem match)."""
    toks = {_stem(t) for t in _TOKEN_RE.findall(query.lower())}
    extra = [v for k, v in _SYNONYMS.items() if _stem(k) in toks]
    return query + " " + " ".join(extra) if extra else query


def parse_refs(text: str) -> Dict[str, List[str]]:
    """Extract far section ids, aim paragraph ids and part numbers from text."""
    far_sections = FAR_SECTION_REF_RE.findall(text)
    aim_paras = AIM_PARA_REF_RE.findall(text)
    parts: List[str] = []
    for m in FAR_PART_REF_RE.finditer(text):
        parts.extend(re.findall(r"\d+", m.group(1)))
    for m in FAR_PART_WORD_RE.finditer(text):
        parts.append(m.group(1))
    return {"far_sections": far_sections, "aim_paras": aim_paras,
            "parts": sorted(set(parts))}


class AcsLinker:
    def __init__(self, data_dir: Path, semantic: bool = True,
                 bm25_threshold: float = 0.12, bm25_topk: int = 2):
        self.data_dir = Path(data_dir)
        self._threshold = bm25_threshold
        self._topk = bm25_topk
        # far lookup: "91.113" -> rec; also "49-1544.3" keyed by title to avoid
        # collisions between title-14 and title-49 numbering.
        self.far_by_section: Dict[str, List[dict]] = defaultdict(list)
        self.far_by_part: Dict[str, List[dict]] = defaultdict(list)
        self.aim_by_para: Dict[str, dict] = {}
        for line in (self.data_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["doc"] == "far":
                key = rec["section_id"]
                self.far_by_section[key].append(rec)
                self.far_by_part[f"{rec.get('part')}"].append(rec)
            else:
                sid = rec["section_id"]
                if sid.startswith("section-"):
                    para = sid[len("section-"):]
                    self.aim_by_para[para] = rec

        self.semantic_docs: List[dict] = []
        self._tfidf = None
        self._matrix = None
        if semantic:
            from sklearn.feature_extraction.text import TfidfVectorizer
            corpus_texts = []
            for line in (self.data_dir / "manifest.jsonl").read_text(
                    encoding="utf-8").splitlines():
                rec = json.loads(line)
                try:
                    text = (self.data_dir / rec["txt_path"]).read_text(encoding="utf-8")
                except OSError:
                    continue
                self.semantic_docs.append(rec)
                corpus_texts.append(((rec.get("title") or "") + " " + text).lower())
            # handbook chunks extend the semantic pool (FAA-H-8083-x refs)
            hb_path = self.data_dir / "handbooks" / "handbook_chunks.jsonl"
            if hb_path.is_file():
                for line in hb_path.read_text(encoding="utf-8").splitlines():
                    c = json.loads(line)
                    self.semantic_docs.append({
                        "doc": "handbook", "section_id": c["id"],
                        "title": f"{c['handbook_title']} — {c.get('chapter_title') or ''}",
                        "html_path": f"handbooks/{c['id']}.html",
                        "url": f"https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/",
                        "handbook": c["handbook"],
                    })
                    corpus_texts.append(
                        (c["handbook"] + " " + c["handbook_title"] + " " +
                         (c.get("chapter_title") or "") + " " + c["text"]).lower())
            self._tfidf = TfidfVectorizer(token_pattern=r"[a-z0-9§¶\-]+",
                                          sublinear_tf=True, min_df=2, max_df=0.7)
            try:
                self._matrix = self._tfidf.fit_transform(corpus_texts)
            except ValueError:
                self._tfidf = None  # corpus too small/degenerate for TF-IDF

    def semantic_hits(self, query: str, topk: int = 3) -> List[tuple]:
        if self._tfidf is None or not query.strip():
            return []
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(self._tfidf.transform([query.lower()]),
                                   self._matrix)[0]
        order = scores.argsort()[::-1][:topk]
        return [(self.semantic_docs[i], float(scores[i])) for i in order
                if scores[i] >= self._threshold]

    def _far_recs(self, section_id: str, parts_hint: List[str]) -> List[dict]:
        recs = self.far_by_section.get(section_id, [])
        if len(recs) > 1 and parts_hint:
            narrowed = [r for r in recs if r.get("part") in parts_hint]
            if narrowed:
                return narrowed
        return recs

    def links_for(self, acs_code: str, text: str, refs: List[str]) -> Dict:
        """Return {links, parts} for one ACS code given its searchable text+refs."""
        blob = text + " " + " ".join(refs)
        parsed = parse_refs(blob)
        links: Dict[str, dict] = {}

        def add(rec: dict, evidence: str, strength: int):
            key = rec["html_path"]
            if key not in links or links[key]["strength"] < strength:
                links[key] = {
                    "type": rec["doc"], "section_id": rec["section_id"],
                    "title": rec.get("title"), "html_path": rec["html_path"],
                    "url": rec["url"], "evidence": evidence, "strength": strength,
                }

        for sid in set(parsed["far_sections"]):
            for rec in self._far_recs(sid, parsed["parts"]):
                add(rec, f"§ {sid}", 3)
        for para in set(parsed["aim_paras"]):
            rec = self.aim_by_para.get(para)
            if rec is not None:
                add(rec, f"AIM {para}", 3)
        for rec, score in self.semantic_hits(text, topk=self._topk):
            add(rec, f"semantic (cosine={score:.2f})", 2)
        # Referenced handbooks (FAA-H-8083-x) from ACS References lines.
        cited_hbks = {h.upper() for h in re.findall(r"FAA-H-8083-\d+\w*", blob, re.I)}
        for code in sorted(cited_hbks):
            add({"doc": "handbook", "section_id": code, "title": None,
                 "html_path": f"handbooks/{code}", "url": ""},
                f"references {code}", 3)
            # prefer semantic hits inside a referenced handbook
            deep = self.semantic_hits(text, topk=30)
            in_hbk = [(r, s) for r, s in deep
                      if r.get("doc") == "handbook"
                      and r.get("handbook", "").upper().startswith(code)]
            if in_hbk:
                rec, score = in_hbk[0]
                add(rec, f"handbook-ref semantic (cosine={score:.2f})", 3)
        # Part boost: when the ACS text names a CFR part, prefer the best hit
        # inside that part even if it loses to generic lexically-similar text.
        detected = set(parsed["parts"])
        if detected:
            deep = self.semantic_hits(text, topk=12)
            in_part = [(r, s) for r, s in deep
                       if r["doc"] == "far" and r.get("part") in detected]
            if in_part:
                rec, score = in_part[0]
                if score >= self._threshold / 2:
                    add(rec, f"semantic+part (cosine={score:.2f})", 2)
        parts = sorted(set(parsed["parts"]))
        return {"links": sorted(links.values(), key=lambda l: -l["strength"]),
                "parts": parts}


def build_links(acs_dir: Path, data_dir: Path) -> dict:
    acs_dir = Path(acs_dir)
    linker = AcsLinker(data_dir)

    forward: List[dict] = []
    reverse: Dict[str, dict] = {}

    def handle(code: str, query: str, text: str, refs: List[str], extra: dict):
        result = linker.links_for(code, query, refs)
        links = result["links"]
        forward.append({"code": code, **extra, "parts": result["parts"], "links": links})
        for l in links:
            entry = reverse.setdefault(l["html_path"], {
                "html_path": l["html_path"], "section_id": l["section_id"],
                "doc": l["type"], "acs_keys": []})
            if code not in entry["acs_keys"]:
                entry["acs_keys"].append(code)

    for line in (acs_dir / "acs_elements.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        query = f"{e.get('task_title','')} {e['text']}"
        handle(e["code"], query, e["text"], [], {
            "pdf": e.get("pdf"), "kind": "element", "elem_type": e["elem_type"],
            "doc_id": e.get("doc_id"), "doc_title": e.get("doc_title")})
    for line in (acs_dir / "acs_tasks.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        query = f"{t.get('task_title','')} {t.get('objective','')}"
        handle(t["code"] + " (task)", query, t.get("objective", ""), t.get("refs", []), {
            "pdf": t.get("pdf"), "kind": "task", "doc_id": None,
            "doc_title": None})

    (acs_dir / "acs_links.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in forward) + "\n",
        encoding="utf-8")
    (acs_dir / "acs_keys_by_section.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in reverse.values()) + "\n",
        encoding="utf-8")

    linked = sum(1 for r in forward if r["links"])
    strong = sum(1 for r in forward if any(l["strength"] == 3 for l in r["links"]))
    stats = {"acs_records": len(forward), "with_links": linked,
             "with_strong_links": strong, "sections_linked": len(reverse)}
    (acs_dir / "acs_link_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Build ACS <-> FAR/AIM crosswalks.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--acs", type=Path, default=Path("data/acs"))
    args = p.parse_args(argv)
    stats = build_links(args.acs, args.data)
    print("link stats:", json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
