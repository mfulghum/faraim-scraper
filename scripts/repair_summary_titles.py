"""One-shot repair: regenerate subsection titles from source text with the
current _block_title (fixes level-mixing and mid-word chops in place)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faraim_scraper.tree_summaries import TOP_BLOCK_RE, _block_title  # noqa: E402


def main(data_dir="data", out="work/summaries.jsonl"):
    data = Path(data_dir)
    manifest = {r["section_id"]: r for r in
                (json.loads(l) for l in (data / "manifest.jsonl")
                 .read_text(encoding="utf-8").splitlines())
                if r["doc"] == "far"}
    cache = {}

    def section_text(sid):
        if sid not in cache:
            rec = manifest.get(sid)
            cache[sid] = (data / rec["txt_path"]).read_text(
                encoding="utf-8") if rec else None
        return cache[sid]

    def title_for(sid, marker):
        text = section_text(sid)
        if not text:
            return None
        marks = list(TOP_BLOCK_RE.finditer(text))
        for i, m in enumerate(marks):
            if f"({m.group(1)})" == marker:
                end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
                return _block_title(text[m.start():end].strip())
        return None

    path = Path(out)
    fixed = 0
    out_lines = []
    for l in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r.get("level") == "subsection" and r.get("section_id"):
            t = title_for(r["section_id"], r["id"].rsplit("#", 1)[-1])
            if t and t != r.get("title"):
                r["title"] = t
                fixed += 1
        out_lines.append(json.dumps(r, ensure_ascii=False))
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(f"titles repaired: {fixed} / {len(out_lines)}")


if __name__ == "__main__":
    main()
