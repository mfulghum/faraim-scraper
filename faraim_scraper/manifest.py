"""Write the dataset manifest and scrape provenance info."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def write_manifest(out_root: Path, records: List[Dict]) -> Path:
    """Write all section records to manifest.jsonl (one JSON object per line)."""
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "manifest.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_scrape_info(
    out_root: Path,
    base_url: str,
    records: List[Dict],
    failed: List[str],
    far_index: Optional[str] = None,
    aim_source: Optional[str] = None,
) -> Path:
    """Write scrape provenance and summary counts to scrape_info.json."""
    by_doc: Dict[str, Dict[str, int]] = {}
    for r in records:
        doc = by_doc.setdefault(r["doc"], {"sections": 0, "images": 0})
        doc["sections"] += 1
        doc["images"] += len(r["images"])
    info = {
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": base_url,
        "far_index": far_index,
        "aim_source": aim_source,
        "documents": by_doc,
        "failed": failed,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "scrape_info.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
