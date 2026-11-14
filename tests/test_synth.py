

import json

from faraim_scraper.synthesize import purge_stale_pairs


def test_purge_stale_pairs(tmp_path):
    path = tmp_path / "qa.jsonl"
    good = {"chunk_id": "hbk:B:ch1#1", "type": "factual",
            "pair": {"quotes": [{"quote": "Altitude 2,000 feet."}]}}
    stale = {"chunk_id": "hbk:B:ch1#2", "type": "factual",
             "pair": {"quotes": [{"quote": "text from the OLD chunking"}]}}
    orphan = {"chunk_id": "hbk:GONE:ch1", "type": "factual",
              "pair": {"quotes": []}}
    noq = {"chunk_id": "hbk:B:ch1#1", "type": "abstention", "pair": {}}
    path.write_text("\n".join(json.dumps(r) for r in
                              (good, stale, orphan, noq)) + "\n",
                    encoding="utf-8")
    texts = {"hbk:B:ch1#1": "…the aircraft altitude 2,000\u2009feet.",
             "hbk:B:ch1#2": "brand new text after re-chunking"}
    purge_stale_pairs(path, texts)
    left = [json.loads(l) for l in path.read_text().splitlines()]
    assert [r["chunk_id"] + r["type"][:4] for r in left] == \
        ["hbk:B:ch1#1fact", "hbk:B:ch1#1abst"]
