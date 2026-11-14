

import json

import pytest

from faraim_scraper import synthesize
from faraim_scraper.synthesize import purge_stale_pairs


class _Msg:
    def __init__(self, content):
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]


class FakeClient:
    """chat.completions.create returns preset content or raises."""
    def __init__(self, content=None, exc=None):
        self._content, self._exc = content, exc
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    async def create(self, **kw):
        self.calls.append(kw)
        if self._exc:
            raise self._exc
        return _Msg(self._content)


class FakeIndex:
    def get(self, sid):
        return {"doc": "far", "section_id": sid, "title": "Rules", "url": ""}

    def text(self, rec):
        return "related excerpt text"


def _batch_payload(types):
    return json.dumps({"pairs": [
        {"type": t, "question": "q", "answer": "a",
         "quotes": [{"section_id": "91.113", "quote": "x"}],
         "citations": ["91.113"]} for t in types]})


def test_generate_pairs_batched_partial_fallback(monkeypatch):
    import asyncio
    client = FakeClient(_batch_payload(["factual", "pilot"]))
    fallbacks = []

    async def fake_gen(client_, chunk, t, index, related, semaphore, **kw):
        fallbacks.append(t)
        return {"chunk_id": "s:1", "type": t, "pair": {}, "problems": [],
                "attempts": 2}
    monkeypatch.setattr(synthesize, "generate_pair", fake_gen)
    monkeypatch.setattr(synthesize, "_check_pair", lambda i, p: [])
    sem = asyncio.Semaphore(1)
    chunk = synthesize.Chunk(id="s:1", text="t", doc="far", part="91",
                             chapter="1", section_id="91.113",
                             title=None, url="", n_tokens=5, split="train")
    recs = asyncio.run(synthesize.generate_pairs_batched(
        client, chunk, ["factual", "pilot", "abstention"], FakeIndex(), [], sem))
    assert [r["type"] for r in recs] == ["factual", "pilot", "abstention"]
    assert recs[0]["batched"] and recs[1]["batched"]
    assert "batched" not in recs[2]
    assert fallbacks == ["abstention"]


def test_generate_pairs_batched_api_error_falls_back_all(monkeypatch):
    import asyncio
    client = FakeClient(exc=RuntimeError("boom"))
    seen = []

    async def fake_gen(client_, chunk, t, index, related, semaphore, **kw):
        seen.append(t)
        return {"chunk_id": "s:1", "type": t, "pair": {}, "problems": [],
                "attempts": 1}
    monkeypatch.setattr(synthesize, "generate_pair", fake_gen)
    sem = asyncio.Semaphore(1)
    chunk = synthesize.Chunk(id="s:1", text="t", doc="far", part="91",
                             chapter="1", section_id="91.113",
                             title=None, url="", n_tokens=5, split="train")
    recs = asyncio.run(synthesize.generate_pairs_batched(
        client, chunk, ["factual", "pilot"], FakeIndex(), [], sem))
    assert len(recs) == 2 and seen == ["factual", "pilot"]
    assert all(not r.get("batched") for r in recs)


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
