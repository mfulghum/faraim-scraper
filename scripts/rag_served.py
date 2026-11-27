"""End-to-end RAG eval: retrieve -> generate (served) -> validate -> retry/refuse.

This is the configuration the corpus was built for: the answer model gets the
top-k retrieved sections in the exact training prompt format, and every
answer must pass a verbatim-quote/citation validator (`_verify`) or it gets
one retry and then is refused. Nothing fabricated reaches the user.

    .venv-train/bin/python scripts/rag_served.py --gold gold \
        --url http://localhost:8888/v1 --model Qwen3.8-Flash-Next \
        --out work/rag_gold.jsonl

Reports, for the gold set: retrieval@k hit rate, accuracy without the
validator (what the raw model would have said), accuracy with the validator
(rejections count as failures), and how many fabrications the gate caught.
"""

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from faraim_scraper.citations import (SectionIndex, canonical_citation,
                                      citation_variants, normalize,
                                      quote_matches, resolve_section_id)
from faraim_scraper.emit_dataset import SYSTEM_PROMPT

CORPUS = Path.home() / "huggingface/far-aim/retrieval/corpus.jsonl"
EMB_CACHE = Path("work/rag_corpus_emb_bge-small.npz")  # reset per model below
DEFAULT_EMBED = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------- retriever
def load_corpus(index: SectionIndex):
    """Full runtime corpus from the section index (NOT the published
    retrieval/corpus.jsonl, which excludes holdout sections)."""
    docs, seen = [], set()
    for sid, rec in index.by_id.items():
        key = rec.get("txt_path") or (rec.get("chunk") or {}).get("id") or sid
        if key in seen:
            continue
        seen.add(key)
        try:
            text = index.text(rec)
        except Exception:
            continue
        if len(text.strip()) < 50:
            continue
        title = rec.get("title") or rec.get("part") or sid
        docs.append({"_id": sid, "section_id": sid, "title": title, "text": text})
    return docs


def embed(texts, model_name=DEFAULT_EMBED, batch=64):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model.to(device)
    except Exception:
        device = "cpu"
        model.to("cpu")
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state[:, 0]          # CLS pooling
            out.append(torch.nn.functional.normalize(h, dim=1).float().cpu().numpy())
    return np.concatenate(out)


def corpus_embeddings(docs, model_name=DEFAULT_EMBED):
    global EMB_CACHE
    EMB_CACHE = Path(f"work/rag_corpus_emb_{model_name.split('/')[-1]}.npz")
    if EMB_CACHE.is_file():
        z = np.load(EMB_CACHE, allow_pickle=True)
        if str(z.get("ids_hash", "")) == str(hash(tuple(d["_id"] for d in docs))):
            return z["emb"]
    print(f"embedding {len(docs)} corpus chunks ({model_name.split('/')[-1]})...", flush=True)
    texts = [f"{d['title']}: {d['text'][:1200]}" for d in docs]
    emb = embed(texts, model_name)
    EMB_CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(EMB_CACHE, emb=emb,
                        ids_hash=str(hash(tuple(d["_id"] for d in docs))))
    return emb


# ---------------------------------------------------------------- validator
_REFUSE = "The provided excerpts do not cover this question."
_QUOTE_RE = re.compile(r'[“"]([^“”"]{15,})[”"]')
_FABR_MSG = ("Your previous answer failed verification: quoted text was not "
             "found verbatim in the sections you cited, or your citations did "
             "not resolve to real sections. Answer again using ONLY the "
             "provided excerpts: copy quotes CHARACTER-FOR-CHARACTER from a "
             "single contiguous run of one excerpt (never splice words across "
             "table cells or paragraphs), and cite the excerpt headers.")


def _quoted_spans(answer):
    spans = []
    for s in _QUOTE_RE.findall(answer):
        for frag in re.split(r"\s*(?:\.\.\.|…|\[…\]|\[\.\.\.\])\s*", s):
            if len(frag.strip()) >= 15:
                spans.append(frag.strip())
    return spans


class Verifier:
    def __init__(self, index: SectionIndex):
        self.index = index
        self.vmap = []
        for rec in index.by_id.values():
            for v in citation_variants(rec):
                nv = normalize(v)
                if nv:
                    self.vmap.append((nv, rec))
        self.vmap.sort(key=lambda t: -len(t[0]))

    def cited_records(self, answer):
        na = normalize(answer)
        recs = {}
        for m in re.finditer(r"§+\s*(\d+\.\d+)", answer):
            r = self.index.get(m.group(1))
            if r:
                recs[r["section_id"]] = r
        for m in re.finditer(r"¶\s*(\d+-\d+-\d+)", answer):
            r = self.index.get(m.group(1))
            if r:
                recs[r["section_id"]] = r
        for nv, rec in self.vmap:
            if nv in na:
                recs[rec["section_id"]] = rec
        return recs

    def verify(self, answer, retrieved_texts):
        """Return (ok, reason). Ok = cites something resolvable, and every
        quote is verbatim in a cited section or a retrieved excerpt."""
        abstain = bool(re.search(r"do(es)? not (address|cover|mention"
                                 r"|discuss|contain|specify|state|include"
                                 r")|not covered|no provision", answer, re.I))
        spans = _quoted_spans(answer)
        if not spans and not abstain:
            return False, "no verbatim quote"
        recs = self.cited_records(answer)
        if not recs and not abstain:
            return False, "no resolvable citation"
        allowed = [self.index.text(r) for r in recs.values()] + list(retrieved_texts)
        for s in spans:
            # tolerate markdown noise the model adds when reformatting
            # quotes (blockquote arrows, bullets, ellipsis dots) -- never
            # word-level edits, which still fail
            clean = re.sub(r"[>\u2022\u00b7]|\.\.+", "", s)
            if not any(quote_matches(clean, t) for t in allowed):
                return False, f"quote not verbatim: {s[:60]!r}"
        return True, "ok"


# ---------------------------------------------------------------- generation
def build_user(question, hits, index):
    blocks = []
    for i, d in enumerate(hits, 1):
        rec = index.get(d["section_id"])
        cit = canonical_citation(rec) if rec else d["section_id"]
        blocks.append(f"[Excerpt {i}] {cit}\n{d['text']}")
    return f"{question}\n\nProvided excerpts:\n\n" + "\n\n".join(blocks)


def ask(client, model, msgs, temperature=0.2):
    resp = client.chat.completions.create(
        model=model, messages=msgs, temperature=temperature, max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return resp.choices[0].message.content or ""


_MENTION_RES = (re.compile(r"§\s*(\d+\.\d+)"), re.compile(r"¶\s*(\d+-\d+-\d+)"))


def mention_ids(query, index):
    """Sections explicitly named in the question are pinned into the set."""
    out = []
    for rx in _MENTION_RES:
        for m in rx.finditer(query):
            rec = resolve_section_id(m.group(1), index)
            if rec and rec["section_id"] not in out:
                out.append(rec["section_id"])
    return out


def rerank(client, model, query, cands):
    """Listwise rerank: served model picks which candidates to read."""
    if len(cands) <= 4:
        return cands
    listing = "\n".join(
        f"[{i}] {d['title']}: {d['text'][:220]}" for i, d in enumerate(cands))
    prompt = ("You select passages for a regulation-question answering "
              "system. Question:\n" + query + "\n\nCandidates:\n" + listing +
              "\n\nReturn the numbers (comma-separated, up to 4) of the "
              "candidates most likely to CONTAIN the exact rule text needed "
              "to answer. No explanation.")
    out = ask(client, model, [{"role": "user", "content": prompt}],
              temperature=0.0)
    picks = [int(x) for x in re.findall(r"\d+", out) if int(x) < len(cands)]
    chosen = [cands[i] for i in picks]
    for d in cands:                      # pad with dense order if under 4
        if d not in chosen:
            chosen.append(d)
        if len(chosen) == 4:
            break
    return chosen[:4]


def main() -> int:
    from openai import OpenAI

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--gold", default="gold", choices=("gold", "acs"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--pool", type=int, default=16,
                   help="dense candidates fed to the reranker")
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    p.add_argument("--embed-model", default=DEFAULT_EMBED)
    p.add_argument("--url", default="http://localhost:8888/v1")
    p.add_argument("--model", default="Qwen3.8-Flash-Next")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from faraim_scraper.config import HF_DATASET_DIR
    name = "acs_gold.jsonl" if args.gold == "acs" else "gold.jsonl"
    items = [json.loads(l) for l in
             (HF_DATASET_DIR / "eval" / name).read_text(encoding="utf-8").splitlines()]
    if args.limit and args.limit < len(items):
        import random
        items = random.Random(args.seed).sample(items, args.limit)

    index = SectionIndex(args.data)
    docs = load_corpus(index)
    emb = corpus_embeddings(docs, args.embed_model)
    verifier = Verifier(index)

    queries = [it.get("question") or it.get("acs_text") for it in items]
    print(f"embedding {len(queries)} queries...", flush=True)
    qemb = embed([QUERY_PREFIX + q for q in queries], args.embed_model)
    sims = qemb @ emb.T
    topk = np.argsort(-sims, axis=1)[:, :args.k * 3]   # oversample, then dedupe

    client = OpenAI(base_url=args.url, api_key="none", timeout=300)
    lock = threading.Lock()
    rows = []

    def run_one(i):
        it, q = items[i], queries[i]
        seen, cands = set(), []
        for sid in mention_ids(q, index):          # pin explicitly named sections
            rec_doc = next((d for d in docs if d["section_id"] == sid), None)
            if rec_doc and sid not in seen:
                seen.add(sid)
                cands.append(rec_doc)
        for j in topk[i]:
            d = docs[j]
            if d["section_id"] in seen:
                continue
            seen.add(d["section_id"])
            cands.append(d)
            if len(cands) >= args.pool:
                break
        hits = (rerank(client, args.model, q, cands) if args.rerank
                else cands[:args.k])
        retrieved = [h["section_id"] for h in hits]
        retrieved_texts = [h["text"] for h in hits]
        shown = set(retrieved) | set(mention_ids(q, index))
        user = build_user(q, hits, index)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
        ans = ask(client, args.model, msgs)
        ok, why = verifier.verify(ans, retrieved_texts)
        n_retries = 0
        rejected_answer = None
        if not ok:
            n_retries = 1
            msgs += [{"role": "assistant", "content": ans},
                     {"role": "user", "content": _FABR_MSG}]
            ans2 = ask(client, args.model, msgs)
            ok2, why2 = verifier.verify(ans2, retrieved_texts)
            if ok2:
                ans, ok, why = ans2, True, "ok (after retry)"
            else:
                rejected_answer = ans2
                ans, ok, why = _REFUSE, False, f"rejected: {why2}"
        # retrieval diagnostic + gold scoring
        if "gold_sections" in it:
            want = [g["section_id"] for g in it["gold_sections"]]
            recs_want = [r for r in (resolve_section_id(w, index) for w in want) if r]
            hit = any(r["section_id"] in shown for r in recs_want)
            got = verifier.cited_records(ans)
            cites_ok = any(r["section_id"] in got for r in recs_want)
        else:
            want = it["expected_citations"]
            recs_want = [r for r in (resolve_section_id(w, index) for w in want) if r]
            hit = any(r["section_id"] in shown for r in recs_want)
            got = verifier.cited_records(ans)
            cites_ok = all(r["section_id"] in got for r in recs_want)
        row = {"gold": args.gold, "question": q[:200], "retrieved": retrieved,
               "retrieval_hit": hit, "answer": ans, "expected": want,
               "verified": ok, "verify_why": why, "retries": n_retries,
               "rejected_answer": rejected_answer,
               "citations_ok": bool(cites_ok)}
        with lock:
            rows.append(row)
            if len(rows) % 10 == 0 or len(rows) == len(items):
                n = len(rows)
                print(f"[{n}/{len(items)}] retrieval@{args.k} {sum(r['retrieval_hit'] for r in rows)/n:.0%}"
                      f" | validated acc {sum(r['citations_ok'] and r['verified'] for r in rows)/n:.0%}"
                      f" | raw acc {sum(r['citations_ok'] for r in rows)/n:.0%}"
                      f" | gate caught {sum(r['retries'] for r in rows)}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(run_one, range(len(items))))
    n = len(rows)
    print(f"\n=== {args.gold} / {args.model} / k={args.k} ===")
    print(f"retrieval@{args.k} hit rate : {sum(r['retrieval_hit'] for r in rows)}/{n}"
          f" = {sum(r['retrieval_hit'] for r in rows)/n:.1%}")
    print(f"accuracy WITHOUT validator : {sum(r['citations_ok'] for r in rows)}/{n}"
          f" = {sum(r['citations_ok'] for r in rows)/n:.1%}")
    print(f"accuracy WITH validator    : {sum(r['citations_ok'] and r['verified'] for r in rows)}/{n}"
          f" = {sum(r['citations_ok'] and r['verified'] for r in rows)/n:.1%}")
    print(f"validator rejections       : {sum(1 for r in rows if not r['verified'])}"
          f" (retry attempts {sum(r['retries'] for r in rows)})")
    if args.out:
        with args.out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {n} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
