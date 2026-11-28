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
    """Sliding-window embeddings: long sections are embedded in overlapping
    ~500-char windows (title repeated in each) and window scores are max-
    pooled per doc at query time. Fixes the old [:1200] truncation, which
    made late text in long sections invisible to the dense channel."""
    global EMB_CACHE
    EMB_CACHE = Path(f"work/rag_corpus_winemb_{model_name.split('/')[-1]}.npz")
    key = str(hash(tuple(d["_id"] for d in docs)))
    if EMB_CACHE.is_file():
        z = np.load(EMB_CACHE, allow_pickle=True)
        if str(z.get("ids_hash", "")) == key:
            return z["wemb"], z["doc_idx"]
    print(f"window-embedding {len(docs)} corpus sections "
          f"({model_name.split('/')[-1]})...", flush=True)
    win_text, win_doc = [], []
    for i, d in enumerate(docs):
        head = f"{d['title']}: "
        body = d["text"]
        if len(body) <= 700:
            windows = [body]
        else:
            windows = [body[p:p + 600] for p in range(0, len(body) - 200, 400)]
        for w in windows:
            win_text.append(head + w[:700])
            win_doc.append(i)
    wemb = embed(win_text, model_name)
    EMB_CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(EMB_CACHE, wemb=wemb,
                        doc_idx=np.asarray(win_doc, dtype=np.int32),
                        ids_hash=key)
    return wemb, np.asarray(win_doc, dtype=np.int32)


def doc_sims(qemb, wemb, doc_idx, n_docs):
    """max-pool window cosine sims into a (n_queries, n_docs) matrix."""
    ws = qemb @ wemb.T
    out = np.full((qemb.shape[0], n_docs), -1.0, dtype=np.float32)
    for wi, di in enumerate(doc_idx):
        np.maximum.at(out[:, di], np.arange(qemb.shape[0]), ws[:, wi])
    return out


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

HYDE_PERSPECTIVES = [
    ("Write 60-100 words of authoritative regulatory text answering this "
     "question, in the style of the FAR/AIM (state the rule and its "
     "conditions; no preamble, no disclaimers):", 0.3),
    ("List the 5-10 key regulatory terms, section-style phrases, and "
     "definitions an FAA inspector would use when answering this question. "
     "Comma-separated, no sentences:\n", 0.7),
    ("Restate this pilot's question in formal regulatory language, the way "
     "a designated pilot examiner would phrase it on a checkride oral, "
     "including the regulation topic it falls under:\n", 0.7),
]



# ------------------------------------------------------------------ BM25
_TOK = re.compile(r"[a-z0-9]+(?:\.\d+)?")
BM25_CACHE = Path("work/rag_bm25.pkl")


class BM25:
    """Tiny inverted-index BM25 (Okapi formula). Section numbers survive
    tokenization as single tokens ('105.15'), which is exactly where pure
    dense retrieval is weakest."""

    def __init__(self, texts, k1=1.4, b=0.4):
        import collections
        import math
        self.k1, self.b = k1, b
        toks = [_TOK.findall(t.lower()) for t in texts]
        self.N = len(toks)
        self.avgdl = sum(len(t) for t in toks) / max(1, self.N)
        self.postings = {}
        for i, t in enumerate(toks):
            for term, f in collections.Counter(t).items():
                self.postings.setdefault(term, []).append((i, f))
        self.idf = {}
        for term, pl in self.postings.items():
            df = len(pl)
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
        self.dl = [len(t) for t in toks]

    def scores(self, query):
        import numpy as np
        s = np.zeros(self.N)
        for term in set(_TOK.findall(query.lower())):
            pl = self.postings.get(term)
            if not pl:
                continue
            idf = self.idf[term]
            for i, f in pl:
                s[i] += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s


def bm25_index(docs):
    import pickle
    key = str(hash(tuple(d["_id"] for d in docs)))
    if BM25_CACHE.is_file():
        with BM25_CACHE.open("rb") as fh:
            cached = pickle.load(fh)
        if cached.get("key") == key:
            return cached["bm25"]
    bm = BM25([f"{d['title']} {d['title']} {d['text']}" for d in docs])
    with BM25_CACHE.open("wb") as fh:
        pickle.dump({"key": key, "bm25": bm}, fh)
    return bm


def rrf_fusion(dense_sids, bm25_sids, k, bm25_weight=0.7):
    """Weighted reciprocal-rank fusion of two unique-section-id rankings.
    The lexical channel gets a lower weight: it rescues sections dense
    misses (exact-number matches) but is too weak to lead the top ranks."""
    score = {}
    for ranking, w in ((dense_sids, 1.0), (bm25_sids, bm25_weight)):
        for rank, sid in enumerate(ranking):
            score[sid] = score.get(sid, 0.0) + w / (60 + rank)
    return [sid for sid, _ in sorted(score.items(), key=lambda kv: -kv[1])[:k]]


def mention_ids(query, index):
    """Sections explicitly named in the question are pinned into the set."""
    out = []
    for rx in _MENTION_RES:
        for m in rx.finditer(query):
            rec = resolve_section_id(m.group(1), index)
            if rec and rec["section_id"] not in out:
                out.append(rec["section_id"])
    return out


def rerank(client, model, query, cands, k=4):
    """Listwise rerank: served model picks which candidates to read."""
    if len(cands) <= k:
        return cands
    listing = "\n".join(
        f"[{i}] {d['title']}: {d['text'][:220]}" for i, d in enumerate(cands))
    prompt = ("You select passages for a regulation-question answering "
              "system. Question:\n" + query + "\n\nCandidates:\n" + listing +
              f"\n\nReturn the numbers (comma-separated, up to {k}) of the "
              "candidates most likely to CONTAIN the exact rule text needed "
              "to answer. No explanation.")
    out = ask(client, model, [{"role": "user", "content": prompt}],
              temperature=0.0)
    picks = [int(x) for x in re.findall(r"\d+", out) if int(x) < len(cands)]
    chosen = [cands[i] for i in picks]
    for d in cands:                      # pad with fused order if under k
        if d not in chosen:
            chosen.append(d)
        if len(chosen) == k:
            break
    return chosen[:k]


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
    p.add_argument("--no-hybrid", dest="hybrid", action="store_false")
    p.add_argument("--embed-model", default=DEFAULT_EMBED)
    p.add_argument("--no-hyde", dest="hyde", action="store_false",
                   default=True,
                   help="disable hypothetical-document expansion (default "
                        "on: the served model drafts regulatory-style text "
                        "per query, embedded and max-fused with the query "
                        "ranking -- closes the colloquial-vs-regulatory "
                        "vocabulary gap; measured pool recall@16 84->92%)")
    p.add_argument("--hyde-n", type=int, default=3, choices=(1, 2, 3),
                   help="number of drafting perspectives fused into the "
                        "dense ranking (measured: n=3 lifts recall@4 "
                        "77->83%% over n=1; the extra drafts cost only "
                        "served calls, ~1s/query at concurrency 12)")
    p.add_argument("--bm25-weight", type=float, default=0.7,
                   help="RRF weight for the lexical channel; only applied "
                        "when hyde is off (windowed+hyde dense superseded "
                        "what bm25 used to rescue)")
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
    wemb, doc_idx = corpus_embeddings(docs, args.embed_model)
    verifier = Verifier(index)
    client = OpenAI(base_url=args.url, api_key="none", timeout=300)

    queries = [it.get("question") or it.get("acs_text") for it in items]
    print(f"embedding {len(queries)} queries...", flush=True)
    qemb = embed([QUERY_PREFIX + q for q in queries], args.embed_model)
    sims = doc_sims(qemb, wemb, doc_idx, len(docs))

    if args.hyde:
        n_p = args.hyde_n
        print(f"drafting {len(queries)}x{n_p} hypothetical documents...",
              flush=True)

        def _draft(job):
            i, k = job
            instr, temp = HYDE_PERSPECTIVES[k]
            try:
                return ask(client, args.model, [{"role": "user", "content":
                    instr + queries[i]}], temperature=temp)
            except Exception:
                return queries[i]
        jobs = [(i, k) for i in range(len(queries)) for k in range(n_p)]
        with ThreadPoolExecutor(max_workers=12) as ex:
            drafts = dict(zip(jobs, ex.map(_draft, jobs)))
        for k in range(n_p):
            texts = [drafts[(i, k)] for i in range(len(queries))]
            hemb = embed(texts, args.embed_model)      # doc side: no prefix
            sims = np.maximum(sims, doc_sims(hemb, wemb, doc_idx, len(docs)))
        print(f"hyde fused ({n_p} perspectives)", flush=True)

    sid2idx = {}
    for j, d in enumerate(docs):
        sid2idx.setdefault(d["section_id"], j)
    bm = bm25_index(docs) if (args.hybrid and not args.hyde
                              and args.bm25_weight > 0) else None
    if bm is not None:
        print(f"bm25 ready over {bm.N} docs", flush=True)

    def fused_order(i):
        """Doc indices in fused (dense [ + bm25 ]) ranking order."""
        dense_rank = [int(j) for j in np.argsort(-sims[i])]
        if bm is None:
            return dense_rank
        bm25_rank = [int(j) for j in np.argsort(-bm.scores(queries[i]))]

        def uniq_sids(idxs):
            seen, out = set(), []
            for j in idxs:
                sid = docs[j]["section_id"]
                if sid not in seen:
                    seen.add(sid)
                    out.append(sid)
            return out
        fused = rrf_fusion(uniq_sids(dense_rank), uniq_sids(bm25_rank),
                           args.pool * 3)
        return [sid2idx[s] for s in fused if s in sid2idx]

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
        for j in fused_order(i):
            d = docs[j]
            if d["section_id"] in seen:
                continue
            seen.add(d["section_id"])
            cands.append(d)
            if len(cands) >= args.pool:
                break
        hits = (rerank(client, args.model, q, cands, args.k) if args.rerank
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
