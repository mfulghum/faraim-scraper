# Evaluation results — faraim-27b LoRA merge (2026-11-27)

Model under test: `out/faraim-27b-merged` (Qwen3.8-27B base + LoRA r16/a32,
3 epochs @ lr 1e-4, completion-only on `sft_closed_book`, 6,840 steps).
Scorer: `scripts/eval_offline.py` (same prompts + citation validator as
`faraim_scraper.eval_gold`; thinking disabled via `enable_thinking:false`).

| Model | Set | Mode | Result | Rows |
|---|---|---|---|---|
| merged | gold (120) | closed | 21/120 = 17.5% | work/eval_merged_offline2.jsonl |
| base   | gold (120) | closed | 22/120 = 18.3% | work/eval_base_closed.jsonl |
| merged | gold (120) | oracle | 115/120 = 95.8% | work/eval_merged_oracle.jsonl |
| base   | gold (120) | oracle | **118/120 = 98.3%** | work/eval_base_oracle.jsonl |
| merged | acs_gold (300 sample, seed 7) | closed | 27/300 = 9.0% hit-any, mean coverage 4.0% | work/eval_merged_acs.jsonl |

Baselines: gold reference answers self-score 120/120 (scorer ceiling);
base model 3-question spot checks: closed 0/3, oracle 3/3.

## Reading (updated after base oracle control)

Base oracle 98.3% vs merged 95.8%: a 3-question gap on n=120 (~1.5 sigma)
— statistically indistinguishable, and the set is saturated for both.
Deeper checks on the same runs (quote extraction + citation resolution):

| measure (oracle, n=120) | merged | base |
|---|---|---|
| quote spans verbatim in gold excerpt or a cited section | 137/185 = 74.1% | 187/257 = 72.8% |
| answers whose quotes are ALL verbatim | 69 | 68 |
| cited-but-nonexistent sections | 6 | 6 |

No measurable axis separates LoRA from base in oracle mode. Mechanistic
explanation: the teacher (Flash-Next) is a distilled variant of the same
Qwen3.8 lineage, so SFT could not inject knowledge the base lacked; at best
it reinforces the contract.

### Distractor test (adversarial context, same 120 rows, seed 7)

| model | refusal rate |
|---|---|
| merged | 120/120 = 100% |
| base   | 120/120 = 100% |

Both refuse cleanly and both quote what the distractor excerpt *does*
cover (119/120 base answers cite the distractor's own section) — the
base model's instruction-following already implements the abstention
contract that the 3,972 SFT abstention pairs were meant to teach.

## Verdict

Four independent axes (closed citation, oracle citation, quote fidelity,
distractor abstention): base Qwen3.8-27B is statistically indistinguishable
from the LoRA merge. The fine-tune is retired from the serving path.

What this project actually produced (and what to ship):
1. **The dataset** (42,680 validator-enforced verbatim-cited pairs + ACS
   crosswalk + retrieval index + gold sets) — published at
   huggingface.co/datasets/mfulghum/far-aim.
2. **The citation validator** (citations.SectionIndex) — enforce at serving
   time: reject/regenerate any answer whose quotes are not verbatim in a
   resolvable cited section. This is the guard the eval exposed as missing
   (both models fabricate quotes when answering from memory, e.g. the
   "15 minutes before" §101.33 confabulation; the validator catches it).
3. **The eval suite** (eval_gold.py + scripts/eval_offline.py incl.
   --mode distractor) — reusable for any future checkpoint, including a
   better-teacher LoRA should one exist.

Serving recommendation: retriever + base Qwen3.8-27B (or the already-served
Flash-Next, same lineage) + the SYSTEM_PROMPT contract + serving-time quote
validation. The merge (out/faraim-27b-merged, 51 GB) is kept as an artifact;
delete at the user's discretion.

## End-to-end validated RAG (scripts/rag_served.py, 2026-11-27)

Retrieval (bge-base dense over the full 9,781-section index + LLM listwise
rerank + query-mention pinning) -> generation (served Flash-Next, thinking
off, training-format prompt) -> verbatim-quote/citation gate with one
retry; anything unverified becomes a clean refusal.

Gold set (120), `work/rag_gold_final.jsonl`:

| measure | value |
|---|---|
| retrieval@4 hit rate | 90/120 = 75.0% |
| validated accuracy | 74/120 = 61.7% |
| precision on shipped answers | 74/96 = 77.1% |
| fabricated quotes shipped | 0 (gate-enforced) |
| clean refusals | 24/120 = 20.0% |
| gate: retry repairs | 29/53 first-pass failures repaired |
| gate: false-reject cost | 16 refusals would have cited correctly |
| validated acc when retrieval hits | 74/90 = 82.2% |

vs the same model WITHOUT validation: ~49-62% raw accuracy with the
remainder confident fabrications ("15 minutes before the operation",
§101.33(b)(1), invented handbook quotes). The gate converts hallucinations
into refusals -- that is the reliability the fine-tune failed to deliver,
manufactured at inference instead.

Remaining headroom (in order of expected payoff): retrieval recall (pool
recall 80%; hybrid BM25+dense or a larger embedder -> ceiling ~75-85%
overall), embedder upgrade, k=6, few-shot exemplars in the prompt for
table-heavy AIM sections (the dominant false-reject source).

## Reading

1. **Oracle mode is the metric that matters.** 95.8% shows the fine-tune
   learned its actual job: extract, quote verbatim, and cite the governing
   text when given the excerpt. The 5 rejects quote correct governing text
   but cite an adjacent paragraph or miss a second required citation
   (3-2-1 vs 4-1-15; 105.25 without 105.17; 3-4-9 x2; 3-5-10) — citation-
   string misses, not content hallucinations. True ceiling ~96-98%.

2. **Closed-book scores do not discriminate** merged (17.5%) vs base
   (18.3%) — the LoRA adds nothing to verbatim recall, as expected: 42k
   verbatim-cited pairs cannot be memorized into 27B weights. Also
   21/120 gold questions leak the expected citation into the question text
   ("Under 14 CFR § 105.1, ..."), so the ~18% is largely echo-of-prompt.
   ACS closed (9%) has no such leaks and measures pure parametric recall.

3. **No retraining on account of the closed numbers.** Eval loss bottomed
   ~epoch 1 (0.63) and rose to 0.688; rerunning 2 epochs @ 5e-5 will not
   move closed-book citation accuracy — that is a model-capacity/task
   limit, not a recipe artifact. Deployment contract: answers must be
   retrieval-grounded (oracle-mode behavior), and any closed-book answer
   should be gated by the same verbatim quote-in-section check the data
   pipeline enforces (`SectionIndex.validate_pair` / `quote_in_section`).

## Artifacts
- `work/eval_merged_offline.jsonl` — first closed run (thinking ON, preamble;
  superseded by eval_merged_offline2.jsonl)
- ACS coverage histogram (300 sample): 273 zeros, 5 full, remainder partial.

## Open items
- base oracle control (command in git log of this file / conversation)
- publish dataset (currently private) — user's call
- quantize merge (llmcompressor compressed-tensors) for TensorFold serving,
  then served eval_gold with --url http://localhost:8888/v1 --model faraim-27b
