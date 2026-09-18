# Demo 1 Overhaul — Implementation Plan

**Goal:** Turn `demo1_doc_intelligence` from a broken demo into a working, verifiable
compliance Q&A tool over IOGP / JIP33 documents.

**Architecture:** Move from "generic semantic search over text" to "structural
retrieval over a document model" — clause-level retrieval, document routing,
guaranteed citations, validated answer objects.

**Stack:** Streamlit · LangChain · ChromaDB · local sentence-transformers ·
Groq (OpenAI-compatible) · pytest

---

## Audit findings this plan addresses

| # | Finding | Evidence |
|---|---|---|
| 1 | UI crashes on every question | `app.py:610` passes `lang_override=` to `ask()` in `retrieval.py:172`, which has no such parameter — `TypeError` reproduced |
| 2 | Both configured Groq models are retired | `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` both absent from the account's live model list; live call returns `model_not_found` |
| 3 | `requirements.txt` cannot reproduce the app | `pdfplumber`, `langdetect`, `deep-translator`, `langchain-chroma`, `langchain-huggingface` imported but undeclared, and nothing pulls them transitively |
| 4 | Retrieval cannot find the answer page | Flagship question returns "not found"; plain vector search returns the table of contents (p3), not the rules page (p7) |
| 5 | Embeddings truncate most of the corpus | 253 chunks, 84% over the 256-token limit; **53% of all indexed text is invisible** to the embedding model |
| 6 | Citations are off by one | `ingest.py:66` stores `enumerate(pdf.pages)` — 0-indexed |
| 7 | Citations are generated, not guaranteed | The prompt *asks* the LLM to cite; nothing verifies the cited page supports the claim |

---

## Phase 0 — Make it run (~1 hour) ✅ COMPLETE

Exit criteria: clone → install → ingest → ask returns a cited answer without crashing.

- [x] 0.1 Branch `demo1-hardening`; save this plan
- [x] 0.2 Wire `data/` to the local assets (PDFs stay gitignored)
- [x] 0.3 Point at current Groq models + validate model IDs at startup
- [x] 0.4 Fix the `ask()` signature (temporary; removed in Phase 1)
- [x] 0.5 Harden `expand_query` parsing
- [x] 0.6 Complete `requirements.txt`
- [x] 0.7 1-based page numbers + re-ingest
- [x] 0.8 Smoke test + `.env.example`

### Phase 0 results

| Check | Result |
|---|---|
| Smoke tests | 6 passed (incl. live answer) |
| `requirements.txt` resolution | 147 packages resolve; all 5 previously-missing deps present |
| Model IDs | `openai/gpt-oss-20b` + `openai/gpt-oss-120b`; retired ID correctly detected with a readable message |
| Ingest from this checkout | 9 PDFs → 221 pages → 253 chunks, both stores built |
| Page numbering | min page = 1; page `0` no longer present |
| Flagship question | "What are the life saving rules?" → all nine rules, cited **459.pdf Page 8** (matches the document's own table of contents) |

Note: the `lang_override` parameter added in 0.4 is a deliberate stopgap — Phase 1
removes the language-detection/translation path entirely in favour of a
multilingual embedding model.

## Phase 1 — Make retrieval correct (~1 day)

Exit criteria: measured increase in page hit-rate over the Phase 1.1 baseline.

- [x] 1.1 Golden set + eval harness; record baseline **before** changing anything
- [ ] 1.2 Document model: Document → Section → Clause
- [ ] 1.3 Clause-level chunking with parent context prepended
- [ ] 1.4 Metadata: `doc_family`, `doc_role`, `section`, `clause_id`, `page`
- [ ] 1.5 Multilingual 512-token embedding model (chosen by eval, not assumption)
- [ ] 1.6 Hybrid retrieval: BM25 ∥ vector → RRF → cached cross-encoder rerank
- [ ] 1.7 Routing: filter by document family / role before similarity search
- [ ] 1.8 Re-run eval; gate at >85–90% page hit-rate
- [ ] 1.9 Remove `deep_translator`, `langdetect`, dual store, `lang_override`,
      and query expansion if the numbers prove it is unnecessary

### Phase 1.1 — baseline (measured 2026-09-18, before any change)

18 questions (12 English, 6 Bahasa); retrieval only, top-6 chunks.

| Variant | doc hit@6 | page hit@6 | EN page | ID page |
|---|---|---|---|---|
| Current pipeline (query expansion on) | 94% | **56%** | 83% | **0%** |
| Same pipeline, expansion off | 94% | **56%** | 75% | 17% |

**Target to beat: 56% page hit-rate (EN 83%, ID 0%).**

Two findings that change the plan:

1. **Query expansion buys nothing.** 56% either way, at the cost of an LLM call
   per question. It slightly helps English (83% vs 75%) and slightly hurts
   Bahasa (0% vs 17%) — both within noise at n=18. Decision for 1.9: remove it.
2. **The Bahasa claim is false in practice — 0% page hit-rate.** The
   translate-then-search path fails: `id-02` retrieved S-737 and S-719 pages for
   a confined-space question about IOGP 459. This is the strongest argument for
   1.5 (multilingual embeddings replacing the translation hop).

Document-level retrieval is already good (94%) — the problem is *ranking*, not
finding the right document. That is exactly what clause-level chunking (1.3),
hybrid search (1.6) and routing (1.7) are meant to fix.

## Phase 2 — Make answers verifiable (~1 day)

Exit criteria: zero fabricated citations; groundedness ~100% on the golden set.

- [ ] 2.1 Validated answer schema (Pydantic + structured output)
- [ ] 2.2 Citations attached by the retrieval layer; LLM may only select known `clause_id`s
- [ ] 2.3 Sufficiency check + bounded retry loop
- [ ] 2.4 Groundedness metric in the eval harness
- [ ] 2.5 UI shows verbatim quotes + clause references
- [ ] 2.6 `test_citations.py`, `test_sufficiency.py`

## Phase 3 — Make it durable (~1 day)

Exit criteria: CI green on a fresh clone; no model ID outside config; live demo link.

- [ ] 3.1 Provider-agnostic LLM layer with fallback chain
- [ ] 3.2 Split stages; `app.py` becomes UI only
- [ ] 3.3 Dependency hygiene (declared direct deps + lock)
- [ ] 3.4 GitHub Actions: smoke test + retrieval-only eval on every push
- [ ] 3.5 Remove dead code and the stale `data/vectorstore/`
- [ ] 3.6 Reconcile the two project copies; repo becomes canonical
- [ ] 3.7 Add one freely-licensed sample document (real PDFs stay out — copyright)
- [ ] 3.8 Update README / CLAUDE.md with real eval numbers
- [ ] 3.9 Push, deploy to Streamlit Cloud, add the live link

## Phase 4 (optional, only after 0–3)

TRS-vs-QRS comparison, requirement tables, export. Do not start before the
foundation is correct.

---

## Working agreement

- Stop at the end of each phase; show verification output; wait for approval.
- Commit at each phase boundary so any phase can be reverted independently.
- Assumptions: Bahasa Indonesia is kept (it simplifies the architecture), the repo
  is the single source of truth, and eval runs are quota-frugal (retrieval metrics
  use the small model; ~5 full answers use the large one).

## Errors encountered

| Error | Attempt | Resolution |
|---|---|---|
| `git checkout -b` — "Operation not permitted" writing `.git/refs` | 1 | `.git` is read-only in the sandbox; re-ran with escalated permissions |
