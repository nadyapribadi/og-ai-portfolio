# Demo 1 — O&G Document Intelligence

RAG chatbot over public upstream O&G documents.
English and Bahasa Indonesia supported.

---

## Status

Measured on an 18-question golden set (12 English, 6 Bahasa Indonesia),
retrieval only, `eval/run_eval.py`:

| Metric | Result |
|---|---|
| Right document retrieved | 100% |
| Right page retrieved (@6 / @12) | 83% / 94% |
| English pages | 100% |
| Bahasa Indonesia pages | 83% |
| Citation validity, verified answers | 100% (14 claims) |

Two rerankers, one pipeline. The mmarco cross-encoder is the better ranker and
the more expensive neighbour (~310 MB resident), so the app asks its container
what it can afford and picks:

| Mode | When | Right page | Page MRR |
|---|---|---|---|
| mmarco cross-encoder | ≥1.4 GB available (laptop, larger host) | 94% @12 | 0.618 |
| late interaction | smaller container (Streamlit's floor is 690 MB) | 89% @20 | 0.531 |

Both run locally on ONNX Runtime and need no torch. Pin either with
`DEMO1_RERANK=on|off` if you want to reproduce the numbers.

One question out of 18 still misses: the Bahasa Indonesian phrasing of "what
are the life saving rules?" retrieves the right document but the foreword page
rather than the page that lists the rules. Everything else lands.

| File | What it does |
|------|--------------|
| `parse.py` | PDF/markdown → Document → Section → Clause |
| `ingest.py` | Clauses → page-bounded chunks → Chroma |
| `retrieval.py` | route → BM25 + vector → RRF → rerank |
| `onnx_models.py` | the embedding and reranking models, on ONNX Runtime |
| `memory.py` | what the container will let the process load |
| `app.css` | the interface layer: metrics and one motif, no colours of its own |
| `DESIGN.md` | the interface direction the layer implements |
| `answer.py` | schema-constrained answers with verified citations |
| `llm.py` | provider-agnostic chat models with a fallback chain |
| `app.py` | Streamlit UI |

The document set is configurable: point `DEMO1_DOCS_DIR` at another folder, or
drop PDFs into `data/raw_docs/` (which then take precedence over the sample).

---

Questions, capability cards and the subtitle follow whichever corpus is
indexed: the full pack gets the IOGP/JIP33 questions and cards, the bundled
sample gets its own. CI asserts that every question the sample corpus offers
retrieves its own answer.

## In Scope

- Text-based PDFs (digitally created, not scanned)
- Technical documents — standards, specifications, procedures, guidelines
- English and Bahasa Indonesia
- Up to ~50 documents, ~500 pages total recommended

## Out of Scope

- Scanned PDFs — pdfplumber extracts nothing, all answers will be "not found"
- Image-heavy pages — engineering drawings, P&IDs, charts, figures
- Questions requiring cross-document comparison ("compare X in doc A with Y in doc B")
- Real-time data or calculations
- Questions where the answer is only in a table spanning multiple pages

---

## Document Pack (default)

| File | Content |
|------|---------|
| 459.pdf | IOGP 459 — Life-Saving Rules |
| 456.pdf | IOGP 456 — Process Safety KPIs |
| S-737v2026-03 TRS.pdf | JIP33 S-737 Deluge Skids — Technical |
| S-737Qv2026-03 QRS.pdf | JIP33 S-737 Deluge Skids — Quality |
| S-717v2025-03 TRS.pdf | JIP33 S-717 Noise Equipment — Technical |
| S-717Qv2020-06 QRS.pdf | JIP33 S-717 Noise Equipment — Quality |
| S-719v2025-01 TRS.pdf | JIP33 S-719 Water Mist — Technical |
| S-719Qv2025-01 QRS.pdf | JIP33 S-719 Water Mist — Quality |
| S-719Jv2025-01 TRS with Justification.pdf | JIP33 S-719 — Justification |

Replace or add any text-based PDFs in `data/raw_docs/` and re-run ingest.

---

## Verified Questions (all passing)

**English:**
- "What are the life saving rules?"
- "What are the 9 IOGP life saving rules?"
- "What must I confirm before entering a confined space?"
- "What are the hot work requirements in a hazardous area?"
- "What is the difference between Tier 1 and Tier 2 process safety events?"
- "How are process safety KPIs measured?"
- "What does LOPC stand for and what are its consequences?"
- "What standards does S-737 reference for electrical installations?"
- "What must workers do before walking under a suspended load?"
- "What does IOGP S-737 specify for deluge skid design?"

**Bahasa Indonesia:**
- "Apa saja aturan keselamatan jiwa?"
- "Apa yang harus dilakukan sebelum memasuki ruang tertutup?"
- "Apa perbedaan antara kejadian keselamatan proses Tier 1 dan Tier 2?"
- "Apa persyaratan desain deluge skid menurut IOGP S-737?"

---

## Architecture

```
PDFs → pdfplumber (text + table markdown)
     → strip running headers, drop the table of contents
     → Document → Section → Clause
     → one chunk per clause, never crossing a page break
     → multilingual-e5-small embeddings (int8 ONNX, local)
     → ChromaDB vectorstore

Query:
Question → route by document family / role
         → BM25 ∥ vector search → reciprocal rank fusion
         → rerank (mmarco cross-encoder, or late interaction on a tight host)
         → Groq → claims schema → every citation verified against the excerpt
```

**Key decisions:**
- Chunk size is derived from the embedding model's own token window, not chosen
- Multilingual by embedding, not by translation — a Bahasa question searches the
  English corpus natively, with no language detection to get wrong
- Citations are verified, not requested: a claim that quotes text not in the
  retrieved excerpt is discarded before the user sees it. A clause id only has
  to appear in the excerpt, so a chunk labelled §3.1.1 can be cited as §3.2.1
  when that is the clause the quoted sentence belongs to
- Documents never share a clause: a document boundary closes the running clause,
  so no chunk carries the next document's text under the previous document's name
- Models run on ONNX Runtime, not torch — see `onnx_models.py` for why

---

## Tips

- Specific questions get better answers than vague ones
  - ✅ "What are the confined space entry requirements?"
  - ⚠️ "Tell me about confined spaces"
- If you get "not found", try rephrasing with different vocabulary
- The system works best with documents that have clear structure — headings, numbered sections
- Language is auto-detected — just type in English or Bahasa Indonesia naturally

---

## Customizing for Your Own Documents

If you replace the default documents with your own, edit `config.py` — no changes needed in `app.py`.

`demo1_doc_intelligence/src/config.py` contains:
- `APP_TITLE` and `APP_SUBTITLE` — app branding
- `DOCUMENT_SOURCES` — list shown in sidebar
- `SAMPLE_QUESTIONS` — clickable buttons by category
- `CAPABILITY_CARDS` — cards shown on first load
- `SOURCE_FRIENDLY` — friendly display names for PDF filenames

Replace the values with content relevant to your documents. The app reads from this file at startup.

---

## Verify Your Index

After running ingest, confirm your documents were indexed correctly:

```bash
python3 -c "
from langchain_chroma import Chroma
import sys; sys.path.insert(0, 'demo1_doc_intelligence/src')
from embeddings import build_embeddings
vs = Chroma(
    collection_name='og_docs',
    embedding_function=build_embeddings(),
    persist_directory='demo1_doc_intelligence/data/vectorstore_en'
)
print(f'Chunks indexed: {vs._collection.count()}')
results = vs.similarity_search('what is this document about', k=3)
for r in results:
    print(r.metadata.get('source_file'), 'page', r.metadata.get('page'))
    print(r.page_content[:150])
    print()
"
```

**What to check:**
- Chunk count should be roughly 20-30 chunks per document
- Content preview should show readable text — not garbled characters
- If chunk count is 0 or very low → likely scanned PDFs (see troubleshooting)

---

## Troubleshooting

**"Not found" for everything**
Your PDFs are likely scanned. Verify with:
```bash
pdftotext yourfile.pdf - | head -20
```
If output is empty or garbled, the PDF has no text layer.
Solution: use OCR software to convert scanned PDFs to text-based PDFs before ingest.

**Chunk count is very low after ingest**
Same issue — pdfplumber extracted nothing from scanned pages.

**Rate limit error in the app**
The app runs on a free Groq key: a daily token budget and a per-minute request
limit. The provider resets those on rolling windows, not at midnight, so waiting
a few minutes usually clears it. The app's own quota messages ("Session limit
reached", "Demo quota used for today") are separate: they are local limits that
keep one visitor from spending the whole day's budget, and they are configured
in `config.py` (`MAX_ANSWERS_PER_SESSION`, `MAX_ANSWERS_PER_DAY`,
`MIN_SECONDS_BETWEEN_ANSWERS`) or through the `DEMO1_*` environment variables.

To lift them for local work, where the key is yours: `DEMO1_QUOTA_GUARD=off`.

**"No Groq API key configured"**
The app retrieves without a key but cannot answer without one — every question
is generated by Groq. Add `GROQ_API_KEY` and restart:

- Streamlit Cloud: app menu → *Settings* → *Secrets*, then
  `GROQ_API_KEY = "gsk_..."` (the app bridges secrets into the environment)
- Local: copy `demo1_doc_intelligence/.env.example` to
  `demo1_doc_intelligence/.env` and fill it in

A free key: https://console.groq.com/keys

**"This app has gone over its resource limits"**
Streamlit Community Cloud gives an app 690 MB–2.7 GB of memory. The app now
stays inside that: retrieval runs on ONNX Runtime with int8 weights instead of
torch (which alone cost 1.35 GB), and it reads the container's own ceiling to
decide whether the second model is affordable. If you still hit the page, check
what the app is being asked to hold:

- `DEMO1_RERANK=off` forces the cheaper reranker (late interaction, no second model)
- `DEMO1_DOCS_DIR` pointing at a large PDF corpus means a large index build at boot
- Rebooting the app clears a one-off spike; it does not fix a standing footprint

**Answer is correct but missing details**
The answer is limited to what was retrieved. Try rephrasing the question.
If consistently incomplete, the relevant content may be in a complex table or figure
that pdfplumber could not extract cleanly.

---

## Known Limitations

| Limitation | Root cause |
|---|---|
| Scanned PDFs return nothing | pdfplumber requires a text layer |
| Complex multi-page tables may be incomplete | pdfplumber reads page by page |
| Questions per day are capped | The demo runs on a free Groq key; see the quota section above |
| Author/metadata questions often fail | Author info is on page 1, semantically far from metadata queries |

---

## File Locations

```
demo1_doc_intelligence/
├── data/
│   ├── raw_docs/          ← PDFs go here (gitignored)
│   ├── vectorstore_en/    ← ChromaDB (gitignored, rebuilt by ingest.py)
│   └── vectorstore_multi/ ← ChromaDB multilingual (gitignored, disabled)
└── src/
    ├── config.py          ← edit this to customize for your documents
    ├── ingest.py          ← PDF → chunks → vectorstore (~30 seconds)
    ├── retrieval.py       ← question → rerank → answer
    └── app.py             ← Streamlit UI (imports from config.py)
```
