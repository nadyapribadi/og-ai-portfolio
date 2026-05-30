# app.py Specification — O&G Document Intelligence UI

Build AFTER ingest.py fixes are verified and all test questions pass.

---

## Decisions (locked, do not change)

| Decision | Choice | Reason |
|----------|--------|--------|
| Streaming | Word-by-word | Best for all 3 personas, modern AI UX |
| Sample questions | Clickable buttons by category | Zero friction for demo |
| App title | "O&G Document Intelligence" + credit | Product credibility + portfolio attribution |
| Language | Auto-detect + visible indicator + manual override | Shows capability to Persona 3 |

---

## UI layout

```
┌─────────────────────────────────────────────────────────────────┐
│ SIDEBAR                                                         │
│                                                                 │
│  🛢️ O&G Document Intelligence                                   │
│  Built by Nadya Boyke Pribadi                                   │
│                                                                 │
│  ──────────────────────────────────                             │
│  📚 Document sources (9 PDFs)                                   │
│  • IOGP 459 — Life-Saving Rules                                 │
│  • IOGP 456 — Process Safety KPIs                               │
│  • JIP33 S-737 Deluge Skids                                     │
│  • JIP33 S-717 Noise Equipment                                  │
│  • JIP33 S-719 Water Mist Fire Protection                       │
│                                                                 │
│  ──────────────────────────────────                             │
│  🌐 Language                                                    │
│  [Auto-detect ▼]  Detected: English 🇬🇧                        │
│                                                                 │
│  ──────────────────────────────────                             │
│  💡 Try asking:                                                 │
│  🦺 HSE Rules                                                   │
│  [What are the life saving rules?        ]                      │
│  [Confined space requirements            ]                      │
│  [Hot work in hazardous area             ]                      │
│                                                                 │
│  📊 Process Safety                                              │
│  [Tier 1 vs Tier 2 PSE difference        ]                      │
│  [How KPIs are measured                  ]                      │
│  [What is LOPC?                          ]                      │
│                                                                 │
│  ⚙️ Equipment Standards                                         │
│  [S-737 deluge skid design               ]                      │
│  [S-737 electrical installation standards]                      │
│  [S-717 noise equipment requirements     ]                      │
│                                                                 │
│  🇮🇩 Bahasa Indonesia                                           │
│  [Aturan keselamatan jiwa                ]                      │
│  [Sebelum memasuki ruang tertutup        ]                      │
│  [Perbedaan Tier 1 dan Tier 2            ]                      │
│                                                                 │
│  ──────────────────────────────────                             │
│  [🗑️ Clear conversation]                                        │
│                                                                 │
│  ──────────────────────────────────                             │
│  ⚠️ AI answers may be incomplete.                               │
│  Always verify against original documents.                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ MAIN AREA                                                       │
│                                                                 │
│  🛢️ O&G Document Intelligence                                   │
│  Ask questions from IOGP standards and JIP33 specifications     │
│                                                                 │
│  ──────────────────────────────────                             │
│  [Chat history scrolls here]                                    │
│                                                                 │
│  👤 USER                                                        │
│  What are the life saving rules?                                │
│                                                                 │
│  🤖 ASSISTANT  🇬🇧 English  📚 459.pdf                         │
│  The Life-Saving Rules are nine rules designed to...            │
│  [answer streams word by word]                                  │
│                                                                 │
│  📚 Sources: 459.pdf (pages 7-8)                               │
│  ▶ View retrieved document excerpts                             │
│                                                                 │
│  ──────────────────────────────────                             │
│                                                                 │
│  Ask about IOGP standards, JIP33 specs, process safety...  [→] │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical implementation

### Imports needed

```python
import streamlit as st
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from retrieval import ask, load_vectorstore
```

### Page config

```python
st.set_page_config(
    page_title="O&G Document Intelligence",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

### Caching — critical for performance

```python
@st.cache_resource(show_spinner="Loading document index...")
def get_vectorstore():
    return load_vectorstore("en")
```

Load once per session. Without this, the embedding model reloads on every question.

### Session state — required for chat history

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

if "language_override" not in st.session_state:
    st.session_state.language_override = "auto"
```

### Streaming implementation

```python
def stream_answer(answer_text):
    words = answer_text.split(" ")
    for word in words:
        yield word + " "
        import time
        time.sleep(0.02)
```

Note: Groq streaming via LangChain is possible but complex. Simpler approach:
get full answer first, then stream the display word-by-word using `st.write_stream()`.

### Language indicator

```python
lang_flags = {"en": "🇬🇧 English", "id": "🇮🇩 Bahasa Indonesia", "ar": "🇸🇦 Arabic"}
detected_flag = lang_flags.get(result["language"], f"🌐 {result['language']}")
```

### Sample questions — clickable buttons

```python
SAMPLE_QUESTIONS = {
    "🦺 HSE Rules": [
        "What are the life saving rules?",
        "What must I confirm before entering a confined space?",
        "What are the hot work requirements in a hazardous area?",
    ],
    "📊 Process Safety": [
        "What is the difference between Tier 1 and Tier 2 process safety events?",
        "How are process safety KPIs measured?",
        "What does LOPC stand for and what are its consequences?",
    ],
    "⚙️ Equipment Standards": [
        "What does IOGP S-737 specify for deluge skid design?",
        "What standards does S-737 reference for electrical installations?",
        "What does IOGP S-717 cover for noise emitting equipment?",
    ],
    "🇮🇩 Bahasa Indonesia": [
        "Apa saja aturan keselamatan jiwa menurut IOGP?",
        "Apa yang harus dilakukan sebelum memasuki ruang tertutup?",
        "Apa perbedaan antara kejadian keselamatan proses Tier 1 dan Tier 2?",
    ],
}

# In sidebar:
for category, questions in SAMPLE_QUESTIONS.items():
    st.sidebar.markdown(f"**{category}**")
    for q in questions:
        if st.sidebar.button(q, key=f"btn_{q[:20]}", use_container_width=True):
            st.session_state.pending_question = q
```

### Handling pending questions from button clicks

```python
question = st.chat_input("Ask about upstream O&G standards...")

# Handle button clicks
if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")
```

### Answer display with sources and expander

```python
with st.chat_message("assistant"):
    # Language badge
    st.caption(f"{detected_flag} · 📚 {', '.join(result['sources'])}")

    # Stream the answer
    st.write_stream(stream_answer(result["answer"]))

    # Sources detail
    if result["sources"]:
        st.caption(f"**Sources:** {', '.join(result['sources'])}")

    # Raw chunks expander (for technical credibility)
    with st.expander("View retrieved document excerpts"):
        for i, chunk in enumerate(result["chunks"], 1):
            source = chunk.metadata.get("source_file", "unknown")
            page = chunk.metadata.get("page", "?")
            st.markdown(f"**[{i}]** `{source}` — Page {page}")
            st.text(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)
            st.divider()
```

---

## Run locally

```bash
streamlit run demo1_doc_intelligence/src/app.py
```

---

## Deploy to Streamlit Cloud

1. Go to share.streamlit.io
2. Connect GitHub repo: nadyapribadi/og-ai-portfolio
3. Main file: `demo1_doc_intelligence/src/app.py`
4. Add secrets in Streamlit Cloud dashboard:
   ```
   GROQ_API_KEY = "gsk_..."
   ```
5. Note: vectorstore must be included in repo OR rebuilt on deploy
   - Option A: Add vectorstore to git (large files — not ideal)
   - Option B: Run ingest on deploy (add to requirements + startup script)
   - Option C: Use Pinecone instead of ChromaDB (cloud vectorstore)
   - **Recommended for now: Option A with git-lfs or rebuild on deploy**

---

## Success criteria for app.py

```
✅ App loads without errors
✅ Vectorstore loads once (cached), not on every question
✅ 12 sample questions render as clickable buttons in sidebar
✅ Clicking a button submits the question automatically
✅ Answer streams word by word
✅ Language detected and displayed (🇬🇧 / 🇮🇩)
✅ Source documents listed after answer
✅ "View retrieved excerpts" expander works
✅ Chat history persists within session
✅ Clear chat button works
✅ Manual language override works
✅ Error message if Groq rate limited
✅ Deployed to Streamlit Cloud with working demo link
```
