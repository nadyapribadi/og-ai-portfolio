# Pending fix: ingest.py improvements

Two changes needed before building app.py.
Do NOT move to app.py until both fixes are verified.

---

## Fix 1 — pdfplumber table → markdown conversion

### Problem
pdfplumber extracts table content as flowing text — column structure is lost.
LLM cannot determine which value belongs to which column (e.g. Tier 1 vs Tier 2).

### Solution
Use `page.extract_tables()` alongside `page.extract_text()`.
Convert table arrays to markdown format with | separators.
Append markdown tables to page text before chunking.

### Change in `load_pdfs()` function

Replace the pdfplumber extraction block with this pattern:

```python
text = page.extract_text() or ""

tables = page.extract_tables()
if tables:
    table_md = []
    for table in tables:
        if not table:
            continue
        rows = []
        for row in table:
            clean = [cell or "" for cell in row]
            rows.append("| " + " | ".join(clean) + " |")
        if rows:
            header = rows[0]
            separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
            table_md.append(header + "\n" + separator + "\n" + "\n".join(rows[1:]))
    if table_md:
        text = text + "\n\n[TABLE]\n" + "\n\n".join(table_md)
```

### Verify before running full ingest
```bash
python3 -c "
import pdfplumber
with pdfplumber.open('demo1_doc_intelligence/data/raw_docs/456.pdf') as pdf:
    page = pdf.pages[28]
    tables = page.extract_tables()
    for table in tables:
        if not table:
            continue
        rows = []
        for row in table:
            clean = [cell or '' for cell in row]
            rows.append('| ' + ' | '.join(clean) + ' |')
        if rows:
            header = rows[0]
            sep = '| ' + ' | '.join(['---'] * len(table[0])) + ' |'
            print(header)
            print(sep)
            for r in rows[1:]:
                print(r)
        print()
"
```

Expected: Table E.1 renders with Tier 1 and Tier 2 columns clearly separated.

---

## Fix 2 — Increase questions per chunk from 3 to 5

### Problem
3 questions per chunk doesn't cover enough vocabulary variants.
QRS documents (inspection, quality requirements) not retrieved for relevant questions.
S-717 Annex J content not retrieved for "what does S-717 cover?" type questions.

### Solution
Generate 5 questions per chunk with explicit instruction to vary vocabulary.

### Change in `generate_questions_for_chunk()` function

```python
# Change prompt from:
"Generate 3 short questions that this text chunk would answer. "
"Return only the 3 questions, one per line, nothing else."

# To:
"Generate 5 questions that this text chunk would answer. "
"Make them diverse — use different vocabulary, technical terms, "
"and layman phrasings. Cover both specific details and general topics. "
"Return only the 5 questions, one per line, nothing else."

# Change return from:
return questions[:3]

# To:
return questions[:5]
```

---

## After both fixes — run ingest

```bash
python demo1_doc_intelligence/src/ingest.py
# Takes ~10-12 minutes
```

Expected output:
```
=== Step 1: Load PDFs ===
  → 223 pages loaded from 9 PDFs

=== Step 2: Chunk documents ===
  → 246 chunks created

=== Step 3: Generate questions per chunk (Reverse HyDE) ===
  Generating questions for 246 chunks...
  ✅ Questions generated for all chunks

=== Step 4a: Build English vectorstore ===
  ✅ Done — 246 chunks indexed

=== Step 4b: Build multilingual vectorstore ===
  ✅ Done — 246 chunks indexed
```

---

## After ingest — test these questions

Update `__main__` block in retrieval.py and verify all pass:

```python
test_questions = [
    # Previously failing — should now work
    "What does IOGP S-737 specify for deluge skid inspection?",
    "What does S-717 cover?",
    "What does LOPC stand for and what are its consequences?",
    # Regression check — must still work
    "What are the life saving rules?",
    "Apa perbedaan antara kejadian keselamatan proses Tier 1 dan Tier 2?",
    "What standards does S-737 reference for electrical installations?",
]
```

Success criteria:
- S-737 inspection → returns QRS content (CAS levels, hold points, witness points)
- S-717 → returns more than one line (Annex J noise testing requirements)
- LOPC consequences → table columns clearly mapped to Tier 1 / Tier 2
- All regression questions still passing

Only move to app.py after all 6 pass.
