import os
from pathlib import Path
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VECTORSTORE_EN       = Path(__file__).parent.parent / "data" / "vectorstore_en"
VECTORSTORE_MULTI    = Path(__file__).parent.parent / "data" / "vectorstore_multi"
MODEL_EN             = "all-MiniLM-L6-v2"
MODEL_MULTI          = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION           = "og_docs"
TOP_K                = 4
GROQ_MODEL           = "llama-3.1-8b-instant"
MULTILINGUAL_ENABLED = False

SYSTEM_PROMPT = """You are an expert assistant for upstream oil and gas operations,
procurement, HSE standards, and supply chain management.

You will be given numbered document excerpts and a question.
Your job:
- Read ALL excerpts carefully
- Extract and combine relevant information from whichever excerpts contain it
- If multiple excerpts contribute to the answer, combine them into one complete answer
- Always cite the source filename and page number for each piece of information used
- If a list appears incomplete, show ONLY what is in the excerpts. Never complete,
  extend, or guess the remaining items. End with:
  "Note: this list may be incomplete — only retrieved excerpts are shown."
- If genuinely no excerpt contains relevant information, say:
  "This information is not found in the loaded documents."
- Never guess or use knowledge outside the provided excerpts
- Answer in the same language the user used to ask the question

Document excerpts:
{context}"""

_vs_cache = {}


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def translate_to_english(text):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source='auto', target='en').translate(text)

def expand_query(question):
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    messages = [
        SystemMessage(content=(
            "You are helping search an oil and gas document database. "
            "Given a question, generate 3 search queries that would find the answer. "
            "Make each query use DIFFERENT vocabulary than the original — "
            "focus on the specific content, actions, or procedures the answer would contain. "
            "Return only the 3 queries, one per line, nothing else."
        )),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    variants = [v.strip() for v in response.content.strip().split("\n") if v.strip()]
    return [question] + variants

def load_vectorstore(lang):
    if not MULTILINGUAL_ENABLED:
        lang = "en"
    if lang in _vs_cache:
        return _vs_cache[lang]

    model_name = MODEL_EN if lang == "en" else MODEL_MULTI
    persist_dir = VECTORSTORE_EN if lang == "en" else VECTORSTORE_MULTI

    print(f"  Language: {lang} → using {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    print(f"  Vectorstore loaded: {vectorstore._collection.count()} chunks")
    _vs_cache[lang] = vectorstore
    return vectorstore


def search_chunks(vectorstore, question):
    seen_ids = set()
    all_chunks = []

    variants = expand_query(question)
    print(f"  Query variants: {len(variants)}")

    for variant in variants:
        results = vectorstore.max_marginal_relevance_search(
            variant, k=TOP_K + 2, fetch_k=20
        )
        for r in results:
            if r.metadata.get("page", 99) > 0:
                doc_id = f"{r.metadata.get('source_file')}_{r.metadata.get('page')}_{r.page_content[:50]}"
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_chunks.append(r)

    return all_chunks[:TOP_K * 2]


def build_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source_file", "unknown")
        page   = chunk.metadata.get("page", "?")
        parts.append(
            f"[{i}] Source: {source} | Page: {page}\n{chunk.page_content}"
        )
    return "\n\n".join(parts)


def ask(question):
    lang = detect_language(question)
    search_question = question

    if lang != "en":
        print(f"  Translating from {lang} to English for search...")
        search_question = translate_to_english(question)
        print(f"  Translated: {search_question}")

    vectorstore = load_vectorstore(lang)
    chunks      = search_chunks(vectorstore, search_question)
    context     = build_context(chunks)

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    sources  = list({c.metadata.get("source_file", "unknown") for c in chunks})

    return {
        "answer":   response.content,
        "sources":  sources,
        "chunks":   chunks,
        "language": lang,
    }


if __name__ == "__main__":
    test_questions = [
        "What are the life saving rules?",
        "What are the 9 IOGP life saving rules?",
        "Apa saja aturan keselamatan jiwa?",
        "What are the Tier 1 and Tier 2 process safety KPIs?",
        "What is required before entering a confined space?",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = ask(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Language detected: {result['language']}")