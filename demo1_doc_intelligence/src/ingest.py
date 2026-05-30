import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DOCS_DIR          = Path(__file__).parent.parent / "data" / "raw_docs"
VECTORSTORE_EN    = Path(__file__).parent.parent / "data" / "vectorstore_en"
VECTORSTORE_MULTI = Path(__file__).parent.parent / "data" / "vectorstore_multi"
MODEL_EN          = "all-MiniLM-L6-v2"
MODEL_MULTI       = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION        = "og_docs"
GROQ_MODEL        = "llama-3.3-70b-versatile"


def load_pdfs():
    import pdfplumber
    from langchain_core.documents import Document

    docs = []
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DOCS_DIR}")

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
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
                            separator = "| " + " | ".join(
                                ["---"] * len(table[0])
                            ) + " |"
                            table_md.append(
                                header + "\n" + separator + "\n" +
                                "\n".join(rows[1:])
                            )
                    if table_md:
                        text = text + "\n\n[TABLE]\n" + "\n\n".join(table_md)

                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source_file": pdf_path.name,
                            "page": page_num,
                        }
                    )
                    docs.append(doc)

    print(f"  → {len(docs)} pages loaded from {len(pdf_files)} PDFs")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  → {len(chunks)} chunks created")
    return chunks


def generate_questions_for_chunk(llm, chunk_text):
    try:
        messages = [
            SystemMessage(content=(
                "Generate 5 questions that this text chunk would answer. "
                "Make them diverse — use different vocabulary, technical terms, "
                "and layman phrasings. Cover both specific details and general topics. "
                "Return only the 5 questions, one per line, nothing else."
            )),
            HumanMessage(content=chunk_text[:2000]),
        ]
        response = llm.invoke(messages)
        questions = [
            q.strip() for q in response.content.strip().split("\n")
            if q.strip()
        ]
        return questions[:5]
    except Exception as e:
        print(f"    Warning: question generation failed — {e}")
        return []


def enrich_chunks_with_questions(chunks):
    print(f"  Generating questions for {len(chunks)} chunks...")
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    for i, chunk in enumerate(chunks):
        questions = generate_questions_for_chunk(llm, chunk.page_content)
        chunk.metadata["questions"] = " | ".join(questions)
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(chunks)} done...")
            time.sleep(1)
    print(f"  ✅ Questions generated for all chunks")
    return chunks


def build_vectorstore(chunks, persist_dir, model_name):
    print(f"  Model: {model_name}")
    print(f"  Saving to: {persist_dir}")

    if persist_dir.exists():
        print(f"  Clearing existing vectorstore...")
        import shutil
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    enhanced_chunks = []
    for chunk in chunks:
        from langchain_core.documents import Document
        questions = chunk.metadata.get("questions", "")
        enhanced_content = f"{questions}\n\n{chunk.page_content}" if questions else chunk.page_content
        enhanced_chunks.append(Document(
            page_content=enhanced_content,
            metadata=chunk.metadata,
        ))

    vectorstore = Chroma.from_documents(
        documents=enhanced_chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(persist_dir),
    )
    print(f"  ✅ Done — {vectorstore._collection.count()} chunks indexed")


if __name__ == "__main__":
    print("=== Step 1: Load PDFs ===")
    docs = load_pdfs()

    print("\n=== Step 2: Chunk documents ===")
    chunks = chunk_documents(docs)

    print("\n=== Step 3: Generate questions per chunk (Reverse HyDE) ===")
    chunks = enrich_chunks_with_questions(chunks)

    print("\n=== Step 4a: Build English vectorstore ===")
    build_vectorstore(chunks, VECTORSTORE_EN, MODEL_EN)

    print("\n=== Step 4b: Build multilingual vectorstore ===")
    build_vectorstore(chunks, VECTORSTORE_MULTI, MODEL_MULTI)

    print("\n✅ Both vectorstores complete.")