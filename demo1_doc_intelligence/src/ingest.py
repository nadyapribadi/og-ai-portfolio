import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DOCS_DIR          = Path(__file__).parent.parent / "data" / "raw_docs"
VECTORSTORE_EN    = Path(__file__).parent.parent / "data" / "vectorstore_en"
VECTORSTORE_MULTI = Path(__file__).parent.parent / "data" / "vectorstore_multi"
MODEL_EN          = "all-MiniLM-L6-v2"
MODEL_MULTI       = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION        = "og_docs"


def load_pdfs():
    import pdfplumber

    docs = []
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DOCS_DIR}")

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""

                # Convert tables to markdown — preserves column structure
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

                # Skip empty pages
                if len(text.strip()) < 50:
                    continue

                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source_file": pdf_path.name,
                        "page": page_num,
                    }
                ))

    print(f"  → {len(docs)} pages loaded from {len(pdf_files)} PDFs")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Filter out near-empty chunks
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 100]
    print(f"  → {len(chunks)} chunks created")
    return chunks


def build_vectorstore(chunks, persist_dir, model_name):
    print(f"  Model: {model_name}")
    print(f"  Saving to: {persist_dir}")

    if persist_dir.exists():
        print(f"  Clearing existing vectorstore...")
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma.from_documents(
        documents=chunks,
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

    print("\n=== Step 3a: Build English vectorstore ===")
    build_vectorstore(chunks, VECTORSTORE_EN, MODEL_EN)

    print("\n=== Step 3b: Build multilingual vectorstore ===")
    build_vectorstore(chunks, VECTORSTORE_MULTI, MODEL_MULTI)

    print("\n✅ Both vectorstores complete.")