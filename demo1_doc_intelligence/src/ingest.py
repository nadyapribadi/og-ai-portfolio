import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DOCS_DIR = Path(__file__).parent.parent / "data" / "raw_docs"
VECTORSTORE_DIR = Path(__file__).parent.parent / "data" / "vectorstore"
COLLECTION_NAME = "og_docs"

def load_pdfs():
    docs = []
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DOCS_DIR}")
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
        docs.extend(pages)
    print(f"  → {len(docs)} pages loaded from {len(pdf_files)} PDFs")
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  → {len(chunks)} chunks created")
    return chunks

def build_vectorstore(chunks):
    print(f"  Building vectorstore in {VECTORSTORE_DIR}")
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )
    print(f"  ✅ Done — {vectorstore._collection.count()} chunks indexed")
    return vectorstore

if __name__ == "__main__":
    print("=== Step 1: Load PDFs ===")
    docs = load_pdfs()
    print("\n=== Step 2: Chunk documents ===")
    chunks = chunk_documents(docs)
    print("\n=== Step 3: Build vectorstore ===")
    build_vectorstore(chunks)
    print("\n✅ Ingestion complete.")