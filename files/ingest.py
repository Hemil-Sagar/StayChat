# ingest.py
# Loads the hotel PDF, splits it into chunks, embeds with a multilingual
# sentence-transformer, and persists a FAISS index to disk.
# On subsequent runs the saved index is loaded instead of re-building.

import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    PDF_PATH, FAISS_INDEX_DIR, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
)


def get_embedder() -> HuggingFaceEmbeddings:
    """
    Multilingual MiniLM — produces language-agnostic semantic vectors.
    A Hindi query for 'swiming pool timings' maps close to the English
    chunk about 'pool open 6 AM', so retrieval works across languages
    without translating the KB.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # cosine via dot-product
    )


def build_index(embedder: HuggingFaceEmbeddings) -> FAISS:
    """
    1. PyPDFLoader  → one Document per PDF page
    2. RecursiveCharacterTextSplitter → 500-char chunks, 60-char overlap
    3. FAISS.from_documents → embed + index in one call
    4. Save to disk so we never re-build unless the PDF changes
    """
    print(f"[ingest] Loading PDF: {PDF_PATH}")
    loader = PyPDFLoader(str(PDF_PATH))
    pages  = loader.load()                     # list[Document]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"[ingest] {len(pages)} pages → {len(chunks)} chunks")

    print("[ingest] Building FAISS index …")
    vectorstore = FAISS.from_documents(chunks, embedder)

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))
    print(f"[ingest] Index saved to {FAISS_INDEX_DIR}/")
    return vectorstore


def load_or_build_index(embedder: HuggingFaceEmbeddings) -> FAISS:
    """
    Load persisted FAISS index if it exists; otherwise build from PDF.
    Pass allow_dangerous_deserialization=True because we trust our own
    locally saved pickle files.
    """
    index_file = FAISS_INDEX_DIR / "index.faiss"
    if index_file.exists():
        print(f"[ingest] Loading existing index from {FAISS_INDEX_DIR}/")
        return FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embedder,
            allow_dangerous_deserialization=True,
        )
    return build_index(embedder)
