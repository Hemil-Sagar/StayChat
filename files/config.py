# config.py
# All tuneable constants for the Hotel RAG Bot.
# Change values here; nothing else needs editing.

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_PATH        = Path("hotel_brochure.pdf")   # source KB document
FAISS_INDEX_DIR = Path("faiss_index")          # persisted index folder

# ── Models ────────────────────────────────────────────────────────────────────
GEMINI_MODEL     = "gemini-2.5-flash"
EMBEDDING_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Retrieval ─────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 500    # characters per chunk
CHUNK_OVERLAP   = 60     # overlap between consecutive chunks
TOP_K           = 3      # chunks returned per query

# ── Guardrail ─────────────────────────────────────────────────────────────────
SCORE_THRESHOLD = 0.35   # min cosine similarity; below this → KB miss → escalate

# ── Intent labels ─────────────────────────────────────────────────────────────
INTENT_LABELS = [
    "booking_inquiry",
    "amenity_question",
    "complaint",
    "staff_command",
    "other",
]

# ── API key ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
