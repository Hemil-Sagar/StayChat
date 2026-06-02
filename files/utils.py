# utils.py
# Language detection (pure Python, no external library) and
# intent classification (single lightweight Gemini call).

import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import INTENT_LABELS

# ── Language detection ────────────────────────────────────────────────────────
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Common Hindi words written in Latin script that often appear in Hinglish
_HINGLISH_VOCAB = {
    "kya", "hai", "hain", "nahi", "nahin", "kaise", "kab", "kahan",
    "mujhe", "mera", "meri", "mere", "aap", "tum", "toh", "bhi",
    "chahiye", "batao", "bata", "theek", "accha", "sahi", "please",
    "yaar", "bhai", "didi", "hotel", "room", "khana", "pool",
    "wifi", "booking", "checkout", "checkin", "tak", "se", "ko",
    "ka", "ki", "ke", "ek", "do", "din", "raat", "subah", "shaam",
}


def detect_language(text: str) -> str:
    """
    Returns 'hindi', 'hinglish', or 'english'.
    • Hindi   → text contains Devanagari Unicode characters
    • Hinglish→ Latin-script text with ≥2 Hindi vocabulary words
    • English → default
    """
    if _DEVANAGARI.search(text):
        return "hindi"
    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if len(tokens & _HINGLISH_VOCAB) >= 2:
        return "hinglish"
    return "english"


# ── Intent classification ─────────────────────────────────────────────────────
_INTENT_PROMPT = """Classify the hotel guest query into EXACTLY ONE of these labels:
booking_inquiry, amenity_question, complaint, staff_command, other

Definitions:
- booking_inquiry   : questions about reservations, check-in, check-out, cancellations, room availability
- amenity_question  : questions about hotel facilities, services, dining, Wi-Fi, gym, transport, policies
- complaint         : expressing dissatisfaction, reporting a problem or an issue
- staff_command     : requesting an action from hotel staff (e.g. send towels, call a cab, wake-up call)
- other             : anything that doesn't fit the above

Guest query: {query}

Reply with ONLY the label — no explanation, no punctuation."""


def classify_intent(query: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Sends a minimal, constrained prompt to Gemini.
    Falls back to 'other' if the response isn't a known label.
    """
    prompt  = _INTENT_PROMPT.format(query=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    label   = response.content.strip().lower().replace(" ", "_").rstrip(".")
    return label if label in INTENT_LABELS else "other"
