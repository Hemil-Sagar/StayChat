# guardrail.py
# Anti-hallucination guardrail — runs AFTER the LLM generates, BEFORE
# the answer reaches the user.  Deterministic regex checks mean even a
# jailbreak or creative prompt injection cannot leak invented prices.

import re
from dataclasses import dataclass
from typing import Optional

from config import SCORE_THRESHOLD

# ── Patterns that must never appear in an answer ─────────────────────────────
PRICE_RE = re.compile(
    r"₹\s*\d"                   # ₹1500, ₹ 999
    r"|Rs\.?\s*\d"              # Rs.500, Rs 800
    r"|\$\s*\d"                 # $45
    r"|USD\s*\d"                # USD 100
    r"|INR\s*\d"                # INR 2000
    r"|price\s+is\s+\w"        # price is affordable
    r"|costs?\s+(?:Rs|INR|₹|\$)",  # costs Rs
    re.IGNORECASE,
)

LINK_RE = re.compile(
    r"https?://"                # any URL
    r"|payment\s+link"         # payment link
    r"|book\s+(?:at|via|on)\s+http"
    r"|click\s+here",
    re.IGNORECASE,
)


@dataclass
class GuardrailResult:
    safe: bool
    reason: str          # "ok" | "price_invented" | "link_invented" | "kb_miss:<score>"


def check(answer: str, top_score: float) -> GuardrailResult:
    """
    Three sequential checks:
    1. Invented price or fee in the answer text
    2. Invented URL or payment link
    3. Best retrieved chunk score below threshold (question not in KB)
    """
    if PRICE_RE.search(answer):
        return GuardrailResult(safe=False, reason="price_invented")
    if LINK_RE.search(answer):
        return GuardrailResult(safe=False, reason="link_invented")
    if top_score < SCORE_THRESHOLD:
        return GuardrailResult(safe=False, reason=f"kb_miss:{top_score:.3f}")
    return GuardrailResult(safe=True, reason="ok")


# ── Escalation messages (pre-written, never LLM-generated) ───────────────────
_ESCALATION = {
    "english": (
        "I'm sorry, I don't have that information in my knowledge base. "
        "For accurate details please contact our front desk directly — "
        "call extension 0 from any in-room phone, dial +91-120-4567890, "
        "or visit the lobby. A team member will be happy to help."
    ),
    "hindi": (
        "मुझे खेद है, यह जानकारी मेरे पास उपलब्ध नहीं है। "
        "सटीक जानकारी के लिए कृपया हमारे फ्रंट डेस्क से संपर्क करें — "
        "इन-रूम फोन से extension 0 डायल करें या +91-120-4567890 पर कॉल करें।"
    ),
    "hinglish": (
        "Sorry yaar, yeh information mere paas abhi available nahi hai. "
        "Sahi details ke liye please front desk ko contact karo — "
        "room phone se extension 0 dial karo ya +91-120-4567890 pe call karo. "
        "Hamaari team help karegi!"
    ),
}

_PRICE_PREFIX = {
    "english":  "I can't provide pricing information — rates vary and must be confirmed directly with the hotel. ",
    "hindi":    "मैं कीमत की जानकारी नहीं दे सकता — कृपया होटल से सीधे confirm करें। ",
    "hinglish": "Pricing ki info main nahi de sakta — please hotel se directly confirm karo. ",
}

_LINK_PREFIX = {
    "english":  "For bookings and payments please contact our reservations team directly. ",
    "hindi":    "बुकिंग और भुगतान के लिए सीधे हमारी reservations टीम से संपर्क करें। ",
    "hinglish": "Booking aur payment ke liye seedha hamaari reservations team se baat karo. ",
}


def escalation_message(reason: str, language: str) -> str:
    base = _ESCALATION.get(language, _ESCALATION["english"])
    if "price_invented" in reason:
        prefix = _PRICE_PREFIX.get(language, _PRICE_PREFIX["english"])
        return prefix + base
    if "link_invented" in reason:
        prefix = _LINK_PREFIX.get(language, _LINK_PREFIX["english"])
        return prefix + base
    return base   # kb_miss
