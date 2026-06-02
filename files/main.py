#!/usr/bin/env python3
# main.py
# Entry point for the Hotel RAG Bot.
# Run:  python main.py
# Eval: type 'eval' at the prompt

import sys

from ingest import get_embedder, load_or_build_index
from bot   import HotelBot
from eval  import run_eval
from config import PDF_PATH

SESSION = "default"

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        Cornerstone Business Hotel — Concierge Bot           ║
║        Powered by LangChain · FAISS · Gemini                ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                   ║
║    eval   → enter interactive eval mode                      ║
║    reset  → clear conversation history                       ║
║    quit   → exit                                             ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    if not PDF_PATH.exists():
        sys.exit(
            f"[error] Hotel PDF not found at '{PDF_PATH}'.\n"
            "Place hotel_brochure.pdf in the same folder as main.py."
        )

    print("[startup] Loading embedding model …")
    embedder    = get_embedder()

    print("[startup] Loading / building FAISS index …")
    vectorstore = load_or_build_index(embedder)

    print("[startup] Initialising bot …")
    bot = HotelBot(vectorstore)

    print(BANNER)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "quit":
            print("Goodbye!")
            break

        if cmd == "reset":
            bot.reset_session(SESSION)
            print("[Bot] Conversation history cleared.\n")
            continue

        if cmd == "eval":
            run_eval(bot)
            continue

        # ── Normal chat ───────────────────────────────────────────────────
        resp = bot.chat(user_input, session_id=SESSION)

        tag = f"[{resp.intent} | {resp.language} | score={resp.top_score:.2f}]"
        print(f"\nBot {tag}:\n{resp.answer}\n")


if __name__ == "__main__":
    main()
