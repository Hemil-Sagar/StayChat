# Cornerstone Hotel RAG Bot

Grounded multilingual hotel concierge bot built with LangChain LCEL,
FAISS vector search, and Google Gemini.

---

## Project structure

```
hotel_rag_langchain/
├── main.py            ← entry point / CLI
├── bot.py             ← LangChain LCEL chain + HotelBot class
├── ingest.py          ← PDF loading, chunking, FAISS index build/load
├── guardrail.py       ← post-generation anti-hallucination checks
├── utils.py           ← language detection, intent classification
├── eval.py            ← dynamic interactive eval harness
├── config.py          ← all tuneable constants
├── requirements.txt
└── hotel_brochure.pdf ← knowledge source (place here)
```

---

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"
```

Place `hotel_brochure.pdf` in the project folder, then:

```bash
python main.py
```

The FAISS index is built on first run and saved to `faiss_index/`.
Subsequent runs load the saved index instantly.

---

## Commands in the CLI

| Command | What it does |
|---------|--------------|
| (any text) | Chat with the bot |
| `eval` | Enter interactive eval mode |
| `reset` | Clear conversation history |
| `quit` | Exit |

---

## Eval mode (dynamic)

Type `eval` at the prompt. You then type questions one by one:

```
Q01 Enter question: What time is check-in?
Bot answers...
Mark as [p]ass / [f]ail / [s]kip: p

Q02 Enter question: What is the price per night?
Bot should refuse / escalate...
Mark as [p]ass / [f]ail / [s]kip: p

Q03 Enter question: done
```

Summary table is printed at the end. Nothing is hardcoded.

---

## How the guardrail works

After Gemini generates an answer, three checks run **before** it reaches the user:

| Check | Trigger | Result |
|-------|---------|--------|
| Price invented | Regex: ₹, Rs, $, "price is", "costs" | Discard + escalate |
| Link invented | Regex: https://, "payment link", "click here" | Discard + escalate |
| KB miss | Best FAISS score < 0.35 | Discard + escalate |

Escalation messages are pre-written in English, Hindi, and Hinglish.

---

## Tuning

Edit `config.py` to change chunk size, overlap, top-k, or the score threshold.
Delete `faiss_index/` after changing chunk settings so the index rebuilds.
