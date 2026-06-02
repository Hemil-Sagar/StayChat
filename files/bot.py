# bot.py
# Core Hotel RAG Bot built with LangChain LCEL.
#
# Chain pipeline (LCEL | operator):
#   retriever  ──►  prompt (KB chunks + history + intent)  ──►  Gemini  ──►  StrOutputParser
#
# Wrapped in RunnableWithMessageHistory for automatic multi-turn memory.
# Guardrail + intent run outside the chain (pre/post hooks).
# Every turn is saved to conversations.json via conversation_store.

from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

import guardrail
import conversation_store
from utils import detect_language, classify_intent
from config import GEMINI_MODEL, GEMINI_API_KEY, TOP_K

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = """You are the helpful, friendly concierge bot for Cornerstone Business Hotel.

STRICT RULES — follow every one, no exceptions:
1. Answer ONLY from the KB excerpts provided below.
2. NEVER invent, guess, or imply any price, rate, fee, or tariff.
   If a price is not explicitly stated in the excerpts, say you don't have that information.
3. NEVER generate, suggest, or hint at any booking URL or payment link.
4. If the excerpts do not contain enough information to answer the question, respond:
   "I'm sorry, I don't have that information. Please contact the front desk at extension 0."
5. Be warm, concise, and professional.
6. Reply in the SAME language the guest used (English, Hindi, or Hinglish).
7. Do not repeat the question back to the guest.

Intent detected: {intent}
Language detected: {language}

--- KB EXCERPTS (use ONLY these) ---
{context}
--- END KB EXCERPTS ---
"""


@dataclass
class BotResponse:
    answer:     str
    intent:     str
    language:   str
    top_score:  float
    escalated:  bool
    num_chunks: int


class HotelBot:
    """
    Encapsulates the full RAG pipeline:
      ingest → retrieve → generate → guardrail → log → respond
    """

    def __init__(self, vectorstore: FAISS):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
        )

        # Plain retriever — we call similarity_search_with_score ourselves
        # so we get raw dot-product scores (always in valid range for normalised
        # vectors) and avoid LangChain's relevance normalisation warning.
        self.vectorstore = vectorstore
        self.retriever   = vectorstore.as_retriever(
            search_kwargs={"k": TOP_K}
        )

        self._history_store: dict[str, ChatMessageHistory] = {}
        self._chain = self._build_chain()
        self._chain_with_history = RunnableWithMessageHistory(
            self._chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    # ── History helpers ───────────────────────────────────────────────────────
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._history_store:
            self._history_store[session_id] = ChatMessageHistory()
        return self._history_store[session_id]

    def reset_session(self, session_id: str = "default") -> None:
        self._history_store.pop(session_id, None)
        conversation_store.clear_session(session_id)

    # ── Chain construction ────────────────────────────────────────────────────
    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return (
            RunnablePassthrough()
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def _retrieve(self, query: str) -> tuple[str, float, int]:
        """
        Uses similarity_search_with_score (raw dot-product, no normalisation).
        For L2-normalised vectors this equals cosine similarity and is always
        in [0, 1] — no negative scores, no LangChain warning.
        """
        results = self.vectorstore.similarity_search_with_score(query, k=TOP_K)
        if not results:
            return "", 0.0, 0

        best_score = max(score for _, score in results)
        context = "\n\n".join(
            f"[Chunk {i+1} | relevance={score:.2f}]\n{doc.page_content}"
            for i, (doc, score) in enumerate(results)
        )
        return context, float(best_score), len(results)

    # ── Main chat method ──────────────────────────────────────────────────────
    def chat(self, user_query: str, session_id: str = "default") -> BotResponse:
        """
        Full pipeline:
        1. Detect language
        2. Classify intent
        3. Retrieve top-k chunks from FAISS (real vector search, no warnings)
        4. Run LCEL chain (prompt + history + Gemini)
        5. Guardrail check on generated answer
        6. Save turn to conversations.json
        7. Return BotResponse
        """
        language = detect_language(user_query)
        intent   = classify_intent(user_query, self.llm)

        context, top_score, num_chunks = self._retrieve(user_query)

        raw_answer = self._chain_with_history.invoke(
            {
                "input":    user_query,
                "context":  context,
                "intent":   intent,
                "language": language,
            },
            config={"configurable": {"session_id": session_id}},
        )

        # Post-generation guardrail
        result = guardrail.check(raw_answer, top_score)
        if result.safe:
            final_answer = raw_answer
            escalated    = False
        else:
            final_answer = guardrail.escalation_message(result.reason, language)
            escalated    = True

        # Persist turn to JSON
        conversation_store.append_turn(
            session_id=session_id,
            user_query=user_query,
            bot_answer=final_answer,
            intent=intent,
            language=language,
            top_score=top_score,
            escalated=escalated,
        )

        return BotResponse(
            answer=final_answer,
            intent=intent,
            language=language,
            top_score=top_score,
            escalated=escalated,
            num_chunks=num_chunks,
        )