# eval.py
# Dynamic interactive eval harness.
# No questions are hardcoded — you type each question live,
# see the bot's answer, then mark it pass or fail yourself.

from bot import HotelBot, BotResponse

DIVIDER = "─" * 65


def _print_response(resp: BotResponse, q_num: int) -> None:
    print(f"\n{'─'*65}")
    print(f"  Q{q_num:02d} | intent={resp.intent}  lang={resp.language}  "
          f"top_score={resp.top_score:.2f}  escalated={resp.escalated}")
    print(f"{'─'*65}")
    print(f"  Answer:\n  {resp.answer}")
    print(f"{'─'*65}")


def run_eval(bot: HotelBot) -> None:
    """
    Interactive eval loop:
    - Type a question → bot answers
    - Mark pass (p) / fail (f) / skip (s)
    - Type 'done' to finish and see summary
    Each question gets a fresh session (no carry-over history).
    """
    print(f"\n{'═'*65}")
    print("  EVAL MODE — Dynamic Question Set")
    print("  Type your question, review the answer, then mark p/f/s.")
    print("  Type 'done' instead of a question to finish and see summary.")
    print(f"{'═'*65}\n")

    results = []   # list of {"q": str, "pass": bool|None}
    q_num   = 1

    while True:
        try:
            question = input(f"  Q{q_num:02d} Enter question (or 'done'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() == "done":
            break
        if not question:
            continue

        # Fresh session per eval question — no history carry-over
        session = f"eval_{q_num}"
        resp = bot.chat(question, session_id=session)
        _print_response(resp, q_num)

        while True:
            try:
                mark = input("  Mark as  [p]ass / [f]ail / [s]kip : ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                mark = "s"
                break
            if mark in ("p", "f", "s"):
                break
            print("  Please type p, f, or s.")

        results.append({
            "q":      question,
            "intent": resp.intent,
            "lang":   resp.language,
            "score":  resp.top_score,
            "esc":    resp.escalated,
            "result": {"p": True, "f": False, "s": None}[mark],
        })
        q_num += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    if not results:
        print("\n  No questions evaluated.\n")
        return

    judged  = [r for r in results if r["result"] is not None]
    passed  = sum(1 for r in judged if r["result"])
    failed  = sum(1 for r in judged if not r["result"])
    skipped = sum(1 for r in results if r["result"] is None)

    print(f"\n{'═'*65}")
    print("  EVAL SUMMARY")
    print(f"{'═'*65}")
    print(f"  Total   : {len(results)}")
    print(f"  Passed  : {passed}")
    print(f"  Failed  : {failed}")
    print(f"  Skipped : {skipped}")
    if judged:
        pct = passed / len(judged) * 100
        print(f"  Score   : {passed}/{len(judged)}  ({pct:.0f}%)")
    print(f"{'─'*65}")
    for i, r in enumerate(results, 1):
        symbol = {True: "✓", False: "✗", None: "–"}[r["result"]]
        print(f"  {symbol} Q{i:02d} [{r['intent']} | {r['lang']} | score={r['score']:.2f} | esc={r['esc']}]")
        # Wrap question text at 55 chars for readability
        q_wrapped = r["q"][:55] + ("…" if len(r["q"]) > 55 else "")
        print(f"       {q_wrapped}")
    print(f"{'═'*65}\n")
