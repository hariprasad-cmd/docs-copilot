"""Evaluate the copilot before "go-live".

For every question in eval/questions.json it checks three things:
  1. Retrieval hit  - did the expected document show up in the retrieved chunks? (pure code, free)
  2. Grounded       - is every claim in the answer supported by the sources?  (LLM judge)
  3. Correct        - does the answer match the expected answer?              (LLM judge)
Out-of-scope questions pass only if the copilot declines instead of inventing an answer.

The LLM judge only answers yes/no questions. All the scores and percentages are
computed in Python, so the arithmetic is always right.

Run from the project root (after ingesting the sample docs):
    python -m eval.run_eval
"""

import json
import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from app import config
from app.llm import structured
from app.rag import answer

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"

JUDGE_PROMPT = """You are grading a customer-support assistant's answer.

<sources>
{sources}
</sources>

<question>{question}</question>
<expected_answer>{expected}</expected_answer>
<assistant_answer>{answer}</assistant_answer>

Decide each field:
- declined: true if the assistant said it does not have the information / could not answer.
- grounded: true if every factual claim in the assistant answer is supported by the sources.
  An answer that only declines makes no claims, so it is always grounded.
  Citation markers like [1] are not claims; ignore them.
- correct: true if the assistant answer contains the key facts of the expected answer
  (numbers, timing, yes/no). Different wording, extra correct detail, or a shorter
  answer with the same facts are all still correct. Only mark false if a key fact is
  missing or wrong.
  If the expected answer starts with OUT_OF_SCOPE, correct is true exactly when the assistant declined.
- reason: one short sentence explaining any field that is false."""


class Grade(BaseModel):
    declined: bool
    grounded: bool
    correct: bool
    reason: str


def judge(question: str, expected: str, result: dict) -> Grade:
    sources = "\n\n".join(f"[{s['n']}] ({s['source']}) {s['text']}" for s in result["sources"])
    prompt = JUDGE_PROMPT.format(sources=sources, question=question, expected=expected, answer=result["answer"])
    return structured(prompt, Grade)


def main() -> None:
    cases = json.loads((EVAL_DIR / "questions.json").read_text())
    rows = []
    print(f"Evaluating with {config.LLM_PROVIDER} / {config.MODEL}\n")

    for i, case in enumerate(cases, start=1):
        start = time.time()
        result = answer(case["question"], config.DEFAULT_WORKSPACE)
        seconds = round(time.time() - start, 1)
        grade = judge(case["question"], case["expected"], result)

        in_scope = case["expected_source"] is not None
        retrieved = [s["source"] for s in result["sources"]]
        if in_scope:
            retrieval_hit = case["expected_source"] in retrieved
            passed = grade.grounded and grade.correct and not grade.declined
        else:
            retrieval_hit = None  # nothing to find for out-of-scope questions
            passed = grade.declined and grade.grounded

        rows.append({
            "question": case["question"],
            "in_scope": in_scope,
            "retrieval_hit": retrieval_hit,
            "grounded": grade.grounded,
            "correct": grade.correct,
            "declined": grade.declined,
            "passed": passed,
            "seconds": seconds,
            "answer": result["answer"],
            "judge_reason": grade.reason,
        })
        mark = "PASS" if passed else "FAIL"
        print(f"[{i:2}/{len(cases)}] {mark}  {case['question']}")
        if not passed:
            print(f"         answer: {result['answer']}")
            print(f"         judge:  {grade.reason}")

    # --- Summary (all computed in code) ---
    in_scope_rows = [r for r in rows if r["in_scope"]]
    out_scope_rows = [r for r in rows if not r["in_scope"]]

    def pct(part: int, whole: int) -> float:
        return round(100 * part / whole, 1) if whole else 0.0

    summary = {
        "model": f"{config.LLM_PROVIDER} / {config.MODEL}",
        "total_questions": len(rows),
        "pass_rate": pct(sum(r["passed"] for r in rows), len(rows)),
        "retrieval_hit_rate": pct(sum(r["retrieval_hit"] for r in in_scope_rows), len(in_scope_rows)),
        "grounded_rate": pct(sum(r["grounded"] for r in rows), len(rows)),
        "correct_rate_in_scope": pct(sum(r["correct"] for r in in_scope_rows), len(in_scope_rows)),
        "out_of_scope_declined_rate": pct(sum(r["declined"] for r in out_scope_rows), len(out_scope_rows)),
        "avg_seconds_per_answer": round(sum(r["seconds"] for r in rows) / len(rows), 1),
    }

    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key:28} {value}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"eval_{datetime.now():%Y-%m-%d_%H-%M}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nSaved full results to {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
