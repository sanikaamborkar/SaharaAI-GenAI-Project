"""
crew.py

Wires the full SereneShield pipeline together into one function:
Input guardrail -> Screener -> Planner -> Retriever -> Worker ->
Evaluator -> Referral (if HIGH) -> final response.

This is the single entry point app.py should call. It also introduces
CONVERSATION HISTORY support: a simple list of {"role", "content"} dicts
that the caller (app.py, owning the Streamlit session) maintains and
passes in on every call. History is kept as the last N turns (not
summarized) — simple and sufficient for now; can be upgraded to a
rolling summary later if conversations get long enough for that to matter.
"""

import os

# MUST be set before `import crewai` — CrewAI reads these at import time to
# decide whether to initialize its telemetry/tracing client. Without this,
# every agent call wastes 15-20+ seconds retrying failed uploads to
# app.crewai.com with exponential backoff, on top of real inference time.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Crew, Process

from guardrails.input_guardrail import check_input
from agents.screener import get_screener_signal
from agents.planner import planner_agent, build_planner_task, PlannerOutput, RiskLevel
from tools.rag_search_tool import search
from tools.grounding_tool import build_grounding_context
from agents.worker import worker_agent, build_worker_task
from agents.evaluator import evaluator_agent, build_evaluator_task, EvaluatorOutput, get_final_reply
from agents.referral import get_referral_message

# How many recent turns of conversation history to include as context.
# Kept simple (last N turns, not summarized) — sufficient for now.
MAX_HISTORY_TURNS = 6


def _format_history(conversation_history: list) -> str:
    """
    Formats recent conversation turns into a short text block for Planner/
    Worker context. Expects a list of {"role": "user"/"assistant",
    "content": "..."} dicts, oldest first. Returns "" if no history.
    """
    if not conversation_history:
        return ""

    recent = conversation_history[-MAX_HISTORY_TURNS:]
    lines = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent]
    return "\n".join(lines)


def run_pipeline(user_message: str, conversation_history: list = None,
                  country: str = "India") -> dict:
    """
    Runs the full pipeline for a single user message.

    Args:
        user_message: the raw text the user typed
        conversation_history: optional list of {"role", "content"} dicts
                               for prior turns in this session (oldest first).
                               Pass None or [] for a fresh conversation.
        country: user's country, for Referral's helpline lookup. Currently
                 must be supplied by the caller (e.g. asked once at session
                 start in app.py) — the pipeline does not detect location.

    Returns:
        dict with keys:
          - "reply": the final text to show the user
          - "blocked": True if the Input guardrail rejected the message
                       (in which case "reply" is the block reason)
          - "risk_level": Planner's classification, or None if blocked
          - "debug": a dict of intermediate values, useful for logging/testing
    """
    # --- Step 0: Input guardrail — runs before anything else ---
    guardrail_result = check_input(user_message)
    if not guardrail_result.passed:
        return {
            "reply": guardrail_result.reason,
            "blocked": True,
            "risk_level": None,
            "debug": {"blocked_reason": guardrail_result.reason},
        }

    history_text = _format_history(conversation_history or [])

    # --- Step 1: Screener (fast path, no LLM overhead) ---
    screener_signal = get_screener_signal(user_message)

    # --- Step 2: Planner ---
    planner_task = build_planner_task(
        user_message, screener_signal=screener_signal,
        conversation_history=history_text,
    )
    planner_crew = Crew(
        agents=[planner_agent], tasks=[planner_task],
        process=Process.sequential, verbose=False,
    )
    classification: PlannerOutput = planner_crew.kickoff().pydantic

    # --- Step 3: Retriever ---
    passages = search(user_message, condition_tag=classification.condition_tag.value)

    # --- Step 4: Worker ---
    context = build_grounding_context(
        retrieved_passages=passages,
        primary_emotion=classification.primary_emotion,
        risk_level=classification.risk_level.value,
        condition_tag=classification.condition_tag.value,
    )
    worker_task = build_worker_task(user_message, context, conversation_history=history_text)
    worker_crew = Crew(
        agents=[worker_agent], tasks=[worker_task],
        process=Process.sequential, verbose=False,
    )
    worker_reply = worker_crew.kickoff().raw

    # --- Step 5: Evaluator ---
    # Skip the LLM call entirely for LOW risk — informational questions
    # carry much less safety stakes than MODERATE/HIGH, so paying for a
    # full evaluation LLM call on every single one is disproportionate.
    # This is a real latency win: one fewer sequential LLM call for what
    # is usually the most common message type.
    if classification.risk_level == RiskLevel.LOW:
        final_reply = worker_reply
        evaluation = EvaluatorOutput(
            approved=True, issues_found=[],
            reasoning="Evaluator skipped for LOW risk (informational message).",
        )
    else:
        eval_task = build_evaluator_task(
            user_message, worker_reply,
            risk_level=classification.risk_level.value,
            primary_emotion=classification.primary_emotion,
            retrieved_passages=passages,
        )
        eval_crew = Crew(
            agents=[evaluator_agent], tasks=[eval_task],
            process=Process.sequential, verbose=False,
        )
        evaluation: EvaluatorOutput = eval_crew.kickoff().pydantic
        final_reply = get_final_reply(worker_reply, evaluation)

    # --- Step 6: Referral (only if HIGH risk) ---
    if classification.risk_level == RiskLevel.HIGH:
        final_reply += get_referral_message(country=country)

    return {
        "reply": final_reply,
        "blocked": False,
        "risk_level": classification.risk_level.value,
        "debug": {
            "screener_signal": screener_signal,
            "primary_emotion": classification.primary_emotion,
            "condition_tag": classification.condition_tag.value,
            "planner_reasoning": classification.reasoning,
            "worker_draft": worker_reply,
            "evaluator_approved": evaluation.approved,
            "evaluator_issues": evaluation.issues_found,
        },
    }


# ---------------------------------------------------------------------------
# Standalone test — simulates a short multi-turn conversation.
# Run from project root: python -m crew
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    history = []

    test_messages = [
        "What is anxiety?",
        "I've been feeling really anxious lately and can't stop worrying about work.",
        "I feel like there's no point in anything anymore and I don't want to be here.",
        "",  # tests the Input guardrail block path
    ]

    for msg in test_messages:
        print("\n" + "#" * 70)
        print(f"User: {msg!r}")
        print("#" * 70)

        result = run_pipeline(msg, conversation_history=history, country="India")

        print(f"\nblocked={result['blocked']} | risk_level={result['risk_level']}")
        print(f"\nReply:\n{result['reply']}")

        if not result["blocked"]:
            history.append({"role": "user", "content": msg})
            history.append({"role": "assistant", "content": result["reply"]})
