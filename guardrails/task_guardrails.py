"""
guardrails/task_guardrails.py

CrewAI Task-level guardrails: functions attached to a Task's `guardrail`
parameter. Each receives the task's output and returns either:
  - (True, validated_output)  -> output passes, used as-is
  - (False, error_message)    -> output rejected, CrewAI automatically
                                   retries the task, feeding the agent
                                   `error_message` so it can self-correct

These are a safety NET underneath each agent's own reasoning — they catch
structural/safety failures the LLM might slip past its own instructions,
rather than replacing the instructions themselves.
"""

import re

# Unambiguous high-risk phrases. If any of these appear in the raw user
# message, Planner's risk_level MUST be HIGH — this is a hard backstop,
# not a suggestion, regardless of how the LLM reasoned about tone/context.
HARD_HIGH_RISK_PHRASES = [
    "kill myself", "end my life", "want to die", "don't want to be here",
    "don't want to live", "better off without me", "no reason to live",
    "want to end it", "suicide", "kill me",
]

# Rough phone-number-like pattern: sequences of digits (with optional
# spaces/dashes) that look like a contact number. Worker should NEVER
# produce one of these — that's Referral's job alone, using verified data.
PHONE_NUMBER_PATTERN = re.compile(r"(\+?\d[\d\s\-]{6,}\d)")

# Patterns indicating the model leaked its own prompt instructions into the
# reply instead of just following them silently (e.g. "(Note: if risk level
# is HIGH...)"). Caught here so Worker gets a cheap, fast retry instead of
# the whole reply being discarded later by Evaluator.
META_LEAKAGE_PATTERNS = [
    r"\(note:",
    r"as (an? )?ai",
    r"as instructed",
    r"per (the |these )?instructions",
    r"the instructions (say|state|indicate)",
    r"if (the )?risk level is",
]

MIN_WORKER_REPLY_LENGTH = 20  # characters — catches suspiciously empty replies


def make_planner_guardrail(user_message: str):
    """
    Factory that builds a guardrail bound to THIS specific user message only.
    This matters: the guardrail must never scan the full rendered task
    description, since that includes conversation history and instructions
    — and a prior turn's Referral helpline text (e.g. "Suicide Prevention")
    would otherwise falsely trigger on every later message in the same
    conversation, regardless of what the user actually just said.
    """
    text_lower = user_message.lower()

    def _guardrail(task_output):
        try:
            result = task_output.pydantic
            if result is None:
                return (False, "Output did not match the required schema. "
                                "Return a valid structured classification.")

            matched_phrase = next(
                (phrase for phrase in HARD_HIGH_RISK_PHRASES if phrase in text_lower),
                None
            )

            if matched_phrase and result.risk_level.value != "HIGH":
                return (
                    False,
                    f"The user's message contains the phrase '{matched_phrase}', "
                    f"which is an unambiguous high-risk indicator. risk_level "
                    f"MUST be HIGH and condition_tag MUST be 'crisis'. "
                    f"Re-classify accordingly."
                )

            return (True, result)

        except Exception as e:
            return (False, f"Guardrail check failed to process output: {e}")

    return _guardrail


def worker_guardrail(task_output):
    """
    Safety-net check on Worker's drafted reply. Rejects if it contains
    anything resembling a phone number (Referral's job alone), leaked
    meta-instructions/self-commentary, or is suspiciously short/empty.
    """
    try:
        reply_text = task_output.raw or ""
        text_lower = reply_text.lower()

        if len(reply_text.strip()) < MIN_WORKER_REPLY_LENGTH:
            return (False, "Reply is too short or empty. Write a complete, "
                            "warm response to the user's message.")

        if PHONE_NUMBER_PATTERN.search(reply_text):
            return (False, "Reply appears to contain a phone number or "
                            "contact detail. NEVER include a specific phone "
                            "number — helpline information is handled "
                            "separately by the Referral step. Rewrite the "
                            "reply without any number sequences.")

        for pattern in META_LEAKAGE_PATTERNS:
            if re.search(pattern, text_lower):
                return (False, "Reply leaked internal instructions or "
                                "self-commentary into the message (e.g. "
                                "'(Note: ...)' or explaining your own rules). "
                                "Rewrite the reply to contain ONLY the "
                                "actual message to the user — no mention "
                                "of instructions, rules, or your own "
                                "reasoning about them.")

        return (True, reply_text)

    except Exception as e:
        return (False, f"Guardrail check failed to process output: {e}")


def evaluator_guardrail(task_output):
    """
    Structural check on Evaluator's output: if it rejects a reply
    (approved=False), it must actually list at least one issue. An
    empty issues_found on a rejection is an internal inconsistency,
    not a valid evaluation.
    """
    try:
        result = task_output.pydantic
        if result is None:
            return (False, "Output did not match the required schema. "
                            "Return a valid structured evaluation.")

        if not result.approved and not result.issues_found:
            return (
                False,
                "You rejected the reply (approved=False) but issues_found is "
                "empty. If you reject, you MUST list at least one specific "
                "issue. If there's no real issue, set approved=True instead."
            )

        return (True, result)

    except Exception as e:
        return (False, f"Guardrail check failed to process output: {e}")
