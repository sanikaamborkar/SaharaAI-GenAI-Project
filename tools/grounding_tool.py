"""
tools/grounding_tool.py

Assembles Retriever's passages and Planner's classification into a single,
clean grounding context block for the Worker agent's prompt. This is pure
prompt-assembly logic — no external API call — but kept as its own tool
file so grounding presentation is testable and reusable on its own.
"""


def build_grounding_context(retrieved_passages: str, primary_emotion: str,
                             risk_level: str, condition_tag: str) -> str:
    """
    Combines Retriever's raw passage output with Planner's classification
    into a single instructional context block for the Worker agent.

    Args:
        retrieved_passages: formatted output from tools.rag_search_tool.search()
        primary_emotion: Planner's detected primary emotion
        risk_level: Planner's risk classification (LOW/MODERATE/HIGH)
        condition_tag: Planner's detected condition category
    """
    risk_guidance = {
        "LOW": (
            "This is a general/informational question. Answer clearly and "
            "warmly using the grounding passages below."
        ),
        "MODERATE": (
            "The user is expressing real distress but is not in immediate "
            "crisis. Acknowledge their feelings first, briefly and genuinely, "
            "before offering grounded information. Do not rush to information "
            "before acknowledging the emotion."
        ),
        "HIGH": (
            "The user may be in crisis or expressing thoughts of self-harm or "
            "suicide. Prioritize a calm, caring, non-judgmental acknowledgment "
            "of their pain. Do NOT provide a specific helpline number or "
            "clinical diagnosis yourself — a Referral agent will supply the "
            "correct regional helpline separately. Keep your response short, "
            "warm, and focused on making the person feel heard and not alone."
        ),
    }.get(risk_level, "Answer using the grounding passages below.")

    return (
        f"--- CLASSIFICATION ---\n"
        f"Detected emotion: {primary_emotion}\n"
        f"Risk level: {risk_level}\n"
        f"Condition category: {condition_tag}\n\n"
        f"--- RESPONSE GUIDANCE ---\n"
        f"{risk_guidance}\n\n"
        f"--- GROUNDING PASSAGES (use these, don't invent facts) ---\n"
        f"{retrieved_passages}"
    )
