"""
agents/evaluator.py

The Evaluator agent is the final output guardrail — it checks Worker's
drafted reply BEFORE anything reaches the user. It does NOT evaluate
Referral's helpline block, since that's a deterministic, pre-verified
lookup (not LLM-generated) with nothing for a safety check to usefully
catch.

Design choice: Evaluator either APPROVES the reply as-is, or REJECTS it
and substitutes a safe, generic fallback message. It does not attempt to
"fix" or rewrite Worker's text — having one LLM patch another LLM's
output is a good way to introduce a second set of problems. Binary
approve/replace is safer and more predictable.
"""

from enum import Enum

from crewai import Agent, Task, LLM
from pydantic import BaseModel, Field

from guardrails.task_guardrails import evaluator_guardrail



evaluator_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0.1,
)



SAFE_FALLBACK_MESSAGE = (
    "I can tell this matters to you, and I want to make sure I respond in "
    "a way that's genuinely helpful and safe.\n\n"
    "What you're feeling is valid, and you don't have to go through it "
    "alone. I'd also gently encourage reaching out to a mental health "
    "professional or someone you trust, alongside talking with me here.\n\n"
    "Could you tell me a little more about what's on your mind? I'm here "
    "to listen."
)



class EvaluatorOutput(BaseModel):
    approved: bool = Field(
        description="True if Worker's reply is safe, accurate, and appropriate "
                    "to send to the user as-is. False if it has a real problem."
    )
    issues_found: list[str] = Field(
        default_factory=list,
        description="Specific problems found, if any. Empty list if approved. "
                    "E.g. ['contains an invented phone number', 'diagnoses the "
                    "user with GAD', 'tone too clinical for HIGH risk message']."
    )
    reasoning: str = Field(
        description="One or two sentences explaining the approve/reject decision."
    )



evaluator_agent = Agent(
    role="Final Safety Evaluator",
    goal=(
        "Review the drafted reply before it reaches the user, catching any "
        "invented facts, invented contact information, unauthorized "
        "diagnosis, or tone mismatched to the user's risk level — as the "
        "last checkpoint before anything is sent."
    ),
    backstory=(
        "A meticulous, calm reviewer — the last line of defense before a "
        "response reaches someone who may be vulnerable. Does not rewrite "
        "or improve replies; only approves or rejects. When in doubt, "
        "rejects rather than lets a questionable reply through, since a "
        "safe generic fallback is always better than a risky specific one."
    ),
    llm=evaluator_llm,
    verbose=True,
    allow_delegation=False,
)



def build_evaluator_task(user_message: str, worker_reply: str, risk_level: str,
                          primary_emotion: str, retrieved_passages: str = "") -> Task:
    """
    Creates an Evaluator Task to check Worker's drafted reply.

    Args:
        user_message: the original user input
        worker_reply: Worker's drafted reply (BEFORE any Referral block is appended)
        risk_level: Planner's risk classification (LOW/MODERATE/HIGH)
        primary_emotion: Planner's detected primary emotion
        retrieved_passages: the SAME grounding passages Worker was given, so
                             Evaluator can actually verify claims against
                             source material instead of guessing blind
    """
    passages_block = (
        f"--- SOURCE PASSAGES WORKER WAS GROUNDED IN ---\n{retrieved_passages}\n\n"
        if retrieved_passages else
        "(No source passages available for this check.)\n\n"
    )

    return Task(
        description=(
            f'Original user message: "{user_message}"\n\n'
            f"User's risk level: {risk_level} | detected emotion: {primary_emotion}\n\n"
            f"{passages_block}"
            f'Drafted reply to review:\n"""\n{worker_reply}\n"""\n\n'
            "Check this reply for the following. Be precise — only reject for "
            "a REAL, clear violation. Do not invent a problem just to have "
            "something to flag; if the reply is reasonably safe and grounded, "
            "APPROVE it.\n\n"
            "1. INVENTED FACTS: Compare specific claims in the reply against "
            "the source passages above. Only reject if the reply states a "
            "specific fact, statistic, or claim that is NOT supported by the "
            "passages AND is not a normal, generic supportive statement. "
            "Generic supportive phrases like 'many people feel this way' or "
            "'you're not alone in this' are NOT invented facts — do not reject "
            "for these. Paraphrased content from the passages (even loosely "
            "worded) is NOT invented — only reject if the claim contradicts "
            "or has no basis at all in the passages.\n"
            "2. DIAGNOSIS: Only reject if the reply explicitly tells THE USER "
            "they have/are experiencing a specific named condition (e.g. "
            "'you have GAD', 'you are depressed'). Explaining what a "
            "condition generally is, or saying the user 'might be struggling "
            "with anxiety' in a soft, general way, is NOT a diagnosis — do "
            "NOT reject for this.\n"
            "3. INVENTED CONTACT INFO: Reject only if a specific phone number, "
            "helpline name, or contact detail appears in the reply itself.\n"
            "4. TONE: For HIGH risk, reject ONLY if the reply is genuinely "
            "long (more than ~150 words), reads like an information pamphlet, "
            "or fails to acknowledge the person's feelings before anything "
            "else. A short, warm reply that acknowledges pain and gently "
            "offers to help is GOOD tone for HIGH risk — do not reject it.\n"
            "5. GENERAL SAFETY: Reject only for a genuine safety concern "
            "(e.g. judgmental language, dismissiveness, harmful advice) — "
            "not a stylistic preference.\n"
            "6. META-LEAKAGE: Reject if the reply mentions its own "
            "instructions, includes notes like '(Note: if risk level is "
            "HIGH...)', or explains its own reasoning about what it should "
            "or shouldn't do. The reply must contain ONLY the actual "
            "message to the user — never commentary about how or why it "
            "was written.\n\n"
            "Default to APPROVE unless you can point to a specific, clear "
            "violation of one of the above."
        ),
        expected_output="A structured evaluation matching the EvaluatorOutput schema.",
        agent=evaluator_agent,
        output_pydantic=EvaluatorOutput,
        guardrail=evaluator_guardrail,
    )


def get_final_reply(worker_reply: str, evaluation: EvaluatorOutput) -> str:
    """
    Applies Evaluator's verdict: returns Worker's reply if approved,
    otherwise the safe fallback message.
    """
    return worker_reply if evaluation.approved else SAFE_FALLBACK_MESSAGE



if __name__ == "__main__":
    from crewai import Crew, Process

    from tools.rag_search_tool import search
    from tools.grounding_tool import build_grounding_context
    from agents.planner import planner_agent, build_planner_task, PlannerOutput, RiskLevel
    from agents.screener import get_screener_signal
    from agents.worker import worker_agent, build_worker_task
    from agents.referral import get_referral_message

    test_messages = [
        "I've been feeling really anxious lately and can't stop worrying about work.",
        "I feel like there's no point in anything anymore and I don't want to be here.",
    ]

    for test_message in test_messages:
        print("\n" + "#" * 70)
        print(f"Message: {test_message!r}")
        print("#" * 70)

        print("\n[0/5] Running Screener...")
        screener_signal = get_screener_signal(test_message)
        print(f"  {screener_signal}")

        print("\n[1/5] Running Planner...")
        planner_task = build_planner_task(test_message, screener_signal=screener_signal)
        planner_crew = Crew(agents=[planner_agent], tasks=[planner_task],
                             process=Process.sequential, verbose=False)
        classification: PlannerOutput = planner_crew.kickoff().pydantic
        print(f"  risk_level={classification.risk_level.value} | "
              f"emotion={classification.primary_emotion} | "
              f"condition={classification.condition_tag.value}")

        print("\n[2/5] Running Retriever...")
        passages = search(test_message, condition_tag=classification.condition_tag.value)
        print(f"  Retrieved {passages.count('[Passage')} passages")

        print("\n[3/5] Running Worker...")
        context = build_grounding_context(
            retrieved_passages=passages,
            primary_emotion=classification.primary_emotion,
            risk_level=classification.risk_level.value,
            condition_tag=classification.condition_tag.value,
        )
        worker_task = build_worker_task(test_message, context)
        worker_crew = Crew(agents=[worker_agent], tasks=[worker_task],
                            process=Process.sequential, verbose=False)
        worker_reply = worker_crew.kickoff().raw
        print(f"  Draft reply ({len(worker_reply)} chars) ready.")

        print("\n[4/5] Running Evaluator...")
        eval_task = build_evaluator_task(
            test_message, worker_reply,
            risk_level=classification.risk_level.value,
            primary_emotion=classification.primary_emotion,
            retrieved_passages=passages,
        )
        eval_crew = Crew(agents=[evaluator_agent], tasks=[eval_task],
                          process=Process.sequential, verbose=False)
        evaluation: EvaluatorOutput = eval_crew.kickoff().pydantic
        print(f"  approved={evaluation.approved} | issues={evaluation.issues_found}")
        print(f"  reasoning: {evaluation.reasoning}")

        final_reply = get_final_reply(worker_reply, evaluation)

        if classification.risk_level == RiskLevel.HIGH:
            print("\n[5/5] Risk is HIGH — running Referral...")
            final_reply += get_referral_message(country="India")
        else:
            print("\n[5/5] Risk is not HIGH — skipping Referral.")

        print("\n" + "=" * 70)
        print("FINAL REPLY (post-Evaluator):")
        print("=" * 70)
        print(final_reply)
