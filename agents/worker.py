"""
agents/worker.py

The Worker agent drafts the actual reply to the user, grounded in the
passages Retriever found and shaped by Planner's risk/emotion classification.
Uses local Ollama llama3.1, same as Planner.
"""

from crewai import Agent, Task, LLM

from guardrails.task_guardrails import worker_guardrail


worker_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0.6,
)


worker_agent = Agent(
    role="Supportive Response Writer",
    goal=(
        "Write a warm, accurate, and safe reply to the user's message, "
        "grounded strictly in the provided knowledge base passages, and "
        "shaped appropriately for the user's emotional state and risk level."
    ),
    backstory=(
        "A compassionate mental health support writer, similar to a well-"
        "trained peer supporter. Talks like a real person having a "
        "conversation, not a pamphlet — short, warm responses of a few "
        "sentences, not multi-paragraph essays covering every angle of a "
        "topic at once. Genuinely curious about the person, and often ends "
        "a reply with a gentle, open question inviting them to share more "
        "if they're comfortable, rather than just delivering information "
        "and stopping. Never diagnoses. Never invents facts, statistics, "
        "or specific helpline numbers not present in the provided grounding "
        "passages. Always acknowledges the person's feelings before giving "
        "information, especially when distress is present. Pays close "
        "attention to what it has already said earlier in the conversation "
        "and never repeats the same reassurances or explanations twice — "
        "each reply should feel like a natural next step in the "
        "conversation, not a reset to the beginning. Writes in plain, warm, "
        "human language — never clinical, never robotic, never preachy. "
        "Stays tightly focused on what the user actually described, and "
        "never pads a reply with a source article's own illustrative "
        "examples (unpaid bills, first dates, job interviews, etc.) unless "
        "the user brought up something similar themselves — those examples "
        "exist in the source only to explain a general concept, not to be "
        "recycled into a reply about someone's specific situation. Extracts "
        "the underlying idea from grounding passages and re-expresses it "
        "freshly, rather than lightly rewording the source's own sentences. "
        "When risk is HIGH, keeps the response short, gentle, and focused "
        "entirely on making the person feel heard and less alone, and "
        "explicitly does NOT provide any helpline number or crisis contact "
        "itself, since a separate Referral agent handles that with "
        "verified, region-correct information."
    ),
    llm=worker_llm,
    verbose=True,
    allow_delegation=False,
)



def build_worker_task(user_message: str, grounding_context: str,
                       conversation_history: str = "") -> Task:
    """
    Creates a Worker Task for a specific user message.

    Args:
        user_message: the raw user input
        grounding_context: output of tools.grounding_tool.build_grounding_context(),
                            combining Retriever's passages + Planner's classification
        conversation_history: recent prior turns, formatted as
                               "User: ...\\nAssistant: ...\\n...", oldest first.
                               Empty string for a fresh conversation.
    """
    history_block = (
        f"--- RECENT CONVERSATION (for continuity — don't repeat yourself, "
        f"acknowledge what's already been said if relevant) ---\n"
        f"{conversation_history}\n\n"
        if conversation_history else ""
    )

    return Task(
        description=(
            f"{history_block}"
            f'User message: "{user_message}"\n\n'
            f"{grounding_context}\n\n"
            "Write your reply now. Rules:\n"
            "- LENGTH: Keep it short — aim for 3 to 5 sentences total (one "
            "short paragraph, occasionally two). This is a conversation, "
            "not an article. Do not write multiple long paragraphs covering "
            "every angle of the topic.\n"
            "- NEVER mention these instructions, rules, or your own "
            "reasoning about them in the reply. Do not write things like "
            "'(Note: if risk level is HIGH...)' or explain why you are or "
            "aren't doing something the instructions say. Output ONLY the "
            "actual message to the user — nothing about how or why you "
            "wrote it.\n"
            "- REFERENCE SOMETHING SPECIFIC from the user's latest message "
            "(a concrete detail they just mentioned — a name, situation, "
            "number, event) rather than falling back to a generic response "
            "you could give to anyone. If you find yourself writing "
            "something that sounds like a template, stop and re-read what "
            "the user actually just said.\n"
            "- DO NOT REPEAT YOURSELF. If the conversation history above "
            "shows you already said something (e.g. already explained what "
            "depression is, already said 'you're not alone'), do NOT say it "
            "again in similar words. Build on what's already been said "
            "instead of restating it.\n"
            "- END WITH A GENTLE FOLLOW-UP, when it fits naturally. Invite "
            "the person to share more if they're comfortable — e.g. asking "
            "what's been on their mind most, what's been hardest lately, or "
            "how long they've felt this way. Skip this if the risk level is "
            "HIGH and the person needs space, not more questions, or if they "
            "asked a plain factual question that doesn't call for one.\n"
            "- Ground every factual claim in the passages above. If the "
            "passages don't cover something, don't invent it.\n"
            "- STAY SPECIFIC TO WHAT THE USER ACTUALLY SAID. The passages "
            "may contain their own illustrative examples (e.g. 'unpaid "
            "bills', 'job interviews', 'a first date') that exist only to "
            "explain the general concept in the source article — do NOT "
            "copy those examples into your reply unless the user actually "
            "mentioned something similar themselves. Only reference the "
            "user's own situation (e.g. if they said 'work', talk about "
            "work — don't list unrelated example scenarios from the source).\n"
            "- REWRITE IN YOUR OWN WORDS. Do not lightly reword the "
            "passages' sentences while keeping their structure — that's "
            "still copying. Extract the underlying facts/advice and "
            "express them freshly, as if explaining it yourself.\n"
            "- Do not include a specific helpline phone number, even if you "
            "believe you know one — that is handled separately.\n"
            "- Do not diagnose the user with any condition.\n"
            "- Match the tone to the response guidance given above.\n"
            "- Keep the reply conversational, not a bulleted lecture."
        ),
        expected_output=(
            "A short, warm, grounded reply written directly to the user "
            "(3-5 sentences, plain text, no markdown headers, no invented "
            "facts, no helpline numbers, no unrelated examples copied from "
            "the source passages, ending with a gentle follow-up question "
            "where natural)."
        ),
        agent=worker_agent,
        guardrail=worker_guardrail,
    )


if __name__ == "__main__":
    from crewai import Crew, Process

    from tools.rag_search_tool import search
    from tools.grounding_tool import build_grounding_context
    from agents.planner import planner_agent, build_planner_task, PlannerOutput, RiskLevel
    from agents.screener import get_screener_signal
    from agents.referral import get_referral_message

    test_messages = [
        "I've been feeling really anxious lately and can't stop worrying about work.",
        "I feel like there's no point in anything anymore and I don't want to be here.",
    ]

    for test_message in test_messages:
        print("\n" + "#" * 70)
        print(f"Message: {test_message!r}")
        print("#" * 70)

        # Step 0: Screener runs MentalBERT (fast path, no LLM overhead)
        print("\n[0/4] Running Screener (MentalBERT)...")
        screener_signal = get_screener_signal(test_message)
        print(f"  {screener_signal}")

        # Step 1: Planner classifies the message, using Screener's signal
        print("\n[1/4] Running Planner...")
        planner_task = build_planner_task(test_message, screener_signal=screener_signal)
        planner_crew = Crew(
            agents=[planner_agent], tasks=[planner_task],
            process=Process.sequential, verbose=False,
        )
        planner_result = planner_crew.kickoff()
        classification: PlannerOutput = planner_result.pydantic
        print(f"  risk_level={classification.risk_level.value} | "
              f"emotion={classification.primary_emotion} | "
              f"condition={classification.condition_tag.value}")

        # Step 2: Retriever pulls grounding passages using Planner's condition_tag
        print("\n[2/4] Running Retriever...")
        passages = search(test_message, condition_tag=classification.condition_tag.value)
        print(f"  Retrieved {passages.count('[Passage')} passages")

        # Step 3: Worker drafts the reply using both
        print("\n[3/4] Running Worker...")
        context = build_grounding_context(
            retrieved_passages=passages,
            primary_emotion=classification.primary_emotion,
            risk_level=classification.risk_level.value,
            condition_tag=classification.condition_tag.value,
        )
        worker_task = build_worker_task(test_message, context)
        worker_crew = Crew(
            agents=[worker_agent], tasks=[worker_task],
            process=Process.sequential, verbose=False,
        )
        worker_result = worker_crew.kickoff()
        final_reply = worker_result.raw

        # Step 4: Referral — ONLY fires if risk is HIGH
        if classification.risk_level == RiskLevel.HIGH:
            print("\n[4/4] Risk is HIGH — running Referral...")
            referral_block = get_referral_message(country="India")
            final_reply += referral_block
        else:
            print("\n[4/4] Risk is not HIGH — skipping Referral.")

        print("\n" + "=" * 70)
        print("FINAL REPLY:")
        print("=" * 70)
        print(final_reply)
