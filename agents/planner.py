"""
agents/planner.py

The Planner agent reasons over the user's message (and, later, the
Screener agent's distress signal) to produce a structured classification:
risk level, primary emotion, and detected language. This is the schema
everything downstream branches on — Retriever's condition_tag filter,
Referral's trigger condition, and Evaluator's checks all depend on
Planner's output being reliably structured.

Pure LLM reasoning — no external tool file needed.
"""

from enum import Enum

from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field

from guardrails.task_guardrails import make_planner_guardrail



planner_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0.2,  # low temperature: we want consistent classification, not creativity
)


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class ConditionTag(str, Enum):
    """
    MUST exactly match the unique condition_tag values that actually exist
    in ingestion/master_rag_documents.csv. If this list and the CSV ever
    drift apart, Retriever's `where={"condition_tag": ...}` filter will
    silently return zero results for the mismatched value — so update
    both together if new categories are added to the knowledge base.
    """
    DEPRESSION = "depression"
    PTSD = "ptsd"
    BIPOLAR = "bipolar"
    SCHIZOPHRENIA = "schizophrenia"
    EATING_DISORDER = "eating_disorder"
    ADHD = "adhd"
    ANXIETY = "anxiety"
    GENERAL = "general"
    CRISIS = "crisis"
    OCD = "ocd"
    STRESS = "stress"
    BURNOUT = "burnout"
    GRIEF = "grief"


class PlannerOutput(BaseModel):
    risk_level: RiskLevel = Field(
        description="LOW: general questions/info-seeking. MODERATE: expressing "
                    "distress, sadness, anxiety, struggling but no self-harm/suicide "
                    "indication. HIGH: any mention of suicide, self-harm, wanting to "
                    "die, or being in immediate crisis."
    )
    primary_emotion: str = Field(
        description="The single dominant emotion expressed, e.g. 'hopelessness', "
                    "'anxiety', 'anger', 'sadness', 'confusion', 'neutral'."
    )
    detected_language: str = Field(
        description="The language the user wrote in, as an ISO 639-1 code "
                    "(e.g. 'en', 'hi', 'mr')."
    )
    condition_tag: ConditionTag = Field(
        description="Best-matching condition category from the fixed list of "
                    "categories that exist in the knowledge base. If the message "
                    "involves suicidal ideation or immediate crisis, use 'crisis' "
                    "rather than inventing a combined label. If nothing else fits, "
                    "use 'general'."
    )
    reasoning: str = Field(
        description="One or two sentences explaining why this risk level and "
                    "emotion were assigned. Internal use only, not shown to user."
    )



planner_agent = Agent(
    role="Risk & Emotion Planner",
    goal=(
        "Accurately classify the risk level, dominant emotion, language, and "
        "likely condition category of a user's message, so downstream agents "
        "can retrieve the right information and respond appropriately to the "
        "level of care needed."
    ),
    backstory=(
        "A careful clinical triage assistant trained to read messages the way "
        "a mental health first-responder would: never dismissive of distress, "
        "never alarmist about ordinary struggles, and always precise about "
        "the difference between someone who is sad and someone who may be in "
        "danger. Errs toward caution when signals are ambiguous — a message "
        "that could plausibly indicate self-harm risk is always flagged HIGH "
        "rather than downgraded on uncertainty."
    ),
    llm=planner_llm,
    verbose=True,
    allow_delegation=False,
)



def build_planner_task(user_message: str, screener_signal: str = "",
                        conversation_history: str = "") -> Task:
    """
    Creates a Planner Task for a specific user message.

    Args:
        user_message: the raw user input (already passed the Input guardrail)
        screener_signal: optional distress signal from the Screener agent
                          (e.g. "MentalBERT flagged: high distress language").
                          Empty string if Screener hasn't run / isn't wired in yet.
        conversation_history: recent prior turns, formatted as
                               "User: ...\\nAssistant: ...\\n...", oldest first.
                               Empty string for a fresh conversation.
    """
    signal_context = (
        f"\n\nAdditional signal from the Screener agent: {screener_signal}"
        if screener_signal else
        "\n\n(No Screener signal available — classify based on message alone.)"
    )

    history_context = (
        f"\n\nRecent conversation so far (for context only — classify based "
        f"on the LATEST user message below, using this history to understand "
        f"what's already been discussed):\n{conversation_history}\n"
        if conversation_history else ""
    )

    return Task(
        description=(
            f"Analyze the following user message and classify it.\n\n"
            f'User message: "{user_message}"'
            f"{signal_context}"
            f"{history_context}\n\n"
            "Return your classification strictly according to the required "
            "output schema. Be especially careful with risk_level: if there is "
            "ANY indication of suicidal ideation, self-harm, or wanting to die "
            "— even indirect or ambiguous phrasing — classify as HIGH. Do not "
            "downgrade risk based on politeness or calm tone; distress can be "
            "expressed calmly.\n\n"
            "condition_tag MUST be exactly one of: depression, ptsd, bipolar, "
            "schizophrenia, eating_disorder, adhd, anxiety, general, crisis, "
            "ocd, stress, burnout, grief. Use 'crisis' for suicidal ideation "
            "or immediate danger rather than combining categories."
        ),
        expected_output="A structured classification matching the PlannerOutput schema.",
        agent=planner_agent,
        output_pydantic=PlannerOutput,
        guardrail=make_planner_guardrail(user_message),
    )



if __name__ == "__main__":
    test_messages = [
        "What is depression and how common is it?",
        "I've been feeling really anxious lately and can't stop worrying about work.",
        "I feel like there's no point in anything anymore and I don't want to be here.",
        "I can't sleep and I keep thinking everyone would be better off without me.",
    ]

    for msg in test_messages:
        print(f"\n{'=' * 70}")
        print(f"Message: {msg!r}")
        print("=" * 70)

        task = build_planner_task(msg)
        crew = Crew(
            agents=[planner_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        output: PlannerOutput = result.pydantic
        print(f"  risk_level:        {output.risk_level.value}")
        print(f"  primary_emotion:   {output.primary_emotion}")
        print(f"  detected_language: {output.detected_language}")
        print(f"  condition_tag:     {output.condition_tag.value}")
        print(f"  reasoning:         {output.reasoning}")
