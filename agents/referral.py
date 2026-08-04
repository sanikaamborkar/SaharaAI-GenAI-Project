"""
agents/referral.py

The Referral agent only fires when Planner classifies risk_level as HIGH.
Its job is a deterministic lookup (find the right country's helplines),
not reasoning — so like Screener, this provides a fast no-LLM-overhead
path as the recommended way to call it in the real pipeline, plus an
Agent object for architectural completeness.
"""

from crewai import Agent

from tools.helpline_tool import helpline_lookup_tool, get_helplines, DEFAULT_COUNTRY



def get_referral_message(country: str = DEFAULT_COUNTRY) -> str:
    """
    Returns a formatted block of verified regional helplines, ready to be
    appended to Worker's reply. This should ONLY be called when Planner's
    risk_level == HIGH.
    """
    helplines = get_helplines(country=country)
    return (
        "\n\nIf things feel overwhelming right now, please know support is "
        f"available. {helplines}\n\n"
        "You don't have to go through this alone — reaching out to one of "
        "these is a strong, brave step, not a last resort."
    )



referral_agent = Agent(
    role="Crisis Referral Specialist",
    goal=(
        "When a user is at HIGH risk, provide verified, region-correct "
        "crisis helpline information so they have a real, immediate way "
        "to reach human support."
    ),
    backstory=(
        "A careful resource coordinator who never invents a phone number "
        "or contact detail — only ever reports what's verified in the "
        "knowledge base for the user's region. Understands that in a "
        "crisis moment, accuracy matters more than anything else: a wrong "
        "or outdated number could cost precious time."
    ),
    tools=[helpline_lookup_tool],
    verbose=True,
    allow_delegation=False,
)


if __name__ == "__main__":
    for country in ["India", "United States", "United Kingdom"]:
        print(f"\n{'=' * 60}")
        print(f"Country: {country}")
        print("=" * 60)
        print(get_referral_message(country))
