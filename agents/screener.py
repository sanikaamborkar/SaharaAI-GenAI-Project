"""
agents/screener.py

The Screener agent's job is simple: run MentalBERT on the message and
hand the signal to Planner. Because this is a deterministic classifier
call with no actual reasoning required, this file provides TWO ways to
use it:

1. `screener_agent` — a proper CrewAI Agent, for architectural
   consistency with the rest of the pipeline (useful if you want
   Screener to appear as a step in crew.py's process/logs).

2. `get_screener_signal()` — a direct function that calls the tool's
   underlying classification without spinning up an LLM call at all.

RECOMMENDATION: use `get_screener_signal()` in the actual pipeline.
Screener doesn't need an LLM to decide anything — routing this through
a full CrewAI Agent turn would mean paying for an extra llama3.1 call
just to have it call one tool and repeat the result back, which adds
latency for zero benefit. The Agent object below is kept for
completeness / if you want it to show up in crew.py's visible steps,
but the fast path is what should actually run in production.
"""

from crewai import Agent

from tools.mentalbert_tool import mentalbert_screen, _classify



def get_screener_signal(message: str) -> str:
    """
    Runs MentalBERT directly and returns the formatted signal string,
    exactly as `mentalbert_screen` would, but without any CrewAI/LLM
    overhead. This is what should feed into Planner's screener_signal
    parameter in the real pipeline.
    """
    return mentalbert_screen.func(message)


screener_agent = Agent(
    role="Distress Screener",
    goal=(
        "Run the MentalBERT classifier on every incoming message and "
        "report the full distress signal — predicted level, score "
        "distribution, and any uncertainty flags — without editorializing "
        "or overriding it."
    ),
    backstory=(
        "A fast, first-pass screening assistant. Does not reason about the "
        "message itself — simply runs the classifier tool and reports its "
        "output exactly, including uncertainty warnings, so the Planner "
        "agent downstream has the full picture rather than a single "
        "flattened label."
    ),
    tools=[mentalbert_screen],
    verbose=True,
    allow_delegation=False,
)


if __name__ == "__main__":
    test_messages = [
        "I had a great day at college today!",
        "I feel really overwhelmed and can't cope with things lately.",
        "I don't want to be here anymore.",
        "I keep thinking there's no point but I also don't want to die.",
    ]

    print("Testing fast path (get_screener_signal) — no LLM overhead:\n")
    for msg in test_messages:
        print(f"'{msg}'")
        print(f"  -> {get_screener_signal(msg)}\n")
