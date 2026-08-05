"""
tools/helpline_tool.py

Looks up regional crisis helplines directly from the same CSV used for
ingestion (master_rag_documents.csv), filtered to resource_type in
['helpline', 'helpline_info']. This is a deterministic filter/lookup,
NOT a semantic search — no Chroma/embeddings needed, just pandas.

With 300+ helpline rows covering many issue types (domestic violence,
gambling, eating disorders, etc.) per country, this prioritizes lines
whose section text actually matches crisis/suicide keywords, so Referral
doesn't hand back an irrelevant helpline (e.g. a gambling hotline) when
someone is in a mental health crisis.
"""

import os
import re

import pandas as pd
from crewai.tools import tool


CSV_PATH = os.path.join("ingestion", "master_rag_documents.csv")
HELPLINE_RESOURCE_TYPES = ["helpline", "helpline_info"]

# Keywords that indicate a helpline is specifically crisis/suicide-relevant,
# not just any mental-health-adjacent hotline. Checked against the `section`
# column (which contains the helpline name/description).
CRISIS_KEYWORDS = [
    "suicide", "crisis", "distress", "lifeline", "mental health",
    "emotional support", "self-harm", "depression",
]


DEFAULT_COUNTRY = "India"

MAX_RESULTS = 3


def _load_helplines() -> pd.DataFrame:
    """Loads and filters the helpline rows from the ingestion CSV."""
    df = pd.read_csv(CSV_PATH)
    return df[df["resource_type"].isin(HELPLINE_RESOURCE_TYPES)].copy()


def _crisis_relevance_score(section_text: str) -> int:
    """Counts how many crisis-related keywords appear in the helpline's name/description."""
    if not isinstance(section_text, str):
        return 0
    text = section_text.lower()
    return sum(1 for kw in CRISIS_KEYWORDS if kw in text)


def get_helplines(country: str = DEFAULT_COUNTRY, max_results: int = MAX_RESULTS) -> str:
    """
    Returns a formatted string of the most crisis-relevant helplines for
    a given country. Falls back to a generic message if the country isn't
    in the dataset.

    Args:
        country: country name, must match the CSV's `country` column
                  (e.g. "India", "United States", "United Kingdom")
        max_results: how many helplines to return
    """
    df = _load_helplines()

    country_matches = df[df["country"].str.lower() == country.lower()]

    if country_matches.empty:
        return (
            f"No specific helpline data available for '{country}' in the "
            f"local database. Recommend directing the user to "
            f"findahelpline.com to search for helplines in their region, "
            f"or, if in the US, the 988 Suicide & Crisis Lifeline (dial 988) "
            f"as a widely-known international-adjacent option."
        )

    # Drop rows with no phone number — not useful in a crisis context
    country_matches = country_matches[country_matches["phone"].notna()]

    if country_matches.empty:
        return f"Helpline entries exist for '{country}' but none have a listed phone number."

    # Score and sort by crisis relevance, most relevant first
    country_matches = country_matches.copy()
    country_matches["_relevance"] = country_matches["section"].apply(_crisis_relevance_score)
    country_matches = country_matches.sort_values("_relevance", ascending=False)

    top = country_matches.head(max_results)

    lines = []
    for _, row in top.iterrows():
        name = re.sub(r"^Helpline:\s*", "", str(row["section"])).strip()
        phone = str(row["phone"]).strip()
        lines.append(f"- {name}: {phone}")

    return f"Verified helplines for {country}:\n" + "\n".join(lines)


@tool("Helpline Lookup")
def helpline_lookup_tool(country: str = DEFAULT_COUNTRY) -> str:
    """
    Looks up verified, region-specific crisis helplines for a given
    country from the knowledge base. Use this ONLY when risk level is
    HIGH, to provide the user with real, verified contact information —
    never invent a helpline number yourself.
    """
    return get_helplines(country=country)


if __name__ == "__main__":
    for country in ["India", "United States", "United Kingdom", "Australia", "Atlantis"]:
        print(f"\n{country}:")
        print(get_helplines(country))
