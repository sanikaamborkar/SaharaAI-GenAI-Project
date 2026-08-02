"""
helpguide_scraper.py
--------------------
Scrapes HelpGuide.org for in-depth self-help and coping articles.

HelpGuide articles are long-form, practical, and written by
mental health professionals. They cover real-life coping strategies
(not just definitions) — perfect for mild-moderate severity RAG retrieval.

Output: data/scraped/helpguide.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.base_scraper import get_driver, get_page_soup, clean_text, logger
import pandas as pd
import time

BASE_URL    = "https://www.helpguide.org"
OUTPUT_FILE = "data/scraped/helpguide.csv"

# (path, condition_tag, resource_type)
TARGET_PAGES = [
    # Depression
    ("/mental-health/depression/depression-symptoms-and-warning-signs", "depression", "symptoms"),
    ("/mental-health/depression/coping-with-depression",                "depression", "coping"),
    ("/mental-health/depression/dealing-with-depression",               "depression", "coping"),

    # Anxiety
    ("/mental-health/anxiety/anxiety-disorders-and-anxiety-attacks",    "anxiety", "condition_info"),
    ("/mental-health/anxiety/how-to-stop-worrying",                     "anxiety", "coping"),
    ("/mental-health/anxiety/relaxation-techniques-for-stress-relief",  "anxiety", "coping"),

    # PTSD
    ("/mental-health/ptsd-trauma/ptsd-symptoms-self-help-treatment",    "ptsd", "coping"),
    ("/mental-health/ptsd-trauma/emotional-and-psychological-trauma",   "ptsd", "condition_info"),

    # Stress & Burnout
    ("/mental-health/stress/stress-management",                         "stress",   "coping"),
    ("/mental-health/stress/burnout-prevention-and-recovery",           "burnout",  "coping"),

    # Sleep
    ("/mental-health/sleep/how-to-sleep-better",                        "sleep",    "coping"),
    ("/mental-health/sleep/insomnia",                                   "sleep",    "condition_info"),

    # Grief & Loss
    ("/mental-health/grief-loss/coping-with-grief-and-loss",           "grief",    "coping"),

    # Loneliness
    ("/mental-health/social-life/loneliness-isolation",                 "loneliness", "coping"),

    # Therapy & finding help
    ("/mental-health/treatment/therapy-and-counseling",                 "therapy",  "treatment"),
    ("/mental-health/treatment/finding-a-therapist",                    "therapy",  "treatment"),

    # Self-harm & crisis
    ("/mental-health/suicide-prevention/suicide-prevention",            "crisis",   "crisis"),
]

NOISE_SECTIONS = {
    "related articles",
    "more information",
    "references",
    "authors",
    "last updated",
    "helpguide uses cookies",
    "about the author",
    "print",
}


def scrape_helpguide():
    logger.info("=" * 55)
    logger.info("  HelpGuide Scraper Starting")
    logger.info("=" * 55)

    driver  = get_driver()
    records = []

    for path, condition_tag, resource_type in TARGET_PAGES:
        url  = BASE_URL + path
        # HelpGuide content is inside <article> tag
        soup = get_page_soup(driver, url, wait_selector="article", sleep=2)

        if not soup:
            logger.warning(f"  Skipping {path}")
            continue

        article = soup.find("article") or soup.find("main")

        print("Article found:", article is not None)

        if article:
            print("Paragraphs:", len(article.find_all("p")))
            print("Headings:", len(article.find_all(["h2", "h3", "h4"])))
        if not article:
            logger.warning(f"  No article/main tag: {path}")
            continue

        current_section = "overview"
        buffer          = []

        for tag in article.find_all(["h2", "h3", "h4", "p", "li"]):

            if tag.name in ["h2", "h3", "h4"]:
                if buffer:
                    content       = clean_text(" ".join(buffer))
                    section_lower = current_section.lower().strip()
                    is_noise      = any(n in section_lower for n in NOISE_SECTIONS)

                    if not is_noise and len(content) > 100:
                        records.append({
                            "source":        "HelpGuide",
                            "condition_tag": condition_tag,
                            "section":       current_section,
                            "resource_type": resource_type,
                            "content":       content,
                            "url":           url,
                            "collection":    "coping_strategies"
                        })

                current_section = clean_text(tag.get_text())
                buffer          = []

            elif tag.name in ["p", "li"]:
                text = clean_text(tag.get_text())
                if len(text) > 40:
                    buffer.append(text)

        if buffer:
            content       = clean_text(" ".join(buffer))
            section_lower = current_section.lower().strip()
            is_noise      = any(n in section_lower for n in NOISE_SECTIONS)
            if not is_noise and len(content) > 100:
                records.append({
                    "source":        "HelpGuide",
                    "condition_tag": condition_tag,
                    "section":       current_section,
                    "resource_type": resource_type,
                    "content":       content,
                    "url":           url,
                    "collection":    "coping_strategies"
                })

        logger.info(f"  ✅ {condition_tag} ({resource_type}): scraped")
        time.sleep(3)

    driver.quit()

    print("Records collected:", len(records))

    os.makedirs("data/scraped", exist_ok=True)
    df = pd.DataFrame(records)
    df.drop_duplicates(subset=["content"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    logger.info(f"\n✅ HelpGuide complete: {len(df)} records → {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    scrape_helpguide()
