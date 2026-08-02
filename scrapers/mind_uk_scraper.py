"""
mind_uk_scraper.py
------------------
Scrapes Mind UK (mind.org.uk) for coping strategies, therapy info,
self-help guides, and condition explanations.

Mind UK is the BEST source for coping content — written by mental
health professionals but in plain, human language. This is what the
LLM will draw on most for mild-moderate severity responses.

Output: data/scraped/mind_uk.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.base_scraper import get_driver, get_page_soup, clean_text, logger
import pandas as pd
import time

BASE_URL    = "https://www.mind.org.uk"
OUTPUT_FILE = "data/scraped/mind_uk.csv"

# Each entry: (path, condition_tag, resource_type)
# resource_type helps ChromaDB route queries correctly
TARGET_PAGES = [
    # Conditions
    ("/information-support/types-of-mental-health-problems/depression/about-depression",           "depression",    "condition_info"),
    ("/information-support/types-of-mental-health-problems/depression/symptoms-of-depression",     "depression",    "symptoms"),
    ("/information-support/types-of-mental-health-problems/depression/self-care-for-depression",   "depression",    "coping"),
    ("/information-support/types-of-mental-health-problems/depression/treatment-for-depression",   "depression",    "treatment"),

    ("/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/about-anxiety",       "anxiety", "condition_info"),
    ("/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/symptoms",            "anxiety", "symptoms"),
    ("/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/self-care",           "anxiety", "coping"),
    ("/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/treatment",           "anxiety", "treatment"),

    ("/information-support/types-of-mental-health-problems/post-traumatic-stress-disorder-ptsd/about-ptsd","ptsd",    "condition_info"),
    ("/information-support/types-of-mental-health-problems/post-traumatic-stress-disorder-ptsd/symptoms",  "ptsd",    "symptoms"),
    ("/information-support/types-of-mental-health-problems/post-traumatic-stress-disorder-ptsd/treatment", "ptsd",    "treatment"),

    ("/information-support/types-of-mental-health-problems/bipolar-disorder/about-bipolar-disorder",       "bipolar", "condition_info"),
    ("/information-support/types-of-mental-health-problems/bipolar-disorder/symptoms",                     "bipolar", "symptoms"),
    ("/information-support/types-of-mental-health-problems/bipolar-disorder/treatment",                    "bipolar", "treatment"),

    ("/information-support/types-of-mental-health-problems/obsessive-compulsive-disorder-ocd/about-ocd",   "ocd",     "condition_info"),
    ("/information-support/types-of-mental-health-problems/obsessive-compulsive-disorder-ocd/treatment",   "ocd",     "treatment"),

    ("/information-support/types-of-mental-health-problems/stress/what-is-stress",                        "stress",  "condition_info"),
    ("/information-support/types-of-mental-health-problems/stress/signs-and-symptoms-of-stress",          "stress",  "symptoms"),
    ("/information-support/types-of-mental-health-problems/stress/what-you-can-do-to-manage-stress",      "stress",  "coping"),

    ("/information-support/types-of-mental-health-problems/loneliness/about-loneliness",                  "loneliness", "condition_info"),
    ("/information-support/types-of-mental-health-problems/loneliness/how-to-feel-less-lonely",           "loneliness", "coping"),

    ("/information-support/types-of-mental-health-problems/grief",                                        "grief",   "condition_info"),

    # Self-care & coping
    ("/information-support/tips-for-everyday-living/how-to-cope-with-sleep-problems",                     "sleep",   "coping"),
    ("/information-support/tips-for-everyday-living/wellbeing",                                           "general", "coping"),
    ("/information-support/tips-for-everyday-living/how-to-cope-with-suicidal-feelings",                  "crisis",  "crisis"),

    # Therapy & treatment
    ("/information-support/drugs-and-treatments/talking-therapies-and-counselling/types-of-therapy",      "therapy", "treatment"),
    ("/information-support/drugs-and-treatments/cognitive-behavioural-therapy-cbt",                       "therapy", "treatment"),
    ("/information-support/drugs-and-treatments/antidepressants",                                         "medication", "treatment"),
    ("/information-support/guides-to-support-and-services/crisis-services",                               "crisis",  "crisis"),
]

# Sections to skip — navigational noise
NOISE_SECTIONS = {
    "share this information",
    "print this page",
    "was this information helpful",
    "get our newsletters",
    "related topics",
    "next steps",
    "useful contacts",   # sometimes just a list of links with no content
}


def scrape_mind_uk():
    logger.info("=" * 55)
    logger.info("  Mind UK Scraper Starting")
    logger.info("=" * 55)

    driver  = get_driver()
    records = []

    for path, condition_tag, resource_type in TARGET_PAGES:
        url  = BASE_URL + path
        soup = get_page_soup(driver, url, wait_selector="main", sleep=2)

        if not soup:
            logger.warning(f"  Skipping {path}")
            continue

        # Mind UK uses <main> or <article> for content
        main = soup.find("main") or soup.find("article")
        if not main:
            logger.warning(f"  No main/article tag found: {path}")
            continue

        current_section = "overview"
        buffer          = []

        for tag in main.find_all(["h2", "h3", "h4", "p", "li"]):

            if tag.name in ["h2", "h3", "h4"]:
                if buffer:
                    content       = clean_text(" ".join(buffer))
                    section_lower = current_section.lower().strip()
                    is_noise      = any(n in section_lower for n in NOISE_SECTIONS)

                    if not is_noise and len(content) > 100:
                        records.append({
                            "source":        "Mind UK",
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

        # Save final buffer
        if buffer:
            content       = clean_text(" ".join(buffer))
            section_lower = current_section.lower().strip()
            is_noise      = any(n in section_lower for n in NOISE_SECTIONS)
            if not is_noise and len(content) > 100:
                records.append({
                    "source":        "Mind UK",
                    "condition_tag": condition_tag,
                    "section":       current_section,
                    "resource_type": resource_type,
                    "content":       content,
                    "url":           url,
                    "collection":    "coping_strategies"
                })

        logger.info(f"{condition_tag} ({resource_type}): scraped")
        time.sleep(3)

    driver.quit()

    os.makedirs("data/scraped", exist_ok=True)
    df = pd.DataFrame(records)
    df.drop_duplicates(subset=["content"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    logger.info(f"\n Mind UK complete: {len(df)} records {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    scrape_mind_uk()
