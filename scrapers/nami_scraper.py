"""
nami_scraper.py
---------------
Scrapes NAMI (National Alliance on Mental Illness) for patient-friendly
mental health condition guides.

NAMI is written FOR patients and families — plain language, empathetic
tone, stigma-aware. Complements NIMH (which is clinical/technical).

The LLM needs both:
- NIMH for clinical accuracy
- NAMI for empathetic, accessible explanation

Output: data/scraped/nami.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.base_scraper import get_driver, get_page_soup, clean_text, logger
import pandas as pd
import time

BASE_URL    = "https://www.nami.org"
OUTPUT_FILE = "data/scraped/nami.csv"

TARGET_PAGES = [
    # Conditions
    ("/about-mental-illness/mental-health-conditions/depression",                        "depression",    "condition_info"),
    ("/about-mental-illness/mental-health-conditions/anxiety-disorders",                 "anxiety",       "condition_info"),
    ("/about-mental-illness/mental-health-conditions/posttraumatic-stress-disorder",     "ptsd",          "condition_info"),
    ("/about-mental-illness/mental-health-conditions/bipolar-disorder",                  "bipolar",       "condition_info"),
    ("/about-mental-illness/mental-health-conditions/obsessive-compulsive-disorder",     "ocd",           "condition_info"),
    ("/about-mental-illness/mental-health-conditions/schizophrenia",                     "schizophrenia", "condition_info"),
    ("/about-mental-illness/mental-health-conditions/attention-deficit-hyperactivity-disorder", "adhd",   "condition_info"),

    # Treatment
    ("/about-mental-illness/treatments/psychotherapy",                                   "therapy",    "treatment"),
    ("/about-mental-illness/treatments/mental-health-medications",                       "medication", "treatment"),

    # Living with mental illness
    ("/about-mental-illness/common-with-mental-illness/stigma-and-discrimination",       "general", "stigma"),

    # Support
    ("/support/nami-helpline",                                                           "general", "helpline_info"),
]

NOISE_SECTIONS = {
    "nami helpline",         # keep the page but skip the repeated header
    "donate",
    "become a member",
    "find support",          # navigation links only
    "share this page",
    "explore more",
}


def scrape_nami():
    logger.info("=" * 55)
    logger.info("  NAMI Scraper Starting")
    logger.info("=" * 55)

    driver  = get_driver()
    records = []

    for path, condition_tag, resource_type in TARGET_PAGES:
        url  = BASE_URL + path
        soup = get_page_soup(driver, url, wait_selector="main", sleep=2)

        if not soup:
            logger.warning(f"  Skipping {path}")
            continue

        main = soup.find("main") or soup.find("article")
        if not main:
            logger.warning(f"  No main/article tag: {path}")
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
                            "source":        "NAMI",
                            "condition_tag": condition_tag,
                            "section":       current_section,
                            "resource_type": resource_type,
                            "content":       content,
                            "url":           url,
                            "collection":    "patient_friendly"
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
                    "source":        "NAMI",
                    "condition_tag": condition_tag,
                    "section":       current_section,
                    "resource_type": resource_type,
                    "content":       content,
                    "url":           url,
                    "collection":    "patient_friendly"
                })

        logger.info(f"  ✅ {condition_tag} ({resource_type}): scraped")
        time.sleep(3)

    driver.quit()

    os.makedirs("data/scraped", exist_ok=True)
    df = pd.DataFrame(records)
    df.drop_duplicates(subset=["content"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    logger.info(f"\n✅ NAMI complete: {len(df)} records → {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    scrape_nami()
