"""
nimh_scraper.py
---------------
Scrapes NIMH (National Institute of Mental Health) for clinical
condition information.

TWO TYPES OF PAGES:
1. Topic pages   → nimh.nih.gov/health/topics/depression
                   High-level overview, short, fewer details
                   (This is what you already scraped — thin content)

2. Publication pages → nimh.nih.gov/health/publications/depression
                       Full brochures with symptoms, causes, risk factors,
                       treatment details — MUCH richer content for RAG

We scrape BOTH and merge them.

Output: data/scraped/nimh.csv
Columns: source, condition_tag, section, content, url, collection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.base_scraper import get_driver, get_page_soup, clean_text, logger
import pandas as pd
import time

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_URL    = "https://www.nimh.nih.gov"
OUTPUT_FILE = "data/scraped/nimh.csv"

# Sections to SKIP — these are navigation/meta content, useless for RAG
# They say things like "brochures available" or "clinical trials exist"
# but contain zero actual mental health knowledge
NOISE_SECTIONS = {
    "free health information",
    "science updates",
    "explore clinical trials",
    "additional federal resources",
    "find help and support",          # too generic / just links
}

# ── Page 1: Topic overview pages ───────────────────────────────────────────────
# These are landing pages — short summaries per condition
TOPIC_URLS = {
    "depression":     "/health/topics/depression",
    "anxiety":        "/health/topics/anxiety-disorders",
    "ptsd":           "/health/topics/post-traumatic-stress-disorder-ptsd",
    "bipolar":        "/health/topics/bipolar-disorder",
    "ocd":            "/health/topics/obsessive-compulsive-disorder-ocd",
    "schizophrenia":  "/health/topics/schizophrenia",
    "eating_disorder":"/health/topics/eating-disorders",
    "adhd":           "/health/topics/attention-deficit-hyperactivity-disorder-adhd",
}

# ── Page 2: Publication/brochure pages ─────────────────────────────────────────
# These are the FULL brochures — symptoms, causes, risk factors, treatment
# This is where the real clinical content lives
PUBLICATION_URLS = {
    "depression":     "/health/publications/depression",
    "anxiety":        "/health/publications/anxiety-disorders",
    "ptsd":           "/health/publications/post-traumatic-stress-disorder-ptsd",
    "bipolar":        "/health/publications/bipolar-disorder",
    "ocd":            "nimh.nih.gov/health/publications/obsessive-compulsive-disorder-ocd-when-unwanted-thoughts-take-over",
    "schizophrenia":  "/health/publications/schizophrenia",
    "eating_disorder":"/health/publications/eating-disorders",
    "adhd":           "nimh.nih.gov/health/publications/attention-deficit-hyperactivity-disorder-adhd-the-basics",
}


# ── Core extraction function ───────────────────────────────────────────────────
def extract_sections(soup, condition, url, page_type):
    """
    Walks through the page HTML, groups paragraphs under their headings,
    and returns a list of record dicts.

    Logic:
    - When we hit a heading (h2/h3/h4) → save buffered text as one record
    - When we hit a paragraph/list item → add to buffer
    - At end of page → save final buffer

    This gives us one record per section (heading + its content),
    which is the right chunk size for RAG.
    """
    records = []

    # Try to find the main content area
    # Different pages use different wrapper tags
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find("div", class_=lambda c: c and "content" in c.lower()) or
        soup.find("body")
    )

    if not main:
        logger.warning(f" No content area found for {condition} ({page_type})")
        return records

    current_section = "overview"
    buffer          = []

    for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):

        if tag.name in ["h1", "h2", "h3", "h4"]:
            # ── Save what we've buffered so far ──────────────────────────────
            if buffer:
                content = clean_text(" ".join(buffer))
                section_lower = current_section.lower().strip()

                # Skip noise sections
                is_noise = any(noise in section_lower for noise in NOISE_SECTIONS)

                if not is_noise and len(content) > 150:
                    records.append({
                        "source":        "NIMH",
                        "condition_tag": condition,
                        "section":       current_section,
                        "content":       content,
                        "url":           url,
                        "collection":    "clinical_guidelines",
                        "page_type":     page_type   # topic or publication
                    })

            current_section = clean_text(tag.get_text())
            buffer          = []

        elif tag.name in ["p", "li"]:
            text = clean_text(tag.get_text())
            if len(text) > 40:   # skip trivially short lines
                buffer.append(text)

    # ── Save final buffer ────────────────────────────────────────────────────
    if buffer:
        content       = clean_text(" ".join(buffer))
        section_lower = current_section.lower().strip()
        is_noise      = any(noise in section_lower for noise in NOISE_SECTIONS)

        if not is_noise and len(content) > 150:
            records.append({
                "source":        "NIMH",
                "condition_tag": condition,
                "section":       current_section,
                "content":       content,
                "url":           url,
                "collection":    "clinical_guidelines",
                "page_type":     page_type
            })

    return records


# ── Main scraper ───────────────────────────────────────────────────────────────
def scrape_nimh():
    logger.info("=" * 55)
    logger.info("  NIMH Scraper Starting")
    logger.info("=" * 55)

    driver  = get_driver()
    records = []

    # ── PASS 1: Topic pages ──────────────────────────────────────────────────
    logger.info("\n[Pass 1] Scraping topic overview pages...")

    for condition, path in TOPIC_URLS.items():
        url  = BASE_URL + path
        soup = get_page_soup(driver, url, wait_selector="main", sleep=2)

        if not soup:
            logger.warning(f"  Skipping {condition} topic page — failed to load")
            continue

        page_records = extract_sections(soup, condition, url, "topic")
        records.extend(page_records)
        logger.info(f" {condition} (topic): {len(page_records)} sections")
        time.sleep(3)   # polite delay — don't hammer the server

    # ── PASS 2: Publication/brochure pages ──────────────────────────────────
    logger.info("\n[Pass 2] Scraping publication/brochure pages (rich content)...")

    for condition, path in PUBLICATION_URLS.items():
        url  = BASE_URL + path
        soup = get_page_soup(driver, url, wait_selector="main", sleep=2)

        if not soup:
            logger.warning(f"  Skipping {condition} publication — failed to load")
            continue

        page_records = extract_sections(soup, condition, url, "publication")
        records.extend(page_records)
        logger.info(f" {condition} (publication): {len(page_records)} sections")
        time.sleep(3)

    driver.quit()

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs("data/scraped", exist_ok=True)
    df = pd.DataFrame(records)
    df.drop_duplicates(subset=["content"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    logger.info(f"\n NIMH complete: {len(df)} records → {OUTPUT_FILE}")
    logger.info(f"   Conditions covered: {df['condition_tag'].nunique()}")
    logger.info(f"   Topic records:       {len(df[df['page_type']=='topic'])}")
    logger.info(f"   Publication records: {len(df[df['page_type']=='publication'])}")

    return df


if __name__ == "__main__":
    scrape_nimh()
