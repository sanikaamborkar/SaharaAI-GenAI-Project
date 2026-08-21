"""
scrape_all_sites.py  ── FINAL VERSION
---------------------------------------
Scrapes ALL sources in one file. No external dependencies.
NIMH is now included here directly.

Sources:
  1. NIMH          (clinical guidelines)
  2. WHO           (global mental health facts)
  3. Mind UK       (coping strategies)
  4. HelpGuide     (self-help articles)
  5. NAMI          (patient-friendly guides)
  6. Findahelpline (global crisis helplines)
  7. MedlinePlus   (US NLM — easy to scrape, no bot block)
  8. Wikipedia MH  (India mental health + global overview)
  9. Snehi India   (Indian helpline — simpler than iCall)

Usage:
    python scrape_all_sites.py          # run everything
    python scrape_all_sites.py nimh     # run only NIMH
    python scrape_all_sites.py who      # run only WHO
    python scrape_all_sites.py minduk
    python scrape_all_sites.py helpguide
    python scrape_all_sites.py nami
    python scrape_all_sites.py helpline
    python scrape_all_sites.py india
    python scrape_all_sites.py medline

Run from your project root:
    cd mindbridge_scraping
    python scrape_all_sites.py
"""

import sys
import os
import re
import time
import logging
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# ── Directories ────────────────────────────────────────────────────────────────
os.makedirs("logs",         exist_ok=True)
os.makedirs("data/scraped", exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/scraping.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DRIVER
# ══════════════════════════════════════════════════════════════════════════════
def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-gpu")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2
    })

    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    driver.set_page_load_timeout(60)
    return driver


# ══════════════════════════════════════════════════════════════════════════════
# PAGE LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_page(driver, url, sleep=4):
    try:
        driver.get(url)
        time.sleep(sleep)

        # Dismiss cookie banners
        for btn_text in ["accept all", "accept cookies", "accept", "agree", "ok", "got it", "close"]:
            try:
                btns = driver.find_elements(By.XPATH,
                    f"//button[contains(translate(text(),"
                    f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'{btn_text}')]"
                )
                if btns:
                    btns[0].click()
                    time.sleep(1)
                    break
            except Exception:
                pass

        title = driver.title
        logger.info(f"  Page: '{title[:70]}'")

        blocked = ["just a moment", "access denied", "403 forbidden",
                   "captcha", "are you human", "cloudflare", "checking your browser"]
        if any(b in title.lower() for b in blocked):
            logger.warning(f"  [BLOCKED] Bot block detected — skipping")
            return None

        return BeautifulSoup(driver.page_source, "html.parser")

    except TimeoutException:
        logger.error(f"  [TIMEOUT] {url}")
        return None
    except WebDriverException as e:
        logger.error(f"  [ERROR] WebDriver: {str(e)[:100]}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
NOISE_KEYWORDS = [
    "share this", "print this", "was this helpful", "newsletter",
    "subscribe", "cookie", "privacy policy", "terms", "donate",
    "follow us", "social media", "related articles", "references",
    "about the author", "last updated", "explore more", "find support",
    "clinical trials", "reprints", "for more information",
    "free health information", "science updates", "additional federal"
]

def clean(text):
    return " ".join(str(text).split()).strip()

def extract_content(soup, source, condition_tag, resource_type, url, collection, min_words=30):
    records = []

    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find("div", {"id": "content"}) or
        soup.find("div", {"id": "main"}) or
        soup.find("div", class_=lambda c: c and "content" in
                  " ".join(c if isinstance(c, list) else [c]).lower()) or
        soup.find("body")
    )

    if not main:
        return records

    current_section = "overview"
    buffer          = []

    def save_buffer():
        if not buffer:
            return
        content = clean(" ".join(buffer))
        s_lower = current_section.lower()
        is_noise = any(kw in s_lower for kw in NOISE_KEYWORDS)
        if not is_noise and len(content.split()) >= min_words:
            records.append({
                "source":        source,
                "condition_tag": condition_tag,
                "section":       current_section,
                "resource_type": resource_type,
                "content":       content,
                "url":           url,
                "collection":    collection
            })

    for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        if tag.name in ["h1", "h2", "h3", "h4"]:
            save_buffer()
            current_section = clean(tag.get_text())
            buffer = []
        elif tag.name in ["p", "li"]:
            text = clean(tag.get_text())
            if len(text.split()) >= 8:
                buffer.append(text)

    save_buffer()
    return records


def save_csv(records, output_file):
    df = pd.DataFrame(records) if records else pd.DataFrame()
    if len(df) > 0:
        df.drop_duplicates(subset=["content"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. NIMH  (added back — was broken after base_scraper.py was replaced)
# ══════════════════════════════════════════════════════════════════════════════
NIMH_TOPIC_URLS = {
    "depression":      "https://www.nimh.nih.gov/health/topics/depression",
    "anxiety":         "https://www.nimh.nih.gov/health/topics/anxiety-disorders",
    "ptsd":            "https://www.nimh.nih.gov/health/topics/post-traumatic-stress-disorder-ptsd",
    "bipolar":         "https://www.nimh.nih.gov/health/topics/bipolar-disorder",
    "ocd":             "https://www.nimh.nih.gov/health/topics/obsessive-compulsive-disorder-ocd",
    "schizophrenia":   "https://www.nimh.nih.gov/health/topics/schizophrenia",
    "eating_disorder": "https://www.nimh.nih.gov/health/topics/eating-disorders",
    "adhd":            "https://www.nimh.nih.gov/health/topics/attention-deficit-hyperactivity-disorder-adhd",
}

NIMH_PUB_URLS = {
    "depression":      "https://www.nimh.nih.gov/health/publications/depression",
    "anxiety":         "https://www.nimh.nih.gov/health/publications/anxiety-disorders",
    "ptsd":            "https://www.nimh.nih.gov/health/publications/post-traumatic-stress-disorder-ptsd",
    "bipolar":         "https://www.nimh.nih.gov/health/publications/bipolar-disorder",
    "ocd":             "https://www.nimh.nih.gov/health/publications/obsessive-compulsive-disorder-ocd-when-unwanted-thoughts-take-over",
    "schizophrenia":   "https://www.nimh.nih.gov/health/publications/schizophrenia",
    "eating_disorder": "https://www.nimh.nih.gov/health/publications/eating-disorders",
    "adhd":            "https://www.nimh.nih.gov/health/publications/attention-deficit-hyperactivity-disorder-adhd-the-basics",
}

def scrape_nimh(driver):
    logger.info("\n[NIMH] Starting...")
    records = []

    for condition, url in {**NIMH_TOPIC_URLS, **NIMH_PUB_URLS}.items():
        # Clean condition tag (remove _pub suffix if any)
        cond = condition.replace("_pub", "")
        soup = load_page(driver, url, sleep=3)
        if soup is None:
            time.sleep(3)
            continue
        recs = extract_content(soup, "NIMH", cond, "condition_info", url, "clinical_guidelines", min_words=40)
        records.extend(recs)
        logger.info(f"  [OK] {cond}: {len(recs)} chunks")
        time.sleep(3)

    df = save_csv(records, "data/scraped/nimh.csv")
    logger.info(f"[NIMH] Done: {len(df)} records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. WHO  (global mental health — was skipped before, now added)
# WHY: WHO has the most authoritative global mental health content.
# It's a government/UN site — no bot protection, static HTML, easy to scrape.
# Adds international treatment guidelines NIMH doesn't have.
# ══════════════════════════════════════════════════════════════════════════════
WHO_PAGES = [
    ("https://www.who.int/news-room/fact-sheets/detail/depression",                    "depression",    "condition_info"),
    ("https://www.who.int/news-room/fact-sheets/detail/anxiety-disorders",             "anxiety",       "condition_info"),
    ("https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response", "general", "condition_info"),
    ("https://www.who.int/news-room/fact-sheets/detail/suicide",                       "crisis",        "crisis"),
    ("https://www.who.int/news-room/fact-sheets/detail/schizophrenia",                 "schizophrenia", "condition_info"),
    ("https://www.who.int/news-room/fact-sheets/detail/bipolar-disorder",              "bipolar",       "condition_info"),
    ("https://www.who.int/news-room/fact-sheets/detail/post-traumatic-stress-disorder-(ptsd)", "ptsd",  "condition_info"),
    ("https://www.who.int/news-room/fact-sheets/detail/mental-disorders",              "general",       "condition_info"),
]

def scrape_who(driver):
    logger.info("\n[WHO] Starting...")
    records = []

    for url, condition_tag, resource_type in WHO_PAGES:
        soup = load_page(driver, url, sleep=4)
        if soup is None:
            time.sleep(3)
            continue
        recs = extract_content(soup, "WHO", condition_tag, resource_type, url, "clinical_guidelines")
        records.extend(recs)
        logger.info(f"  [OK] {condition_tag}: {len(recs)} chunks")
        time.sleep(4)

    df = save_csv(records, "data/scraped/who.csv")
    logger.info(f"[WHO] Done: {len(df)} records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. MedlinePlus  (US National Library of Medicine)
# WHY: Pure static HTML, zero bot protection, government site.
# Great fallback for conditions where other scrapers failed (OCD, ADHD).
# ══════════════════════════════════════════════════════════════════════════════
MEDLINE_PAGES = [
    ("https://medlineplus.gov/depression.html",                  "depression",    "condition_info"),
    ("https://medlineplus.gov/anxiety.html",                     "anxiety",       "condition_info"),
    ("https://medlineplus.gov/posttraumaticstressdisorder.html", "ptsd",          "condition_info"),
    ("https://medlineplus.gov/bipolardisorder.html",             "bipolar",       "condition_info"),
    ("https://medlineplus.gov/obsessivecompulsivedisorder.html", "ocd",           "condition_info"),
    ("https://medlineplus.gov/attentiondeficithyperactivitydisorder.html", "adhd","condition_info"),
    ("https://medlineplus.gov/schizophrenia.html",               "schizophrenia", "condition_info"),
    ("https://medlineplus.gov/eatingdisorders.html",             "eating_disorder","condition_info"),
    ("https://medlineplus.gov/stress.html",                      "stress",        "condition_info"),
    ("https://medlineplus.gov/suicide.html",                     "crisis",        "crisis"),
]

def scrape_medline(driver):
    logger.info("\n[MedlinePlus] Starting...")
    records = []

    for url, condition_tag, resource_type in MEDLINE_PAGES:
        soup = load_page(driver, url, sleep=3)
        if soup is None:
            time.sleep(3)
            continue
        recs = extract_content(soup, "MedlinePlus", condition_tag, resource_type, url, "clinical_guidelines")
        records.extend(recs)
        logger.info(f"  [OK] {condition_tag}: {len(recs)} chunks")
        time.sleep(3)

    df = save_csv(records, "data/scraped/medlineplus.csv")
    logger.info(f"[MedlinePlus] Done: {len(df)} records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Mind UK
# ══════════════════════════════════════════════════════════════════════════════
MIND_UK_PAGES = [
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/depression/about-depression",             "depression",  "condition_info"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/depression/symptoms-of-depression",       "depression",  "symptoms"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/depression/self-care-for-depression",     "depression",  "coping"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/depression/treatment-for-depression",     "depression",  "treatment"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/about-anxiety", "anxiety",     "condition_info"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/symptoms",      "anxiety",     "symptoms"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/self-care",     "anxiety",     "coping"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/treatment",     "anxiety",     "treatment"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/post-traumatic-stress-disorder-ptsd/about-ptsd", "ptsd", "condition_info"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/post-traumatic-stress-disorder-ptsd/symptoms",   "ptsd", "symptoms"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/post-traumatic-stress-disorder-ptsd/treatment",  "ptsd", "treatment"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/bipolar-disorder/about-bipolar-disorder", "bipolar",    "condition_info"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/bipolar-disorder/symptoms",               "bipolar",    "symptoms"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/bipolar-disorder/treatment",              "bipolar",    "treatment"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/obsessive-compulsive-disorder-ocd/about-ocd","ocd",    "condition_info"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/obsessive-compulsive-disorder-ocd/treatment","ocd",    "treatment"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/stress/what-is-stress",                   "stress",     "condition_info"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/stress/what-you-can-do-to-manage-stress", "stress",     "coping"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/loneliness/about-loneliness",             "loneliness", "condition_info"),
    ("https://www.mind.org.uk/information-support/types-of-mental-health-problems/loneliness/how-to-feel-less-lonely",      "loneliness", "coping"),
    ("https://www.mind.org.uk/information-support/tips-for-everyday-living/wellbeing",                                      "general",    "coping"),
    ("https://www.mind.org.uk/information-support/drugs-and-treatments/talking-therapies-and-counselling/types-of-therapy", "therapy",    "treatment"),
    ("https://www.mind.org.uk/information-support/drugs-and-treatments/cognitive-behavioural-therapy-cbt",                  "therapy",    "treatment"),
    ("https://www.mind.org.uk/information-support/guides-to-support-and-services/crisis-services",                          "crisis",     "crisis"),
]

def scrape_minduk(driver):
    logger.info("\n[Mind UK] Starting...")
    records = []
    for url, condition_tag, resource_type in MIND_UK_PAGES:
        soup = load_page(driver, url, sleep=5)  # extra sleep for Mind UK
        if soup is None:
            time.sleep(5)
            continue
        recs = extract_content(soup, "Mind UK", condition_tag, resource_type, url, "coping_strategies")
        records.extend(recs)
        logger.info(f"  [OK] {condition_tag} ({resource_type}): {len(recs)} chunks")
        time.sleep(5)
    df = save_csv(records, "data/scraped/mind_uk.csv")
    logger.info(f"[Mind UK] Done: {len(df)} records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. HelpGuide
# ══════════════════════════════════════════════════════════════════════════════
HELPGUIDE_PAGES = [
    ("https://www.helpguide.org/mental-health/depression/depression-symptoms-and-warning-signs", "depression", "symptoms"),
    ("https://www.helpguide.org/mental-health/depression/coping-with-depression",                "depression", "coping"),
    ("https://www.helpguide.org/mental-health/depression/dealing-with-depression",               "depression", "coping"),
    ("https://www.helpguide.org/mental-health/anxiety/anxiety-disorders-and-anxiety-attacks",    "anxiety",    "condition_info"),
    ("https://www.helpguide.org/mental-health/anxiety/how-to-stop-worrying",                     "anxiety",    "coping"),
    ("https://www.helpguide.org/mental-health/anxiety/relaxation-techniques-for-stress-relief",  "anxiety",    "coping"),
    ("https://www.helpguide.org/mental-health/ptsd-trauma/ptsd-symptoms-self-help-treatment",    "ptsd",       "coping"),
    ("https://www.helpguide.org/mental-health/ptsd-trauma/emotional-and-psychological-trauma",   "ptsd",       "condition_info"),
    ("https://www.helpguide.org/mental-health/stress/stress-management",                         "stress",     "coping"),
    ("https://www.helpguide.org/mental-health/stress/burnout-prevention-and-recovery",           "burnout",    "coping"),
    ("https://www.helpguide.org/mental-health/sleep/how-to-sleep-better",                        "sleep",      "coping"),
    ("https://www.helpguide.org/mental-health/grief-loss/coping-with-grief-and-loss",           "grief",      "coping"),
    ("https://www.helpguide.org/mental-health/social-life/loneliness-isolation",                 "loneliness", "coping"),
    ("https://www.helpguide.org/mental-health/treatment/therapy-and-counseling",                 "therapy",    "treatment"),
    ("https://www.helpguide.org/mental-health/suicide-prevention/suicide-prevention",            "crisis",     "crisis"),
]

def scrape_helpguide(driver):
    logger.info("\n[HelpGuide] Starting...")
    records = []
    for url, condition_tag, resource_type in HELPGUIDE_PAGES:
        soup = load_page(driver, url, sleep=4)
        if soup is None:
            time.sleep(5)
            continue
        recs = extract_content(soup, "HelpGuide", condition_tag, resource_type, url, "coping_strategies")
        records.extend(recs)
        logger.info(f"  [OK] {condition_tag} ({resource_type}): {len(recs)} chunks")
        time.sleep(4)
    df = save_csv(records, "data/scraped/helpguide.csv")
    logger.info(f"[HelpGuide] Done: {len(df)} records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. NAMI
# ══════════════════════════════════════════════════════════════════════════════
NAMI_PAGES = [
    ("https://www.nami.org/about-mental-illness/mental-health-conditions/depression",                    "depression",    "condition_info"),
    ("https://www.nami.org/about-mental-illness/mental-health-conditions/anxiety-disorders",             "anxiety",       "condition_info"),
    ("https://www.nami.org/about-mental-illness/mental-health-conditions/posttraumatic-stress-disorder", "ptsd",          "condition_info"),
    ("https://www.nami.org/about-mental-illness/mental-health-conditions/bipolar-disorder",              "bipolar",       "condition_info"),
    ("https://www.nami.org/about-mental-illness/mental-health-conditions/obsessive-compulsive-disorder", "ocd",           "condition_info"),
    ("https://www.nami.org/about-mental-illness/mental-health-conditions/schizophrenia",                 "schizophrenia", "condition_info"),
    ("https://www.nami.org/about-mental-illness/treatments/psychotherapy",                               "therapy",       "treatment"),
    ("https://www.nami.org/about-mental-illness/treatments/mental-health-medications",                   "medication",    "treatment"),
    ("https://www.nami.org/about-mental-illness/common-with-mental-illness/stigma-and-discrimination",   "general",       "stigma"),
    ("https://www.nami.org/support/nami-helpline",                                                       "general",       "helpline_info"),
]

def scrape_nami(driver):
    logger.info("\n[NAMI] Starting...")
    records = []
    for url, condition_tag, resource_type in NAMI_PAGES:
        soup = load_page(driver, url, sleep=4)
        if soup is None:
            time.sleep(5)
            continue
        recs = extract_content(soup, "NAMI", condition_tag, resource_type, url, "patient_friendly")
        records.extend(recs)
        logger.info(f"  [OK] {condition_tag} ({resource_type}): {len(recs)} chunks")
        time.sleep(4)
    df = save_csv(records, "data/scraped/nami.csv")
    logger.info(f"[NAMI] Done: {len(df)} records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7. Findahelpline
# ══════════════════════════════════════════════════════════════════════════════
HELPLINE_COUNTRIES = [
    ("in", "India"), ("us", "United States"), ("gb", "United Kingdom"),
    ("au", "Australia"), ("ca", "Canada"), ("nz", "New Zealand"),
    ("ie", "Ireland"), ("za", "South Africa"), ("sg", "Singapore"),
    ("ph", "Philippines"), ("ng", "Nigeria"), ("pk", "Pakistan"),
    ("bd", "Bangladesh"), ("ke", "Kenya"), ("my", "Malaysia"),
]

def scrape_helplines(driver):
    logger.info("\n[Findahelpline] Starting...")
    records = []

    for code, country in HELPLINE_COUNTRIES:
        url  = f"https://findahelpline.com/countries/{code}"
        soup = load_page(driver, url, sleep=4)
        if soup is None:
            time.sleep(5)
            continue

        cards = (
            soup.find_all("div", class_=lambda c: c and "organization" in
                          " ".join(c if isinstance(c, list) else [c]).lower()) or
            soup.find_all("article") or
            soup.find_all("div", class_=lambda c: c and "card" in
                          " ".join(c if isinstance(c, list) else [c]).lower())
        )

        count = 0
        seen_keys = set()
        for card in cards:
            name_tag = card.find("a", attrs={"data-testid": "headingLink"})
            name     = clean(name_tag.get_text()) if name_tag else ""
            if not name:
                continue

            dedup_key = (name, country)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            tel   = card.find("a", href=lambda h: h and "tel:" in h)
            phone = tel["href"].replace("tel:", "") if tel else ""
            if not phone:
                m = re.search(r'[\d\s\-\+\(\)]{7,20}', card.get_text())
                phone = m.group().strip() if m else ""

            link    = card.find("a", href=lambda h: h and h.startswith("http"))
            website = link["href"] if link else ""

            topic_tags = card.find_all(class_=lambda c: c and "tag" in
                                       " ".join(c if isinstance(c, list) else [c]).lower())
            topics = ", ".join([t.get_text(strip=True) for t in topic_tags]) or "mental health"

            content = f"{name} is a mental health helpline in {country}."
            if phone:   content += f" Phone: {phone}."
            if website: content += f" Website: {website}."
            content += f" Topics covered: {topics}."

            records.append({
                "source":        "findahelpline.com",
                "condition_tag": "crisis",
                "section":       f"Helpline: {name}",
                "resource_type": "helpline",
                "content":       content,
                "url":           website or url,
                "collection":    "helplines_crisis",
                "country":       country,
                "phone":         phone,
                
                })
            count += 1

        logger.info(f"  [OK] {country}: {count} helplines")
        time.sleep(4)

    df = save_csv(records, "data/scraped/findahelpline.csv")
    logger.info(f"[Findahelpline] Done: {len(df)} records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. INDIA ALTERNATIVES
# WHY these instead of iCall/Vandrevala:
# iCall and Vandrevala use heavy JS — content only appears after JS runs,
# but Selenium captures the page before JS finishes → empty scrape.
#
# These alternatives are static or lightly dynamic:
# - Wikipedia India mental health: static HTML, rich content, helplines listed
# - Snehi.org: simpler site than iCall, less JS-heavy
# - MHA India (mentalhealthindia.net): static articles
# ══════════════════════════════════════════════════════════════════════════════
INDIA_PAGES = [
    # Wikipedia — fully static, always works, has Indian helpline numbers
    ("https://en.wikipedia.org/wiki/Mental_health_in_India",
     "Wikipedia", "general", "condition_info"),

    # Snehi — Indian emotional support helpline, simpler than iCall
    ("https://www.snehi.org",
     "Snehi India", "general", "helpline_info"),

    ("https://www.snehi.org/about-us",
     "Snehi India", "general", "helpline_info"),

    # MHA India — Mental Health Association of India
    ("https://www.mentalhealthindia.net",
     "MHA India", "general", "condition_info"),

    # iGotMental — Indian mental health platform, good static content
    ("https://www.igotmental.com/mental-health-india",
     "iGotMental", "general", "condition_info"),

    # The Live Love Laugh Foundation (Deepika Padukone's foundation)
    # — popular in India, good awareness content
    ("https://www.thelivelovelaughfoundation.org/find-help/helplines",
     "Live Love Laugh", "crisis", "helpline_info"),
]

def scrape_india(driver):
    logger.info("\n[India Resources] Starting...")
    records = []

    for url, source, condition_tag, resource_type in INDIA_PAGES:
        soup = load_page(driver, url, sleep=5)
        if soup is None:
            time.sleep(5)
            continue
        recs = extract_content(soup, source, condition_tag, resource_type, url, "india_resources", min_words=25)
        records.extend(recs)
        logger.info(f"  [OK] {source}: {len(recs)} chunks")
        time.sleep(5)

    df = save_csv(records, "data/scraped/india_resources.csv")
    logger.info(f"[India] Done: {len(df)} records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
TASK_MAP = {
    "nimh":      scrape_nimh,
    "who":       scrape_who,
    "medline":   scrape_medline,
    "minduk":    scrape_minduk,
    "helpguide": scrape_helpguide,
    "nami":      scrape_nami,
    "helpline":  scrape_helplines,
    "india":     scrape_india,
}

def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    print(f"\n{'='*55}")
    print(f"  MindBridge Scraper — task: {arg}")
    print(f"{'='*55}\n")

    driver = get_driver()
    results = {}

    try:
        if arg == "all":
            for name, fn in TASK_MAP.items():
                df = fn(driver)
                results[name] = len(df) if df is not None else 0
        elif arg in TASK_MAP:
            df = TASK_MAP[arg](driver)
            results[arg] = len(df) if df is not None else 0
        else:
            print(f"Unknown task: {arg}")
            print(f"Valid options: {list(TASK_MAP.keys())} or 'all'")
    finally:
        driver.quit()

    print(f"\n{'='*55}")
    print("  SCRAPING SUMMARY")
    print(f"{'='*55}")
    for name, count in results.items():
        status = "[OK]" if count > 0 else "[FAIL]"
        print(f"  {status} {name:<15} {count:>4} records")
    total = sum(results.values())
    print(f"  {'TOTAL':<15} {total:>4} records")
    print(f"{'='*55}\n")
    print("Next step: run clean_scraped_data.py to merge all CSVs")


if __name__ == "__main__":
    main()

