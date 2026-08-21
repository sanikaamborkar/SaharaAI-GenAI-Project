"""
clean_scraped_data.py  -- FAST VERSION
Removes the slow row-by-row loop. Runs in seconds.
"""

import pandas as pd
import os
import re

INPUT_DIR   = "data/scraped"
OUTPUT_FILE = "data/scraped/master_rag_documents.csv"
MIN_WORDS   = 30
MAX_WORDS   = 400

MIN_WORDS_PER_SOURCE = {
    "findahelpline": 8,
    "india":         10,
}

SOURCE_FILES = {
    "nimh":          "nimh.csv",
    "who":           "who.csv",
    "medlineplus":   "medlineplus.csv",
    "mind_uk":       "mind_uk.csv",
    "helpguide":     "helpguide.csv",
    "findahelpline": "findahelpline.csv",
    "india":         "india_resources.csv",
}

NOISE_KEYWORDS = [
    "clinical trials", "for more information", "reprints",
    "sorry", "page you", "not available", "isn't available",
    "free health information", "science updates", "additional federal",
    "share this", "print this", "was this helpful", "newsletter",
    "subscribe", "cookie", "privacy policy", "terms of use",
    "donate", "become a member", "follow us", "social media",
    "related articles", "references", "about the author",
    "last updated", "find help and support",
]

STANDARD_COLS = [
    "source", "condition_tag", "section",
    "resource_type", "content", "url", "collection",
]


def word_count_series(series):
    """Fast vectorized word count on entire column at once."""
    return series.astype(str).str.split().str.len()


def clean_content_series(series):
    """Fast vectorized text cleaning on entire column at once."""
    s = series.astype(str)
    s = s.str.replace("&amp;",  "&",  regex=False)
    s = s.str.replace("&nbsp;", " ",  regex=False)
    s = s.str.replace("&lt;",   "<",  regex=False)
    s = s.str.replace("&gt;",   ">",  regex=False)
    s = s.str.replace("&quot;", '"',  regex=False)
    s = s.str.replace("&#39;",  "'",  regex=False)
    s = s.str.replace("\xa0",   " ",  regex=False)
    s = s.str.replace(r'\s+',   " ",  regex=True)
    return s.str.strip()


def is_noise_mask(section_series):
    """Fast vectorized noise detection on section column."""
    s = section_series.astype(str).str.lower()
    mask = pd.Series(False, index=section_series.index)
    for kw in NOISE_KEYWORDS:
        mask = mask | s.str.contains(kw, regex=False, na=False)
    return mask


def standardize(df):
    for col in STANDARD_COLS:
        if col not in df.columns:
            df[col] = ""
    extras = [c for c in df.columns if c not in STANDARD_COLS]
    return df[STANDARD_COLS + extras].copy()


def build_helpline_content(df):
    """Vectorized helpline content builder."""
    name    = df["section"].astype(str).str.replace("Helpline:", "", regex=False).str.strip()
    country = df["country"].astype(str) if "country" in df.columns else ""
    phone   = df["phone"].astype(str)   if "phone"   in df.columns else ""
    content = df["content"].astype(str)

    result = (
        name.where(name != "nan", "") + " is a mental health helpline. " +
        "Located in " + country.where(country != "nan", "") + ". " +
        "Phone: " + phone.where(phone != "nan", "") + ". " +
        content.where(content != "nan", "")
    )
    return result.str.replace(r'\s+', ' ', regex=True).str.strip()


def truncate_long_rows(df, max_words=MAX_WORDS):
    """
    Instead of splitting long rows (slow), just truncate them.
    For RAG purposes a 400-word truncation is fine —
    we never need more than ~300 words per chunk anyway.
    This makes the function run instantly instead of minutes.
    """
    mask = word_count_series(df["content"]) > max_words
    if mask.sum() == 0:
        return df

    print(f"  Truncating {mask.sum()} long rows to {max_words} words...")

    def truncate(text):
        words = str(text).split()
        return " ".join(words[:max_words]) if len(words) > max_words else text

    df = df.copy()
    df.loc[mask, "content"] = df.loc[mask, "content"].apply(truncate)
    return df


def run():
    print()
    print("=" * 55)
    print("  MindBridge - Data Cleaning Pipeline")
    print("=" * 55)
    print()

    all_dfs = []

    for key, filename in SOURCE_FILES.items():
        filepath = os.path.join(INPUT_DIR, filename)
        min_w    = MIN_WORDS_PER_SOURCE.get(key, MIN_WORDS)

        print(f"[{key.upper()}]")

        if not os.path.exists(filepath):
            print(f"  SKIP - not found: {filepath}")
            print()
            continue

        if os.path.getsize(filepath) < 10:
            print(f"  SKIP - empty file")
            print()
            continue

        df = pd.read_csv(filepath, encoding="utf-8")
        print(f"  Loaded: {len(df)} rows")

        if len(df) == 0:
            print(f"  SKIP - 0 rows")
            print()
            continue

        # Rebuild helpline content
        if key == "findahelpline":
            df["content"] = build_helpline_content(df)

        # Standardize
        df = standardize(df)

        # Clean content -- vectorized, instant
        df["content"] = clean_content_series(df["content"])

        # Remove noise sections -- vectorized
        before = len(df)
        df = df[~is_noise_mask(df["section"])]
        print(f"  Removed {before - len(df)} noise sections")

        # Remove short rows -- vectorized
        before = len(df)
        df = df[word_count_series(df["content"]) >= min_w]
        print(f"  Removed {before - len(df)} short rows (< {min_w} words)")

        # Deduplicate
        before = len(df)
        df.drop_duplicates(subset=["content"], inplace=True)
        print(f"  Removed {before - len(df)} duplicates")
        print(f"  Clean rows: {len(df)}")

        if len(df) == 0:
            print(f"  SKIP - 0 rows after cleaning")
            print()
            continue

        all_dfs.append(df)
        print()

    if not all_dfs:
        print("ERROR - No data found. Run scrapers first.")
        return

    # Merge
    print("[MERGING]")
    master = pd.concat(all_dfs, ignore_index=True)
    print(f"  Combined: {len(master)} rows")

    before = len(master)
    master.drop_duplicates(subset=["content"], inplace=True)
    print(f"  After global dedup: {len(master)} rows (removed {before - len(master)})")

    # Truncate long rows -- instant (no loop)
    master = truncate_long_rows(master, MAX_WORDS)

    # Add doc_id
    master.reset_index(drop=True, inplace=True)
    master["doc_id"]     = ["doc_" + str(i).zfill(4) for i in range(len(master))]
    master["chunk_num"]  = 0

    # Save
    os.makedirs(INPUT_DIR, exist_ok=True)
    master.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # Summary
    print()
    print("=" * 55)
    print("  CLEANING COMPLETE")
    print("=" * 55)
    print()

    print(f"  {'Source':<22} {'Rows':>6}")
    print(f"  {'-'*30}")
    for source in master["source"].unique():
        count = len(master[master["source"] == source])
        print(f"  {source:<22} {count:>6}")
    print(f"  {'-'*30}")
    print(f"  {'TOTAL':<22} {len(master):>6}")

    print()
    print(f"  {'Collection':<30} {'Rows':>6}")
    print(f"  {'-'*38}")
    for col in master["collection"].unique():
        count = len(master[master["collection"] == col])
        print(f"  {col:<30} {count:>6}")

    print()
    print(f"  {'Condition':<22} {'Rows':>6}")
    print(f"  {'-'*30}")
    for cond, count in master["condition_tag"].value_counts().items():
        print(f"  {cond:<22} {count:>6}")

    wc = word_count_series(master["content"])
    print()
    print(f"  Word count  min:{wc.min()}  max:{wc.max()}  mean:{wc.mean():.0f}  median:{wc.median():.0f}")
    print()
    print(f"  Saved: {OUTPUT_FILE}")
    print(f"  Ready for ChromaDB.")
    print()

    return master


if __name__ == "__main__":
    run()
