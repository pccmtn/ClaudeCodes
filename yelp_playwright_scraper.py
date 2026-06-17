"""
Yelp review scraper using Playwright (headless browser) instead of raw
`requests`. A real browser engine renders pages and executes JS the way a
normal visitor's browser would, which is much harder for Cloudflare-style
bot detection to flag than a bare HTTP client — especially when combined
with proxy rotation (see `yelp_scraper.load_proxies()`).

Setup:
    pip install playwright
    playwright install chromium

Usage:
    from yelp_playwright_scraper import scrape_business, scrape_business_list

    df = scrape_business("gary-danko-san-francisco", base_delay=5)

    # With rotating proxies (reuses yelp_scraper.load_proxies())
    from yelp_scraper import load_proxies
    proxies = load_proxies()
    df = scrape_business("gary-danko-san-francisco", proxies=proxies)
"""

import random
import time

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from yelp_scraper import (
    SEL,
    USER_AGENTS,
    _empty_row,
    _is_managerial_node,
    _extract_previous_review_date,
    _rating_label_to_float,
    save_to_csv,
    already_collected,
)
from bs4 import BeautifulSoup
import re


# ---------------------------------------------------------------------------
# Browser-driven page fetch
# ---------------------------------------------------------------------------
def fetch_page_playwright(
    business_id: str,
    pagestart: int,
    proxies: list[str] | None = None,
    max_retries: int = 5,
    base_delay: float = 5.0,
    headless: bool = True,
) -> str | None:
    """
    Load one Yelp review page in a real (headless) browser and return its
    rendered HTML, or None if no review blocks could be found.
    """
    url = f"https://www.yelp.com/biz/{business_id}?start={pagestart}#reviews"
    delay = base_delay

    for attempt in range(1, max_retries + 1):
        proxy_url = random.choice(proxies) if proxies else None
        proxy_config = {"server": proxy_url} if proxy_url else None

        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=headless, proxy=proxy_config)
                context = browser.new_context(user_agent=random.choice(USER_AGENTS))
                page = context.new_page()
                page.goto(url, timeout=30_000, wait_until="domcontentloaded")

                try:
                    page.wait_for_selector(SEL["review_blocks"], timeout=10_000)
                except PlaywrightTimeoutError:
                    html = page.content()
                    browser.close()
                    if "pending verification" in html.lower() or "captcha" in html.lower():
                        print(f"[Blocked] attempt {attempt}/{max_retries} "
                              f"(proxy={proxy_url}), retrying in {delay:.1f}s…")
                    else:
                        print(f"[Empty] No review blocks found — reached end of pages "
                              f"(start={pagestart})")
                        return None
                else:
                    html = page.content()
                    browser.close()
                    print(f"[OK] page start={pagestart} (proxy={proxy_url})")
                    return html

        except Exception as exc:
            print(f"[Attempt {attempt}/{max_retries}] Playwright error: {exc} "
                  f"(proxy={proxy_url}), retrying in {delay:.1f}s…")

        jitter = random.uniform(0, 5)
        time.sleep(delay + jitter)
        delay = min(delay * 1.5, 120)

    print(f"[Failure] Gave up after {max_retries} attempts for {business_id} start={pagestart}.")
    return None


# ---------------------------------------------------------------------------
# Parsing (mirrors yelp_scraper.parse_reviews, operating on rendered HTML)
# ---------------------------------------------------------------------------
def _parse_single_review(frame, business_id: str, page_number: int, count: int) -> dict:
    username = frame.select(SEL["username"])[0].string
    rating_label = frame.select(SEL["rating"])[0].attrs["aria-label"]
    rating = _rating_label_to_float(rating_label)
    date = frame.select(SEL["date"])[0].string
    review_text = frame.select(SEL["review_text"])[0].get_text()

    updated = bool(frame.select_one(SEL["updated_tag"]))
    managerial_response = False
    previous_version_texts: list | None = None
    previous_version_ratings: list | None = None

    if updated:
        all_prev_nodes = frame.select(SEL["previous_versions"])
        previous_version_texts = []

        for node in all_prev_nodes:
            node_text = node.get_text()
            if _is_managerial_node(node, node_text):
                managerial_response = True
                continue
            previous_version_texts.append(node_text)

        all_rating_nodes = frame.select(SEL["all_ratings"])
        previous_version_ratings = [
            node.attrs.get("aria-label") for node in all_rating_nodes[1:]
        ]

    return {
        "business_id":             business_id,
        "page_number":             page_number,
        "result_id":               count,
        "username":                username,
        "rating":                  rating,
        "date":                    date,
        "review":                  review_text,
        "updated":                 updated,
        "managerial_response":     managerial_response,
        "previous_version":        previous_version_texts if updated else None,
        "previous_version_rating": previous_version_ratings if updated else None,
        "previous_review_date":    None,
    }


def parse_reviews_html(html: str | None, business_id: str, page_number: int) -> pd.DataFrame:
    """Parse all reviews from rendered page HTML into a DataFrame."""
    if html is None:
        return pd.DataFrame([_empty_row(business_id=business_id, page_number=page_number)])

    soup = BeautifulSoup(html, "lxml")
    review_blocks = soup.select(SEL["review_blocks"])
    rows = []

    for count, frame in enumerate(review_blocks, 1):
        try:
            rows.append(_parse_single_review(frame, business_id, page_number, count))
        except Exception as exc:
            print(f"  [Warning] Failed to parse review #{count}: {exc}")
            rows.append(_empty_row(business_id=business_id, page_number=page_number, result_id=count))

    if not rows:
        return pd.DataFrame([_empty_row(business_id=business_id, page_number=page_number)])

    df = pd.DataFrame(rows)
    df = df.explode(["previous_version", "previous_version_rating"], ignore_index=True)
    df["previous_review_date"] = df["previous_version"].apply(
        lambda x: _extract_previous_review_date(x) if pd.notnull(x) else None
    )
    return df


def get_total_review_count_html(html: str) -> int | None:
    soup = BeautifulSoup(html, "lxml")
    nodes = soup.select(SEL["page_count"])
    if not nodes:
        return None
    m = re.search(r"of\s+(\d+)", nodes[0].get_text())
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Main scraping entry point
# ---------------------------------------------------------------------------
def scrape_business(
    business_id: str,
    output_dir: str = "~/Documents/Yelp",
    base_delay: float = 5.0,
    proxies: list[str] | None = None,
    headless: bool = True,
) -> pd.DataFrame:
    """
    Scrape all public reviews for a Yelp business via a headless browser and
    save them to CSV (same CSV format/location as yelp_scraper.scrape_business).
    """
    delay = base_delay + random.uniform(0, 2)
    print(f"[Start] {business_id}  delay={delay:.1f}s")

    first_html = fetch_page_playwright(
        business_id, pagestart=0, proxies=proxies, base_delay=delay, headless=headless
    )

    if first_html is None:
        df = pd.DataFrame([_empty_row(business_id=business_id, page_number="no_pages")])
        save_to_csv(df, business_id, output_dir)
        return df

    total_reviews = get_total_review_count_html(first_html)
    if total_reviews is None:
        print(f"[Warning] Could not determine review count for {business_id}.")
        total_reviews = 10

    pages_needed = (total_reviews + 9) // 10
    print(f"[Info] {total_reviews} reviews across {pages_needed} page(s)")

    all_frames: list[pd.DataFrame] = []

    df0 = parse_reviews_html(first_html, business_id, page_number=0)
    save_to_csv(df0, business_id, output_dir)
    all_frames.append(df0)

    for page in range(1, pages_needed):
        pagestart = page * 10
        print(f"[Fetching] page {page}/{pages_needed - 1}  (start={pagestart})")
        html = fetch_page_playwright(
            business_id, pagestart=pagestart, proxies=proxies, base_delay=delay, headless=headless
        )
        df_page = parse_reviews_html(html, business_id, page_number=page)
        save_to_csv(df_page, business_id, output_dir)
        all_frames.append(df_page)

        jitter = random.uniform(0, 2)
        time.sleep(delay + jitter)

    return pd.concat(all_frames, ignore_index=True)


def scrape_business_list(
    business_ids: list[str],
    output_dir: str = "~/Documents/Yelp",
    base_delay: float = 5.0,
    skip_collected: bool = True,
    proxies: list[str] | None = None,
    headless: bool = True,
) -> None:
    """Scrape a list of business IDs via Playwright, optionally skipping already-done ones."""
    if skip_collected:
        done = already_collected(output_dir)
        business_ids = [b for b in business_ids if b not in done]
        print(f"[Batch] {len(business_ids)} businesses remaining after skip.")

    for business_id in business_ids:
        scrape_business(
            business_id, output_dir=output_dir, base_delay=base_delay,
            proxies=proxies, headless=headless,
        )


# ---------------------------------------------------------------------------
# Example usage (not executed on import)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: rotate through a pool of proxies (see yelp_scraper.load_proxies)
    # from yelp_scraper import load_proxies
    # proxies = load_proxies()

    # Single business
    # df = scrape_business("gary-danko-san-francisco", base_delay=5)

    # Batch
    # from yelp_scraper import load_business_ids_from_dataset
    # ids = load_business_ids_from_dataset("/path/to/yelp_academic_dataset_business.json")
    # scrape_business_list(ids, base_delay=5)
    pass
