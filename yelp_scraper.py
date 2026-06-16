import os
import json
import time
import re
import random

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CSS selectors — centralised so Yelp HTML changes only need one edit here
# ---------------------------------------------------------------------------
SEL = {
    "review_blocks":        ".y-css-1sqelp2 > .y-css-mhg9c5",
    "username":             "#reviews .y-css-160a82h .y-css-1x1e1r2",
    "rating":               "#reviews .y-css-9vtc3g+ .y-css-scqtta .y-css-dnttlc",
    "date":                 ".y-css-9vtc3g+ .y-css-scqtta .y-css-1vi7y4e",
    "review_text":          ".y-css-1pnalxe .raw__09f24__T4Ezm",
    "updated_tag":          ".y-css-9vtc3g+ .y-css-scqtta .y-css-1ob74fm",
    "previous_versions":    ".y-css-xl3e5f",
    "mgr_response_marker":  ".truncated__09f24__lSBbT",
    "mgr_response_nested":  ".y-css-mhg9c5+ .y-css-po0lpl .y-css-xl3e5f",
    "all_ratings":          ".y-css-dnttlc",
    "page_count":           ".y-css-1w88y64 .y-css-1vi7y4e",
}

USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
]

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
_EMPTY_REVIEW = {
    "business_id":              None,
    "page_number":              None,
    "result_id":                None,
    "username":                 None,
    "rating":                   None,
    "date":                     None,
    "review":                   None,
    "updated":                  None,
    "managerial_response":      False,
    "previous_version":         None,
    "previous_version_rating":  None,
    "previous_review_date":     None,
}


def _empty_row(**overrides) -> dict:
    return {**_EMPTY_REVIEW, **overrides}


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------
def _random_headers() -> dict:
    return {"User-Agent": random.choice(USER_AGENTS)}


def fetch_page(business_id: str, pagestart: int,
               max_retries: int = 10, base_delay: float = 20.0) -> requests.Response | None:
    """Fetch one Yelp review page, retrying on transient errors."""
    url = f"https://www.yelp.com/biz/{business_id}?start={pagestart}#reviews"
    delay = base_delay

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=_random_headers(), timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "lxml")
                if soup.select(SEL["review_blocks"]):
                    print(f"[OK {response.status_code}] page start={pagestart}")
                    return response
                print(f"[Empty] No review blocks found — reached end of pages (start={pagestart})")
                return None
            print(
                f"[Attempt {attempt}/{max_retries}] HTTP {response.status_code} "
                f"for '{business_id}' (url={url}), retrying in {delay:.1f}s…"
            )
        except requests.exceptions.RequestException as exc:
            print(f"[Attempt {attempt}/{max_retries}] Request error: {exc}, retrying in {delay:.1f}s…")

        jitter = random.uniform(0, 5)
        time.sleep(delay + jitter)
        delay = min(delay * 1.5, 120)   # cap at 2 minutes

    print(f"[Failure] Gave up after {max_retries} attempts for {business_id} start={pagestart}.")
    return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _extract_previous_review_date(text: str) -> pd.Timestamp | None:
    match = re.search(r"(.*?)(?=Previous review)", text)
    if match:
        try:
            return pd.to_datetime(match.group(1).strip())
        except Exception:
            return None
    return None


def _rating_label_to_float(label: str) -> float | None:
    """'3 star rating' → 3.0"""
    m = re.search(r"([\d.]+)", label)
    return float(m.group(1)) if m else None


def _is_managerial_node(node, text: str) -> bool:
    return (
        bool(node.select(SEL["mgr_response_marker"]))
        or bool(node.select(SEL["mgr_response_nested"]))
        or "Business owner information" in text
    )


def _parse_single_review(frame, business_id: str, page_number: int, count: int) -> dict:
    """Parse one review block; raises on failure so caller can catch."""
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
        # First node is the current rating; the rest are previous ratings
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
        "previous_review_date":    None,  # filled in after explode
    }


def parse_reviews(response: requests.Response | None,
                  business_id: str, page_number: int) -> pd.DataFrame:
    """Parse all reviews from a single page response into a DataFrame."""
    if response is None:
        return pd.DataFrame([_empty_row(business_id=business_id, page_number=page_number)])

    soup = BeautifulSoup(response.content, "lxml")
    review_blocks = soup.select(SEL["review_blocks"])
    rows = []

    for count, frame in enumerate(review_blocks, 1):
        try:
            print(f"  [Parsing] review #{count}, page {page_number}")
            rows.append(_parse_single_review(frame, business_id, page_number, count))
        except Exception as exc:
            print(f"  [Warning] Failed to parse review #{count}: {exc}")
            rows.append(_empty_row(business_id=business_id, page_number=page_number, result_id=count))

    if not rows:
        return pd.DataFrame([_empty_row(business_id=business_id, page_number=page_number)])

    df = pd.DataFrame(rows)

    # Explode list-valued previous_version / previous_version_rating columns
    df = df.explode(["previous_version", "previous_version_rating"], ignore_index=True)

    df["previous_review_date"] = df["previous_version"].apply(
        lambda x: _extract_previous_review_date(x) if pd.notnull(x) else None
    )

    return df


def get_total_review_count(response: requests.Response) -> int | None:
    """Return total number of reviews from the pagination text (e.g. '1-10 of 47')."""
    soup = BeautifulSoup(response.content, "lxml")
    nodes = soup.select(SEL["page_count"])
    if not nodes:
        return None
    m = re.search(r"of\s+(\d+)", nodes[0].get_text())
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------
def save_to_csv(df: pd.DataFrame, business_id: str,
                output_dir: str = "~/Documents/Yelp") -> None:
    dir_path = os.path.expanduser(output_dir)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"yelp_{business_id}.csv")
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, mode="w", header=True, index=False)
    print(f"[Saved] {len(df)} rows → {file_path}")


# ---------------------------------------------------------------------------
# Main scraping entry point
# ---------------------------------------------------------------------------
def scrape_business(business_id: str,
                    output_dir: str = "~/Documents/Yelp",
                    base_delay: float = 20.0) -> pd.DataFrame:
    """
    Scrape all public reviews for a Yelp business and save them to CSV.

    Parameters
    ----------
    business_id : str
        The Yelp business slug/ID as it appears in the URL, e.g.
        'gary-danko-san-francisco' or 'mpf3x-BjTdTEA3yCZrAYPw'.
    output_dir : str
        Directory where per-business CSV files are written.
    base_delay : float
        Base seconds to wait between page requests (jitter is added).

    Returns
    -------
    pd.DataFrame with all scraped reviews (also written to CSV incrementally).
    """
    delay = base_delay + random.uniform(0, 5)
    print(f"[Start] {business_id}  delay={delay:.1f}s")

    first_response = fetch_page(business_id, pagestart=0, base_delay=delay)

    if first_response is None:
        df = pd.DataFrame([_empty_row(business_id=business_id, page_number="no_pages")])
        save_to_csv(df, business_id, output_dir)
        return df

    total_reviews = get_total_review_count(first_response)
    if total_reviews is None:
        print(f"[Warning] Could not determine review count for {business_id}.")
        total_reviews = 10   # assume single page

    pages_needed = (total_reviews + 9) // 10   # Yelp shows 10 per page
    print(f"[Info] {total_reviews} reviews across {pages_needed} page(s)")

    all_frames: list[pd.DataFrame] = []

    # First page — already fetched
    df0 = parse_reviews(first_response, business_id, page_number=0)
    save_to_csv(df0, business_id, output_dir)
    all_frames.append(df0)

    for page in range(1, pages_needed):
        pagestart = page * 10
        print(f"[Fetching] page {page}/{pages_needed - 1}  (start={pagestart})")
        response = fetch_page(business_id, pagestart=pagestart, base_delay=delay)
        df_page = parse_reviews(response, business_id, page_number=page)
        save_to_csv(df_page, business_id, output_dir)
        all_frames.append(df_page)

        jitter = random.uniform(0, 5)
        time.sleep(delay + jitter)

    return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Batch scraping helpers (for Yelp Academic Dataset business lists)
# ---------------------------------------------------------------------------
def already_collected(output_dir: str = "~/Documents/Yelp") -> set[str]:
    """Return set of business_ids for which a CSV already exists."""
    path = os.path.expanduser(output_dir)
    if not os.path.isdir(path):
        return set()
    return {
        f[len("yelp_"):-len(".csv")]
        for f in os.listdir(path)
        if f.startswith("yelp_") and f.endswith(".csv")
    }


def scrape_business_list(business_ids: list[str],
                         output_dir: str = "~/Documents/Yelp",
                         base_delay: float = 20.0,
                         skip_collected: bool = True) -> None:
    """Scrape a list of business IDs, optionally skipping already-done ones."""
    if skip_collected:
        done = already_collected(output_dir)
        business_ids = [b for b in business_ids if b not in done]
        print(f"[Batch] {len(business_ids)} businesses remaining after skip.")

    for business_id in tqdm(business_ids, desc="Scraping"):
        scrape_business(business_id, output_dir=output_dir, base_delay=base_delay)


def load_business_ids_from_dataset(json_path: str) -> list[str]:
    """
    Extract all business_id values from the Yelp Academic Dataset business file
    (yelp_academic_dataset_business.json — one JSON object per line).
    """
    ids = []
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["business_id"])
    print(f"[Info] Loaded {len(ids)} business IDs from {json_path}")
    return ids


def load_business_ids_from_json(json_path: str, chunk_index: int = 0) -> list[str]:
    """
    Load a chunk of business IDs from a JSON or JSON Lines chunks file.

    Supports two layouts:
      - Standard JSON:  a single array-of-arrays  [[id, ...], [id, ...], ...]
      - JSON Lines:     one JSON value per line    [id, ...]\n[id, ...]\n...
    """
    with open(json_path) as f:
        content = f.read().strip()

    # Try standard JSON first
    try:
        chunks = json.loads(content)
        return chunks[chunk_index]
    except json.JSONDecodeError:
        pass

    # Fall back to JSON Lines (one value per line)
    chunks = [json.loads(line) for line in content.splitlines() if line.strip()]
    return chunks[chunk_index]


# ---------------------------------------------------------------------------
# Example usage (not executed on import)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Single business
    # scrape_business("gary-danko-san-francisco", base_delay=3)

    # Batch from Yelp Academic Dataset chunks file
    # ids = load_business_ids_from_json(
    #     "/path/to/unique_business_names.json", chunk_index=0
    # )
    # scrape_business_list(ids, base_delay=3)
    pass
