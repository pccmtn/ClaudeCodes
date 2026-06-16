"""
Yelp Fusion API client.

Docs: https://docs.developer.yelp.com/docs/fusion-intro
Rate limit: 500 requests/day on the free tier.

Usage:
    Set YELP_API_KEY in your environment, then:

        from yelp_fusion import YelpFusion
        client = YelpFusion()

        # Fetch business details by Yelp business ID
        biz = client.get_business("gary-danko-san-francisco")

        # Fetch reviews for a business (up to 3 on free tier)
        reviews = client.get_reviews("gary-danko-san-francisco")

        # Scrape all businesses from a list, saving to CSV
        client.scrape_business_list(["gary-danko-san-francisco", ...])
"""

import os
import time
import random
import pandas as pd
import requests

FUSION_BASE = "https://api.yelp.com/v3"


class YelpFusion:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("YELP_API_KEY")
        if not key:
            raise ValueError(
                "Provide api_key= or set the YELP_API_KEY environment variable. "
                "Get a key at https://www.yelp.com/developers/v3/manage_app"
            )
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {key}"})

    # ------------------------------------------------------------------
    # Core API calls
    # ------------------------------------------------------------------

    def get_business(self, business_id: str) -> dict:
        """Return full business details for a single Yelp business ID."""
        url = f"{FUSION_BASE}/businesses/{business_id}"
        response = self._session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_reviews(self, business_id: str, limit: int = 3,
                    sort_by: str = "newest") -> list[dict]:
        """
        Return public reviews for a business.

        Free-tier accounts receive up to 3 reviews per business.
        Academic/partner access may unlock more.

        Parameters
        ----------
        business_id : str
        limit : int
            Max reviews to return (1–3 on free tier).
        sort_by : str
            "newest" | "oldest" | "rating"
        """
        url = f"{FUSION_BASE}/businesses/{business_id}/reviews"
        params = {"limit": min(limit, 3), "sort_by": sort_by}
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("reviews", [])

    def search(self, term: str = "", location: str = "",
               latitude: float | None = None, longitude: float | None = None,
               limit: int = 20, offset: int = 0, **kwargs) -> dict:
        """
        Search for businesses.

        At least one of `location` or (`latitude` + `longitude`) is required.
        Returns the raw API response dict (keys: businesses, total, region).
        """
        params: dict = {"term": term, "limit": limit, "offset": offset, **kwargs}
        if location:
            params["location"] = location
        if latitude is not None and longitude is not None:
            params["latitude"] = latitude
            params["longitude"] = longitude

        url = f"{FUSION_BASE}/businesses/search"
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # DataFrame helpers
    # ------------------------------------------------------------------

    def business_to_row(self, business_id: str) -> dict:
        """Fetch a business and return a flat dict suitable for a DataFrame row."""
        try:
            biz = self.get_business(business_id)
        except requests.HTTPError as exc:
            print(f"[Error] {business_id}: {exc}")
            return {"business_id": business_id, "error": str(exc)}

        location = biz.get("location", {})
        coords = biz.get("coordinates", {})
        return {
            "business_id":      biz.get("id"),
            "name":             biz.get("name"),
            "rating":           biz.get("rating"),
            "review_count":     biz.get("review_count"),
            "price":            biz.get("price"),
            "phone":            biz.get("phone"),
            "is_closed":        biz.get("is_closed"),
            "url":              biz.get("url"),
            "address":          ", ".join(location.get("display_address", [])),
            "city":             location.get("city"),
            "state":            location.get("state"),
            "zip_code":         location.get("zip_code"),
            "country":          location.get("country"),
            "latitude":         coords.get("latitude"),
            "longitude":        coords.get("longitude"),
            "categories":       ", ".join(
                c["title"] for c in biz.get("categories", [])
            ),
            "hours_open":       _parse_hours(biz.get("hours")),
        }

    def reviews_to_rows(self, business_id: str) -> list[dict]:
        """Fetch reviews and return a list of flat dicts."""
        try:
            reviews = self.get_reviews(business_id)
        except requests.HTTPError as exc:
            print(f"[Error] reviews for {business_id}: {exc}")
            return [{"business_id": business_id, "error": str(exc)}]

        rows = []
        for r in reviews:
            rows.append({
                "business_id": business_id,
                "review_id":   r.get("id"),
                "rating":      r.get("rating"),
                "text":        r.get("text"),
                "time_created": r.get("time_created"),
                "url":         r.get("url"),
                "user_name":   r.get("user", {}).get("name"),
                "user_id":     r.get("user", {}).get("id"),
            })
        return rows

    # ------------------------------------------------------------------
    # Batch scraping
    # ------------------------------------------------------------------

    def scrape_business_list(
        self,
        business_ids: list[str],
        output_dir: str = "~/Documents/Yelp/fusion",
        fetch_reviews: bool = True,
        base_delay: float = 1.5,
        skip_collected: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch business details (and optionally reviews) for a list of IDs.

        Saves two CSVs:
          - yelp_fusion_businesses.csv
          - yelp_fusion_reviews.csv   (if fetch_reviews=True)

        Returns (businesses_df, reviews_df).
        """
        import os as _os
        dir_path = _os.path.expanduser(output_dir)
        _os.makedirs(dir_path, exist_ok=True)

        biz_path = _os.path.join(dir_path, "yelp_fusion_businesses.csv")
        rev_path = _os.path.join(dir_path, "yelp_fusion_reviews.csv")

        # Skip already-collected IDs
        collected: set[str] = set()
        if skip_collected and _os.path.exists(biz_path):
            existing = pd.read_csv(biz_path, usecols=["business_id"])
            collected = set(existing["business_id"].dropna())
            print(f"[Skip] {len(collected)} already collected, "
                  f"{len(business_ids) - len(collected)} remaining.")
            business_ids = [b for b in business_ids if b not in collected]

        biz_rows: list[dict] = []
        rev_rows: list[dict] = []

        for i, bid in enumerate(business_ids, 1):
            print(f"[{i}/{len(business_ids)}] {bid}")

            biz_rows.append(self.business_to_row(bid))

            if fetch_reviews:
                rev_rows.extend(self.reviews_to_rows(bid))

            # Append to CSV every 50 records to avoid data loss on interruption
            if i % 50 == 0 or i == len(business_ids):
                _append_csv(pd.DataFrame(biz_rows), biz_path)
                biz_rows = []
                if fetch_reviews and rev_rows:
                    _append_csv(pd.DataFrame(rev_rows), rev_path)
                    rev_rows = []

            delay = base_delay + random.uniform(0, 0.5)
            time.sleep(delay)

        biz_df = pd.read_csv(biz_path) if _os.path.exists(biz_path) else pd.DataFrame()
        rev_df = pd.read_csv(rev_path) if _os.path.exists(rev_path) else pd.DataFrame()
        return biz_df, rev_df


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _parse_hours(hours_list: list | None) -> str | None:
    """Flatten the hours structure into a readable string."""
    if not hours_list:
        return None
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    segments = []
    for entry in hours_list:
        for slot in entry.get("open", []):
            day = days[slot.get("day", 0)]
            start = slot.get("start", "")
            end = slot.get("end", "")
            segments.append(f"{day} {start[:2]}:{start[2:]}-{end[:2]}:{end[2:]}")
    return "; ".join(segments) if segments else None


def _append_csv(df: pd.DataFrame, path: str) -> None:
    import os as _os
    if df.empty:
        return
    if _os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)
    print(f"  [Saved] {len(df)} rows → {path}")


# ------------------------------------------------------------------
# Example usage (not executed on import)
# ------------------------------------------------------------------

if __name__ == "__main__":
    client = YelpFusion()  # reads YELP_API_KEY from environment

    # --- Single business ---
    # biz = client.get_business("gary-danko-san-francisco")
    # print(biz["name"], biz["rating"])

    # --- Reviews for one business ---
    # reviews = client.get_reviews("gary-danko-san-francisco")
    # for r in reviews:
    #     print(r["rating"], r["text"][:80])

    # --- Batch scrape from the Academic Dataset business list ---
    # from yelp_scraper import load_business_ids_from_dataset
    # ids = load_business_ids_from_dataset("/path/to/yelp_academic_dataset_business.json")
    # client.scrape_business_list(ids, base_delay=1.5)
