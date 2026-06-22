#!/usr/bin/env python3
"""
Script to scrape venue information from Treatwell API for single-employee salon analysis.

For each unique venue in the dataset, this script:
1. Fetches venue data from the Treatwell API
2. Saves the raw JSON response
3. Extracts employee count and venue type
4. Handles different country domains (.de, .ch, .at, .fr, .be, .nl, .co.uk, .es, .pt, .it)
"""

import requests
import time
import json
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Country domains mapping
COUNTRY_DOMAINS = {
    "de": "treatwell.de",
    "ch": "treatwell.ch",
    "at": "treatwell.at",
    "fr": "treatwell.fr",
    "be": "treatwell.be",
    "nl": "treatwell.nl",
    "uk": "treatwell.co.uk",
    "es": "treatwell.es",
    "pt": "treatwell.pt",
    "it": "treatwell.it",
}

# Base headers for API requests
BASE_HEADERS = {
    "accept": "application/json",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "x-accept-api-version": "20250122",
}

# Country-specific header overrides
COUNTRY_HEADERS = {
    "de": {
        "accept-language": "de-DE,de;q=0.9",
        "x-language-code": "de",
    },
    "ch": {
        "accept-language": "de-CH,de;q=0.9",
        "x-language-code": "de",
    },
    "at": {
        "accept-language": "de-AT,de;q=0.9",
        "x-language-code": "de",
    },
    "fr": {
        "accept-language": "fr-FR,fr;q=0.9",
        "x-language-code": "fr",
        "referer": "https://www.treatwell.fr/",
    },
    "be": {
        "accept-language": "fr-BE,fr;q=0.9,nl;q=0.8",
        "x-language-code": "fr",
        "referer": "https://www.treatwell.be/",
    },
    "nl": {
        "accept-language": "nl-NL,nl;q=0.9",
        "x-language-code": "nl",
        "referer": "https://www.treatwell.nl/",
    },
    "uk": {
        "accept-language": "en-GB,en;q=0.9",
        "x-language-code": "en",
        "referer": "https://www.treatwell.co.uk/",
    },
    "es": {
        "accept-language": "es-ES,es;q=0.9",
        "x-language-code": "es",
        "referer": "https://www.treatwell.es/",
    },
    "pt": {
        "accept-language": "pt-PT,pt;q=0.9",
        "x-language-code": "pt",
        "referer": "https://www.treatwell.pt/",
    },
    "it": {
        "accept-language": "it-IT,it;q=0.9",
        "x-language-code": "it",
        "referer": "https://www.treatwell.it/",
    },
}


def get_headers_for_country(country_code: str) -> Dict[str, str]:
    """Get headers appropriate for a specific country."""
    headers = BASE_HEADERS.copy()
    if country_code in COUNTRY_HEADERS:
        headers.update(COUNTRY_HEADERS[country_code])
    else:
        # Default to German if country not found
        headers.update(COUNTRY_HEADERS["de"])
    return headers

# Rate limiting: delay between requests (seconds)
REQUEST_DELAY = 0.25


def get_venue_url(venue_id: int, domain: str, api_version: str = "v1") -> str:
    """Construct the venue API URL for a given venue ID and domain."""
    return f"https://www.{domain}/api/{api_version}/venue/{venue_id}"


def fetch_venue_data(venue_id: int, domain: str, api_version: str = "v1", country_code: str = None) -> Optional[Dict[Any, Any]]:
    """
    Fetch venue data from the API.
    
    Args:
        venue_id: The venue ID to fetch
        domain: The domain (e.g., "treatwell.fr")
        api_version: API version ("v1" or "v2")
        country_code: Country code to determine appropriate headers (e.g., "fr", "be")
    
    Returns:
        JSON response as dict if successful, None otherwise
    """
    url = get_venue_url(venue_id, domain, api_version)
    
    # Determine country code from domain if not provided
    if country_code is None:
        for code, dom in COUNTRY_DOMAINS.items():
            if domain == dom:
                country_code = code
                break
        if country_code is None:
            country_code = "de"  # Default
    
    # Get country-specific headers
    request_headers = get_headers_for_country(country_code)
    
    try:
        response = requests.get(url, headers=request_headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # Venue doesn't exist in this domain/version
            return None
        elif response.status_code == 403:
            # Forbidden - might need different approach
            print(f"  Warning: 403 Forbidden for venue {venue_id} on {domain} (v{api_version})")
            # Try with v2 if we were using v1, or try with different headers
            if api_version == "v1":
                # Try v2 as fallback
                v2_url = get_venue_url(venue_id, domain, "v2")
                v2_response = requests.get(v2_url, headers=request_headers, timeout=10)
                if v2_response.status_code == 200:
                    return v2_response.json()
            return None
        else:
            print(f"  Warning: Status {response.status_code} for venue {venue_id} on {domain} (v{api_version})")
            return None
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching venue {venue_id} from {domain} (v{api_version}): {e}")
        return None


def extract_venue_info(venue_data: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Extract key information from venue data.
    
    Returns:
        Dictionary with employee_count, venue_type, and other relevant info
    """
    employees = venue_data.get("employees", [])
    # Filter to only employees that provide services
    active_employees = [e for e in employees if e.get("providesServices", False)]
    employee_count = len(active_employees)
    
    venue_type = venue_data.get("type", {})
    venue_type_name = venue_type.get("normalisedName", "")
    venue_type_id = venue_type.get("id", None)
    
    return {
        "employee_count": employee_count,
        "venue_type": venue_type_name,
        "venue_type_id": venue_type_id,
        "venue_name": venue_data.get("name", ""),
        "status": venue_data.get("status", ""),
        "listed_on_marketplace": venue_data.get("listedOnMarketplace", False),
    }


def find_venue_domain(venue_id: int, domains_to_try: list = None) -> Tuple[Optional[str], Optional[Dict[Any, Any]]]:
    """
    Try to find which domain a venue exists in.
    
    Returns:
        Tuple of (country_code, venue_data) if found, (None, None) otherwise
    """
    if domains_to_try is None:
        # Try German first (most common), then others
        domains_to_try = ["de"] + [k for k in COUNTRY_DOMAINS.keys() if k != "de"]
    
    # API version mapping: de, ch, at use v1; others use v2
    for country_code in domains_to_try:
        domain = COUNTRY_DOMAINS[country_code]
        # Try v1 first (works for de, ch, at, and sometimes for others)
        api_version = "v1" if country_code in ["de", "ch", "at"] else "v2"
        venue_data = fetch_venue_data(venue_id, domain, api_version, country_code)
        
        # If v2 fails, try v1 as fallback
        if venue_data is None and api_version == "v2":
            venue_data = fetch_venue_data(venue_id, domain, "v1", country_code)
        
        if venue_data is not None:
            return country_code, venue_data
        time.sleep(REQUEST_DELAY)
    
    return None, None


def raw_response_exists(venue_id: int, results_df: pd.DataFrame) -> bool:
    """Check if raw response already exists in the results dataframe."""
    if results_df is None or results_df.empty:
        return False
    venue_row = results_df[results_df["venue_id"] == venue_id]
    if venue_row.empty:
        return False
    # Check if raw_response column exists and has data
    if "raw_response" in venue_row.columns:
        raw_data = venue_row.iloc[0]["raw_response"]
        return pd.notna(raw_data) and raw_data != ""
    return False


def get_raw_response_from_df(venue_id: int, results_df: pd.DataFrame) -> Optional[Dict[Any, Any]]:
    """Get raw response JSON from dataframe if it exists."""
    if results_df is None or results_df.empty:
        return None
    venue_row = results_df[results_df["venue_id"] == venue_id]
    if venue_row.empty:
        return None
    if "raw_response" in venue_row.columns:
        raw_data = venue_row.iloc[0]["raw_response"]
        if pd.notna(raw_data) and raw_data != "":
            try:
                return json.loads(raw_data)
            except Exception:
                return None
    return None


def get_country_from_df(venue_id: int, results_df: pd.DataFrame) -> Optional[str]:
    """Get country code from dataframe if it exists."""
    if results_df is None or results_df.empty:
        return None
    venue_row = results_df[results_df["venue_id"] == venue_id]
    if venue_row.empty:
        return None
    if "country_domain" in venue_row.columns:
        country = venue_row.iloc[0]["country_domain"]
        if pd.notna(country) and country != "":
            return country
    return None


def main():
    # Setup directories
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    snapshots_dir = data_dir / "snapshots"
    
    # Read the dataset
    csv_file = snapshots_dir / "treatwell_without_raw-all-2025-06-02.csv"
    print(f"Reading dataset from {csv_file}")
    df = pd.read_csv(csv_file, low_memory=False)
    
    # Get unique venue IDs
    unique_venue_ids = df["id"].unique().tolist()
    total_venues = len(unique_venue_ids)
    print(f"Found {total_venues} unique venues")
    
    # Check for existing results to resume
    results_file = snapshots_dir / f"venue_info-{datetime.today().strftime('%Y-%m-%d')}.csv"
    existing_df = None
    existing_venue_ids = set()
    if results_file.exists():
        existing_df = pd.read_csv(results_file, low_memory=False)
        existing_venue_ids = set(existing_df["venue_id"].unique())
        print(f"Found existing results for {len(existing_venue_ids)} venues")
        # Count how many have raw responses
        if "raw_response" in existing_df.columns:
            raw_count = existing_df["raw_response"].notna().sum()
            print(f"  {raw_count} venues have raw response data stored")
    
    # Prepare results list
    results = []
    failed_venues = []
    
    # Process each venue
    for idx, venue_id in enumerate(unique_venue_ids, 1):
        # Skip if already processed
        if venue_id in existing_venue_ids:
            print(f"[{idx}/{total_venues}] Skipping venue {venue_id} (already processed)")
            continue
        
        print(f"[{idx}/{total_venues}] Processing venue {venue_id}...")
        
        # Check if we have raw response in existing dataframe but no processed result
        if existing_df is not None and raw_response_exists(venue_id, existing_df) and venue_id not in existing_venue_ids:
            # Load existing raw response from dataframe
            venue_data = get_raw_response_from_df(venue_id, existing_df)
            country_code = get_country_from_df(venue_id, existing_df)
            if country_code is None:
                country_code = "unknown"
            print(f"  Loaded existing raw response from CSV, country: {country_code}")
        else:
            # Try to find the venue in one of the domains (this will fetch the full data)
            country_code, venue_data = find_venue_domain(venue_id)
        
        if venue_data is None:
            print(f"  Failed to find venue {venue_id} in any domain")
            failed_venues.append(venue_id)
            results.append({
                "venue_id": venue_id,
                "country_domain": None,
                "employee_count": None,
                "venue_type": None,
                "venue_type_id": None,
                "venue_name": None,
                "status": None,
                "listed_on_marketplace": None,
                "raw_response": None,
                "found": False,
            })
        else:
            # Extract information
            info = extract_venue_info(venue_data)
            
            # Convert venue_data to JSON string for storage in CSV
            raw_response_json = json.dumps(venue_data, ensure_ascii=False)
            
            print(f"  Found on {country_code}: {info['venue_name']} - {info['employee_count']} employees, type: {info['venue_type']}")
            
            results.append({
                "venue_id": venue_id,
                "country_domain": country_code,
                "employee_count": info["employee_count"],
                "venue_type": info["venue_type"],
                "venue_type_id": info["venue_type_id"],
                "venue_name": info["venue_name"],
                "status": info["status"],
                "listed_on_marketplace": info["listed_on_marketplace"],
                "raw_response": raw_response_json,
                "found": True,
            })
        
        # Save progress periodically (every 100 venues)
        if idx % 100 == 0:
            # Combine existing and new results
            if existing_df is not None and not existing_df.empty:
                # Merge: keep existing rows, update/add new ones
                existing_dict = {row["venue_id"]: row.to_dict() for _, row in existing_df.iterrows()}
                for result in results:
                    existing_dict[result["venue_id"]] = result
                all_results = list(existing_dict.values())
            else:
                all_results = results
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(results_file, index=False)
            print(f"  Progress saved to {results_file}")
    
    # Final save
    if existing_df is not None and not existing_df.empty:
        # Merge: keep existing rows, update/add new ones
        existing_dict = {row["venue_id"]: row.to_dict() for _, row in existing_df.iterrows()}
        for result in results:
            existing_dict[result["venue_id"]] = result
        all_results = list(existing_dict.values())
    else:
        all_results = results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Scraping complete!")
    print(f"Total venues processed: {total_venues}")
    print(f"Successfully found: {len([r for r in all_results if r.get('found', False)])}")
    print(f"Failed to find: {len(failed_venues)}")
    print(f"Results saved to: {results_file}")
    print(f"Raw responses are stored in the 'raw_response' column of the CSV")
    
    if failed_venues:
        print(f"\nFailed venue IDs: {failed_venues[:20]}...")  # Show first 20
        if len(failed_venues) > 20:
            print(f"(and {len(failed_venues) - 20} more)")


if __name__ == "__main__":
    main()
