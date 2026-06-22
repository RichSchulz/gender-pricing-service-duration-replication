#!/usr/bin/env python3
"""
Analyze treatmentCategoryIds from Treatwell dataset to categorize cuts by type.

This script processes the raw JSON data in treatwell-all-2025-06-02.csv to extract and analyze
all treatmentCategoryIds, categorizing them as:
- Machine cuts (clipper/machine cuts)
- Wet cuts (scissors cuts with washing/shampoo)
- Dry cuts (scissors cuts without washing)
- Unknown/ambiguous (cannot be clearly categorized)

Focuses on IDs with at least 1000 cuts and filters out non-cut treatments using filter words.
Shows only German treatment names for better readability.
"""

import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import sys
import os
from pathlib import Path

# Import filter words from utils.py
# Suppress the print statement in utils.py by redirecting stdout temporarily
try:
    from io import StringIO
    _old_stdout = sys.stdout
    sys.stdout = StringIO()
    import utils
    filter_words = utils.filter_words
    sys.stdout = _old_stdout
except ImportError:
    print("Warning: Could not import utils.py. Filter words will not be available.")
    filter_words = []


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


# Known treatment category IDs
MALE_IDS = {716, 795}
FEMALE_IDS = {166, 792}
KIDS_IDS = {729, 766, 767}


# Keywords for machine cuts (case-insensitive)
MACHINE_KEYWORDS = [
    # German
    "maschinen", "maschinenschnitt", "clipper", "rasur",
    # English
    "machine", "clipper", "clipper cut",
    # French
    "machine", "tondeuse",
    # Spanish
    "máquina", "maquina",
    # Italian
    "macchina", "macchinetta",
    # Dutch
    "machine", "tondeuse",
    # Portuguese
    "máquina", "maquina",
]

# Keywords for wet cuts (with washing)
WET_CUT_KEYWORDS = [
    # German
    "waschen", "shampoo", "shampoing", "nass", "mit waschen",
    # English
    "wash", "shampoo", "wet", "with wash", "haircut & wash",
    # French
    "shampoing", "lavage", "avec shampoing",
    # Spanish
    "lavado", "lavar", "con lavado", "corte y lavado",
    # Italian
    "shampoo", "lavaggio", "con shampoo", "taglio e shampoo",
    # Dutch
    "wassen", "shampoo", "met wassen",
    # Portuguese
    "lavado", "lavar", "shampoo", "com lavado", "corte e lavado",
]

# Keywords for dry cuts (without washing)
DRY_CUT_KEYWORDS = [
    # German
    "trocken", "trockenschnitt", "ohne waschen", "trockenhaarschnitt",
    # English
    "dry", "dry cut", "without wash", "no wash", "dry haircut",
    # French
    "à sec", "sec", "sans shampoing", "coupe à sec",
    # Spanish
    "seco", "sin lavado", "corte seco", "corte sin lavado",
    # Italian
    "asciutto", "senza shampoo", "taglio a secco", "taglio senza shampoo",
    # Dutch
    "droog", "zonder wassen", "droge knip",
    # Portuguese
    "seco", "sem lavado", "corte seco", "corte sem lavado",
]


def contains_machine_keyword(text: str) -> bool:
    """Check if text contains any machine-related keywords (case-insensitive)."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in MACHINE_KEYWORDS)


def contains_wet_keyword(text: str) -> bool:
    """Check if text contains wet cut keywords (washing/shampoo)."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in WET_CUT_KEYWORDS)


def contains_dry_keyword(text: str) -> bool:
    """Check if text contains dry cut keywords (no washing)."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in DRY_CUT_KEYWORDS)


def categorize_cut_type(text: str) -> str:
    """
    Categorize cut type: machine, wet, dry, or unknown.

    Priority: machine > wet > dry > unknown
    """
    if not text or not isinstance(text, str):
        return "unknown"

    if contains_machine_keyword(text):
        return "machine"
    elif contains_wet_keyword(text):
        return "wet"
    elif contains_dry_keyword(text):
        return "dry"
    else:
        return "unknown"


def should_filter_out(text: str) -> bool:
    """
    Check if treatment name should be filtered out (contains filter words).

    Filter words indicate non-cut treatments like coloring, perms, extensions, etc.
    """
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(filter_word.lower() in text_lower for filter_word in filter_words)


# Common German words in treatment names
GERMAN_INDICATORS = [
    # Gender/people
    "herren", "damen", "kinder", "jungen", "mädchen", "frau", "mann",
    # Hair cutting
    "haarschnitt", "schnitt", "schneiden", "knippen",
    # Services
    "waschen", "föhnen", "styling", "stylen", "coiffage",
    # Beard
    "bart", "rasur", "bartrasur", "bartschnitt",
    # Other common words
    "inkl", "inklusive", "ab", "von", "bis", "mit", "ohne", "und", "oder",
    # German-specific characters/words
    "ä", "ö", "ü", "ß",
    # Common treatment words
    "behandlung", "paket", "service", "premium", "deluxe",
]


def is_german_name(text: str) -> bool:
    """
    Check if treatment name is likely German.

    Uses common German words and characters to identify German treatment names.
    """
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()

    # Check for German-specific characters
    if any(char in text for char in ["ä", "ö", "ü", "ß", "Ä", "Ö", "Ü"]):
        return True

    # Check for common German words
    if any(indicator in text_lower for indicator in GERMAN_INDICATORS):
        return True

    return False


def infer_gender(treatment_id: int) -> str:
    """Infer gender category from treatment ID."""
    if treatment_id in MALE_IDS:
        return "male"
    elif treatment_id in FEMALE_IDS:
        return "female"
    elif treatment_id in KIDS_IDS:
        return "kids"
    else:
        return "unknown"


def extract_treatment_ids_from_raw(raw_json_str: str) -> List[Tuple[int, str]]:
    """
    Extract all treatmentCategoryIds and their associated names from raw JSON.

    Returns:
        List of tuples: (treatmentCategoryId, treatment_name)
    """
    if not raw_json_str or pd.isna(raw_json_str):
        return []

    try:
        result = json.loads(raw_json_str)
    except (json.JSONDecodeError, TypeError):
        return []

    treatment_pairs = []

    try:
        menu_highlights = result.get("data", {}).get("menuHighlights", [])
        for highlight in menu_highlights:
            highlight_data = highlight.get("data", {})
            treatment_name = highlight_data.get("name", "")
            treatment_category_ids = highlight_data.get("treatmentCategoryIds", [])

            # Each treatment name can be associated with multiple IDs
            for treatment_id in treatment_category_ids:
                if isinstance(treatment_id, int):
                    treatment_pairs.append((treatment_id, treatment_name))
    except (KeyError, AttributeError, TypeError):
        pass

    return treatment_pairs


def analyze_treatment_ids(csv_path: str, chunk_size: int = 1000, min_count: int = 1000):
    """
    Analyze treatmentCategoryIds from the CSV file.

    Args:
        csv_path: Path to the CSV file
        chunk_size: Number of rows to process at a time
        min_count: Minimum number of occurrences to include an ID in analysis
    """
    print(f"Loading and analyzing {csv_path}...")
    print(f"Filtering out non-cut treatments using {len(filter_words)} filter words")
    print(f"Focusing on IDs with at least {min_count} occurrences")
    print("=" * 80)

    # Data structures to collect information
    treatment_stats: Dict[int, Dict] = defaultdict(lambda: {
        "count": 0,
        "filtered_count": 0,  # Count of filtered-out treatments
        "names": set(),
        "filtered_names": set(),  # Names that were filtered out
        "machine_count": 0,
        "wet_count": 0,
        "dry_count": 0,
        "unknown_count": 0,
        "machine_names": set(),
        "wet_names": set(),
        "dry_names": set(),
        "unknown_names": set(),
        "appears_with_male": False,
        "appears_with_female": False,
    })

    # Process CSV in chunks
    processed_rows = 0

    try:
        print("Processing CSV file in chunks...")

        # Process the data
        for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            chunk_start = chunk_idx * chunk_size

            for idx, row in chunk.iterrows():
                processed_rows += 1

                # Extract treatment IDs and names from raw JSON
                treatment_pairs = extract_treatment_ids_from_raw(row.get("raw", ""))

                # Track gender associations from CSV columns
                is_male = row.get("is_male", False)
                is_female = row.get("is_female", False)

                # Process each treatment ID
                for treatment_id, treatment_name in treatment_pairs:
                    # Check if this treatment should be filtered out
                    if should_filter_out(treatment_name):
                        treatment_stats[treatment_id]["filtered_count"] += 1
                        treatment_stats[treatment_id]["filtered_names"].add(treatment_name)
                        continue  # Skip filtered treatments

                    # Only count non-filtered treatments
                    treatment_stats[treatment_id]["count"] += 1
                    treatment_stats[treatment_id]["names"].add(treatment_name)

                    # Categorize cut type
                    cut_type = categorize_cut_type(treatment_name)
                    if cut_type == "machine":
                        treatment_stats[treatment_id]["machine_count"] += 1
                        treatment_stats[treatment_id]["machine_names"].add(treatment_name)
                    elif cut_type == "wet":
                        treatment_stats[treatment_id]["wet_count"] += 1
                        treatment_stats[treatment_id]["wet_names"].add(treatment_name)
                    elif cut_type == "dry":
                        treatment_stats[treatment_id]["dry_count"] += 1
                        treatment_stats[treatment_id]["dry_names"].add(treatment_name)
                    else:
                        treatment_stats[treatment_id]["unknown_count"] += 1
                        treatment_stats[treatment_id]["unknown_names"].add(treatment_name)

                    # Track gender associations
                    if is_male:
                        treatment_stats[treatment_id]["appears_with_male"] = True
                    if is_female:
                        treatment_stats[treatment_id]["appears_with_female"] = True

            # Progress update after each chunk
            if (chunk_idx + 1) % 10 == 0:
                print(f"  Processed {processed_rows} rows (chunk {chunk_idx + 1})...")

        print(f"\nCompleted processing {processed_rows} rows.")
        print("=" * 80)

    except Exception as e:
        print(f"Error processing CSV: {e}")
        raise

    # Filter to only IDs with at least min_count occurrences
    significant_ids = {
        tid: stats for tid, stats in treatment_stats.items()
        if stats["count"] >= min_count
    }

    print(f"\nFound {len(significant_ids)} IDs with at least {min_count} occurrences (after filtering)")
    print("=" * 80)

    # Prepare summary data for significant IDs only
    summary_data = []
    for treatment_id in sorted(significant_ids.keys()):
        stats = treatment_stats[treatment_id]

        # Filter to only German names for each category
        german_machine = sorted([name for name in stats["machine_names"] if is_german_name(name)])[:10]
        german_wet = sorted([name for name in stats["wet_names"] if is_german_name(name)])[:10]
        german_dry = sorted([name for name in stats["dry_names"] if is_german_name(name)])[:10]
        german_unknown = sorted([name for name in stats["unknown_names"] if is_german_name(name)])[:10]

        # Determine primary cut type (most common)
        cut_counts = {
            "machine": stats["machine_count"],
            "wet": stats["wet_count"],
            "dry": stats["dry_count"],
            "unknown": stats["unknown_count"],
        }
        primary_cut_type = max(cut_counts.items(), key=lambda x: x[1])[0] if any(cut_counts.values()) else "unknown"

        summary_data.append({
            "treatmentCategoryId": treatment_id,
            "count": stats["count"],
            "filtered_count": stats["filtered_count"],
            "machine_count": stats["machine_count"],
            "wet_count": stats["wet_count"],
            "dry_count": stats["dry_count"],
            "unknown_count": stats["unknown_count"],
            "primary_cut_type": primary_cut_type,
            "german_machine_names": german_machine,
            "german_wet_names": german_wet,
            "german_dry_names": german_dry,
            "german_unknown_names": german_unknown,
            "likely_gender": infer_gender(treatment_id),
            "appears_with_male": stats["appears_with_male"],
            "appears_with_female": stats["appears_with_female"],
        })

    # Sort by count descending
    summary_data.sort(key=lambda x: x["count"], reverse=True)

    # Print detailed summary table
    print("\n" + "=" * 80)
    print(f"DETAILED SUMMARY: Treatment Category IDs (≥{min_count} occurrences)")
    print("=" * 80)

    for item in summary_data:
        print(f"\n{'='*80}")
        print(f"ID {item['treatmentCategoryId']}: {item['count']:,} total occurrences")
        print(f"  Gender: {item['likely_gender']}")
        print(f"  Primary cut type: {item['primary_cut_type'].upper()}")
        print(f"  Cut type breakdown:")
        if item['count'] > 0:
            print(f"    - Machine cuts: {item['machine_count']:,} ({100*item['machine_count']/item['count']:.1f}%)")
            print(f"    - Wet cuts: {item['wet_count']:,} ({100*item['wet_count']/item['count']:.1f}%)")
            print(f"    - Dry cuts: {item['dry_count']:,} ({100*item['dry_count']/item['count']:.1f}%)")
            print(f"    - Unknown/ambiguous: {item['unknown_count']:,} ({100*item['unknown_count']/item['count']:.1f}%)")
        else:
            print(f"    - Machine cuts: {item['machine_count']:,} (0.0%)")
            print(f"    - Wet cuts: {item['wet_count']:,} (0.0%)")
            print(f"    - Dry cuts: {item['dry_count']:,} (0.0%)")
            print(f"    - Unknown/ambiguous: {item['unknown_count']:,} (0.0%)")
        print(f"  Appears with male: {item['appears_with_male']}")
        print(f"  Appears with female: {item['appears_with_female']}")
        if item['filtered_count'] > 0:
            print(f"  Filtered out (non-cut): {item['filtered_count']:,} treatments")

        # Show German names by category
        if item['german_machine_names']:
            print(f"\n  German MACHINE cut names (up to 10):")
            for i, name in enumerate(item['german_machine_names'], 1):
                print(f"    {i:2d}. {name}")

        if item['german_wet_names']:
            print(f"\n  German WET cut names (up to 10):")
            for i, name in enumerate(item['german_wet_names'], 1):
                print(f"    {i:2d}. {name}")

        if item['german_dry_names']:
            print(f"\n  German DRY cut names (up to 10):")
            for i, name in enumerate(item['german_dry_names'], 1):
                print(f"    {i:2d}. {name}")

        if item['german_unknown_names']:
            print(f"\n  German UNKNOWN/AMBIGUOUS cut names (up to 10):")
            for i, name in enumerate(item['german_unknown_names'], 1):
                print(f"    {i:2d}. {name}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total IDs analyzed (≥{min_count} occurrences): {len(summary_data)}")
    print(f"Total occurrences: {sum(item['count'] for item in summary_data):,}")
    total_filtered = sum(item['filtered_count'] for item in summary_data)
    print(f"Total filtered out (non-cut): {total_filtered:,}")

    total_machine = sum(item['machine_count'] for item in summary_data)
    total_wet = sum(item['wet_count'] for item in summary_data)
    total_dry = sum(item['dry_count'] for item in summary_data)
    total_unknown = sum(item['unknown_count'] for item in summary_data)
    total_cuts = total_machine + total_wet + total_dry + total_unknown

    print(f"\nCut type distribution:")
    if total_cuts > 0:
        print(f"  Machine cuts: {total_machine:,} ({100*total_machine/total_cuts:.1f}%)")
        print(f"  Wet cuts: {total_wet:,} ({100*total_wet/total_cuts:.1f}%)")
        print(f"  Dry cuts: {total_dry:,} ({100*total_dry/total_cuts:.1f}%)")
        print(f"  Unknown/ambiguous: {total_unknown:,} ({100*total_unknown/total_cuts:.1f}%)")
    else:
        print(f"  Machine cuts: {total_machine:,} (0.0%)")
        print(f"  Wet cuts: {total_wet:,} (0.0%)")
        print(f"  Dry cuts: {total_dry:,} (0.0%)")
        print(f"  Unknown/ambiguous: {total_unknown:,} (0.0%)")

    # Group by gender
    print("\n" + "=" * 80)
    print("GROUPED BY GENDER")
    print("=" * 80)
    for gender in ["male", "female", "kids", "unknown"]:
        gender_items = [item for item in summary_data if item["likely_gender"] == gender]
        if gender_items:
            print(f"\n{gender.upper()} ({len(gender_items)} IDs):")
            for item in gender_items:
                cut_type_flag = f" [{item['primary_cut_type'].upper()}]"
                print(f"  ID {item['treatmentCategoryId']}: {item['count']:,} occurrences{cut_type_flag}")
                print(f"    Machine: {item['machine_count']:,}, Wet: {item['wet_count']:,}, Dry: {item['dry_count']:,}, Unknown: {item['unknown_count']:,}")

    # IDs by primary cut type
    print("\n" + "=" * 80)
    print("GROUPED BY PRIMARY CUT TYPE")
    print("=" * 80)
    for cut_type in ["machine", "wet", "dry", "unknown"]:
        type_items = [item for item in summary_data if item["primary_cut_type"] == cut_type]
        if type_items:
            print(f"\n{cut_type.upper()} ({len(type_items)} IDs):")
            for item in type_items:
                print(f"  ID {item['treatmentCategoryId']}: {item['count']:,} total ({item['likely_gender']})")
                print(f"    Breakdown: Machine: {item['machine_count']:,}, Wet: {item['wet_count']:,}, Dry: {item['dry_count']:,}, Unknown: {item['unknown_count']:,}")

    # Flag IDs for manual review (appear with multiple cut types)
    print("\n" + "=" * 80)
    print("IDS REQUIRING MANUAL REVIEW")
    print("=" * 80)
    print("(IDs that appear with multiple cut types - machine, wet, dry)")
    print("-" * 80)

    # Check for IDs that have multiple cut types (German only)
    review_needed = []
    for item in summary_data:
        treatment_id = item['treatmentCategoryId']
        # Count how many different cut types this ID has
        cut_types_present = []
        if item['machine_count'] > 0:
            cut_types_present.append('machine')
        if item['wet_count'] > 0:
            cut_types_present.append('wet')
        if item['dry_count'] > 0:
            cut_types_present.append('dry')
        if item['unknown_count'] > 0:
            cut_types_present.append('unknown')

        if len(cut_types_present) > 1:
            review_needed.append({
                "id": treatment_id,
                "cut_types": cut_types_present,
                "machine_count": item['machine_count'],
                "wet_count": item['wet_count'],
                "dry_count": item['dry_count'],
                "unknown_count": item['unknown_count'],
                "machine_samples": item['german_machine_names'][:3],
                "wet_samples": item['german_wet_names'][:3],
                "dry_samples": item['german_dry_names'][:3],
            })

    if review_needed:
        for item in review_needed:
            print(f"\n  ID {item['id']}:")
            print(f"    Cut types present: {', '.join(item['cut_types'])}")
            print(f"    Counts: Machine={item['machine_count']:,}, Wet={item['wet_count']:,}, Dry={item['dry_count']:,}, Unknown={item['unknown_count']:,}")
            if item['machine_samples']:
                print(f"    Machine samples: {', '.join(item['machine_samples'])}")
            if item['wet_samples']:
                print(f"    Wet samples: {', '.join(item['wet_samples'])}")
            if item['dry_samples']:
                print(f"    Dry samples: {', '.join(item['dry_samples'])}")
    else:
        print("  None found - all IDs have a single cut type.")

    # Save to CSV (flatten sample_names lists - only German names)
    output_path = DATA_DIR / "derived" / "treatment_ids_analysis.csv"
    csv_data = []
    for item in summary_data:
        csv_data.append({
            "treatmentCategoryId": item["treatmentCategoryId"],
            "count": item["count"],
            "filtered_count": item["filtered_count"],
            "machine_count": item["machine_count"],
            "wet_count": item["wet_count"],
            "dry_count": item["dry_count"],
            "unknown_count": item["unknown_count"],
            "primary_cut_type": item["primary_cut_type"],
            "german_machine_names": "; ".join(item["german_machine_names"]),
            "german_wet_names": "; ".join(item["german_wet_names"]),
            "german_dry_names": "; ".join(item["german_dry_names"]),
            "german_unknown_names": "; ".join(item["german_unknown_names"]),
            "likely_gender": item["likely_gender"],
            "appears_with_male": item["appears_with_male"],
            "appears_with_female": item["appears_with_female"],
        })
    df_output = pd.DataFrame(csv_data)
    df_output.to_csv(output_path, index=False)
    print(f"\n" + "=" * 80)
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    return summary_data, treatment_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze treatmentCategoryIds from Treatwell dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_treatment_ids.py
  python analyze_treatment_ids.py treatwell-all-2025-06-02.csv
  python analyze_treatment_ids.py --min-count 500
        """
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DATA_DIR / "snapshots" / "treatwell-all-2025-06-02.csv"),
        help="Path to CSV file (default: data/snapshots/treatwell-all-2025-06-02.csv)"
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1000,
        help="Minimum number of occurrences to include an ID (default: 1000)"
    )

    args = parser.parse_args()

    try:
        analyze_treatment_ids(args.csv_path, min_count=args.min_count)
    except FileNotFoundError:
        print(f"Error: File '{args.csv_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
