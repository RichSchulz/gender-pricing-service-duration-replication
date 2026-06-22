import requests
import time
import json
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"


def get_url(
    country: Literal["de", "ch", "at", "fr", "be", "nl", "uk", "es", "pt", "it"],
    page: int,
    is_male: bool,
) -> str:
    if country in ["de", "ch", "at"]:
        return f"https://www.treatwell.{country}/api/v1/page/browse?page={page}&currentBrowseUri=%2Forte%2Fbehandlung-{'herrenhaarschnitt' if is_male else 'damenhaarschnitt'}%2Fangebot-typ-lokal%2Fin-{country}%2F"
    elif country == "fr":
        return f"https://www.treatwell.fr/api/v2/page/browse?page={page}&currentBrowseUri=%2Fsalons%2Fsoin-coupe-{'homme' if is_male else 'femme'}%2Foffre-type-local%2Fdans-france%2F"
    elif country in ["be", "nl", "uk"]:
        return f"https://www.treatwell.{'co.uk' if country == 'uk' else country}/api/v2/page/browse?page={page}&currentBrowseUri=%2Fen%2Fplaces%2Ftreatment-{'men-s' if is_male else 'ladies'}-haircut%2Foffer-type-local%2Fin-{country}%2F"
    elif country == "es":
        return f"https://www.treatwell.es/api/v2/page/browse?page={page}&currentBrowseUri=%2Festablecimientos%2Ftratamiento-corte-de-pelo{'-hombre' if is_male else ''}%2Foferta-tipo-local%2Fen-es%2F"
    elif country == "pt":
        return f"https://www.treatwell.pt/api/v2/page/browse?page={page}&currentBrowseUri=%2Festabelecimentos%2Ftratamento-{'corte-homem' if is_male else 'corte-de-cabelo-mulher'}%2Foferta-tipolocal%2Fem-portugal%2F"
    elif country == "it":
        return f"https://www.treatwell.it/api/v2/page/browse?page={page}&currentBrowseUri=%2Fsaloni%2Ftrattamento-taglio-{'uomo' if is_male else 'donna'}%2Fofferta-tipo-locale%2Fin-italia%2F"


def get_kids_url(
    country: Literal["de", "ch", "at", "fr", "be", "nl", "uk", "es", "pt", "it"],
    page: int,
    is_boy: bool = True,
) -> str:
    if country in ["de", "ch", "at"]:
        return f"https://www.treatwell.{country}/api/v1/page/browse?page={page}&currentBrowseUri=%2Forte%2Fbehandlung-kinderhaarschnitt%2Fangebot-typ-lokal%2Fin-{country}%2F"
    elif country == "fr":
        return f"https://www.treatwell.fr/api/v2/page/browse?page={page}&currentBrowseUri=%2Fsalons%2Fsoin-coupe-enfant%2Foffre-type-local%2Fdans-france%2F"
    elif country in ["be", "nl", "uk"]:
        return f"https://www.treatwell.{'co.uk' if country == 'uk' else country}/api/v2/page/browse?treatmentCategoryIds=729&page={page}&currentBrowseUri=%2Fen%2Fplaces%2Foffer-type-local%2Fin-{country}%2F"
    elif country == "es":
        return f"https://www.treatwell.es/api/v2/page/browse?page={page}&currentBrowseUri=%2Festablecimientos%2Ftratamiento-corte-infantil%2Foferta-tipo-local%2Fen-es%2F"
    elif country == "pt":
        return f"https://www.treatwell.pt/api/v2/page/browse?treatmentCategoryIds={'766' if is_boy else '767'}&page={page}&currentBrowseUri=%2Festabelecimentos%2Foferta-tipolocal%2Fem-portugal%2F"
    elif country == "it":
        return f"https://www.treatwell.it/api/v2/page/browse?treatmentCategoryIds=729&page={page}&currentBrowseUri=%2Fsaloni%2Fofferta-tipo-locale%2Fin-italia%2Fcomprare-come-prenotazione%2F"


countries: list[Literal["de", "ch", "at", "fr", "be", "nl", "uk", "es", "pt", "it"]] = [
    "de",
    "ch",
    "at",
    "fr",
    "be",
    "nl",
    "uk",
    "es",
    "pt",
    "it",
]

headers = {
    "accept": "application/json",
    "accept-language": "en-DE,en;q=0.9",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-accept-api-version": "20250122",
    "x-language-code": "de",
}


def adults():
    all_results = {"male": [], "female": []}

    for country in countries:
        for gender in ["male", "female"]:
            current_page = 0
            while True:
                print(f"Fetching page {current_page} for {gender} in {country}")
                response = requests.get(
                    get_url(country, current_page, gender == "male"), headers=headers
                )
                current_page += 1
                if response.status_code == 200:
                    cleaned_text = response.text.lstrip(")]}',\n ")  # Remove the prefix
                    cleaned_response = json.loads(cleaned_text)

                    all_results[gender].extend(cleaned_response["results"])

                    print("Current results: ", current_page)
                    print(
                        f"Total results: {cleaned_response['pagination']['totalPages']}"
                    )
                    if cleaned_response["pagination"]["totalPages"] == current_page:
                        break
                else:
                    print(f"Failed to fetch page {current_page}")
                    print(response.status_code)
                    print(response.text)
                    break

                time.sleep(0.25)

    formatted_results = []
    for gender in ["male", "female"]:
        for result in all_results[gender]:
            # we want to keep all haircuts that we find
            haircuts_multiple = []
            for highlight in result["data"]["menuHighlights"]:
                haircuts_multiple.append(highlight)

            for haircut in haircuts_multiple:
                if any(
                    s in haircut["data"].get("treatmentCategoryIds", [])
                    for s in [716, 795, 166, 792]
                ):
                    formatted_results.append(
                        {
                            "id": result["data"]["id"],
                            "name": result["data"]["name"],
                            "averageRating": result["data"]["rating"]["average"],
                            "ratingCount": result["data"]["rating"]["count"],
                            "simpleCutSalePrice": haircut["data"][
                                "fulfilmentPriceRanges"
                            ]["booking"]["minSalePrice"]["salePriceAmount"],
                            "simpleCutFullPrice": haircut["data"][
                                "fulfilmentPriceRanges"
                            ]["booking"]["minSalePrice"]["fullPriceAmount"],
                            "simpleCutDurationMin": haircut["data"]["durationRange"][
                                "minDurationMinutes"
                            ],
                            "simpleCutDurationMax": haircut["data"]["durationRange"][
                                "maxDurationMinutes"
                            ],
                            "simpleCutName": haircut["data"]["name"],
                            "raw": json.dumps(result),
                            "postalCode": result["data"]["location"]["address"].get(
                                "postalCode"
                            ),
                            "adress": " ".join(
                                result["data"]["location"]["address"]["addressLines"]
                            ),
                            "lat": result["data"]["location"]["point"]["lat"],
                            "lon": result["data"]["location"]["point"]["lon"],
                            "is_male": 716
                            in haircut["data"].get("treatmentCategoryIds", [])
                            or 795 in haircut["data"].get("treatmentCategoryIds", []),
                            "is_female": 166
                            in haircut["data"].get("treatmentCategoryIds", [])
                            or 792 in haircut["data"].get("treatmentCategoryIds", []),
                            "type": result["data"]
                            .get("type", {})
                            .get("normalisedName", ""),
                        }
                    )

    df = pd.DataFrame(formatted_results)

    df.to_csv(
        SNAPSHOTS_DIR / f"treatwell-all-{datetime.today().strftime('%Y-%m-%d')}.csv",
        index=False,
    )
    df = df.drop("raw", axis=1)
    df.to_csv(
        SNAPSHOTS_DIR
        / f"treatwell_without_raw-all-{datetime.today().strftime('%Y-%m-%d')}.csv",
        index=False,
    )


def kids():
    all_results = []
    for country in countries:
        genders = ["boys", "girls"] if country in ["pt"] else ["boys"]

        for gender in genders:
            current_page = 0
            while True:
                print(f"Fetching page {current_page} for {gender} in {country}")
                response = requests.get(
                    get_kids_url(country, current_page, gender == "boys"),
                    headers=headers,
                )
                current_page += 1
                if response.status_code == 200:
                    cleaned_text = response.text.lstrip(")]}',\n ")  # Remove the prefix
                    cleaned_response = json.loads(cleaned_text)

                    all_results.extend(cleaned_response["results"])
                    if cleaned_response["pagination"]["totalPages"] == current_page:
                        break
                else:
                    print(f"Failed to fetch page {current_page}")
                    print(response.status_code)
                    print(response.text)
                    break

                time.sleep(0.25)

    formatted_results = []
    for result in all_results:
        for highlight in result["data"]["menuHighlights"]:
            if 729 in highlight["data"].get("treatmentCategoryIds", []):
                formatted_results.append(
                    {
                        "id": result["data"]["id"],
                        "name": result["data"]["name"],
                        "averageRating": result["data"]["rating"]["average"],
                        "ratingCount": result["data"]["rating"]["count"],
                        "simpleCutSalePrice": highlight["data"][
                            "fulfilmentPriceRanges"
                        ]["booking"]["minSalePrice"]["salePriceAmount"],
                        "simpleCutFullPrice": highlight["data"][
                            "fulfilmentPriceRanges"
                        ]["booking"]["minSalePrice"]["fullPriceAmount"],
                        "simpleCutDurationMin": highlight["data"]["durationRange"][
                            "minDurationMinutes"
                        ],
                        "simpleCutDurationMax": highlight["data"]["durationRange"][
                            "maxDurationMinutes"
                        ],
                        "simpleCutName": highlight["data"]["name"],
                        "raw": json.dumps(result),
                        "postalCode": result["data"]["location"]["address"].get(
                            "postalCode"
                        ),
                        "adress": " ".join(
                            result["data"]["location"]["address"]["addressLines"]
                        ),
                        "lat": result["data"]["location"]["point"]["lat"],
                        "lon": result["data"]["location"]["point"]["lon"],
                        "is_girls": 767
                        in highlight["data"].get("treatmentCategoryIds", [])
                        and 766
                        not in highlight["data"].get("treatmentCategoryIds", []),
                        "is_boys": 766
                        in highlight["data"].get("treatmentCategoryIds", [])
                        and 767
                        not in highlight["data"].get("treatmentCategoryIds", []),
                        "is_unisex": 766
                        in highlight["data"].get("treatmentCategoryIds", [])
                        and 767 in highlight["data"].get("treatmentCategoryIds", []),
                        "type": result["data"]
                        .get("type", {})
                        .get("normalisedName", ""),
                    }
                )
    df = pd.DataFrame(formatted_results)
    df.to_csv(
        SNAPSHOTS_DIR / f"treatwell_kids-{datetime.today().strftime('%Y-%m-%d')}.csv",
        index=False,
    )
    df = df.drop("raw", axis=1)
    df.to_csv(
        SNAPSHOTS_DIR
        / f"treatwell_without_raw_kids-{datetime.today().strftime('%Y-%m-%d')}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kids", action="store_true", help="Collect data for children's haircuts"
    )

    args = parser.parse_args()

    if args.kids:
        kids()
    else:
        adults()
