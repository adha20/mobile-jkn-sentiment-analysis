"""
Scrape Google Play reviews for the Mobile JKN app.

Rating metadata rule:
- rating 1-2 -> negatif
- rating 3   -> netral
- rating 4-5 -> positif

Training label rule:
- review text with stronger negative lexicon -> negatif
- review text with stronger positive lexicon -> positif
- ties or no lexicon hit -> netral

The default target is 12,000 rows: 4,000 rows for each rating-derived group.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews


APP_ID = "app.bpjs.mobile"
LANG = "id"
COUNTRY = "id"
LABEL_BY_SCORE = {
    1: "negatif",
    2: "negatif",
    3: "netral",
    4: "positif",
    5: "positif",
}
POSITIVE_TERMS = [
    "bagus",
    "baik",
    "mantap",
    "mudah",
    "membantu",
    "terbaik",
    "puas",
    "cepat",
    "lancar",
    "praktis",
    "oke",
    "ok",
    "terima kasih",
    "terimakasih",
    "sip",
    "top",
    "keren",
    "bermanfaat",
    "memuaskan",
    "good",
    "nice",
    "recommended",
    "rekomendasi",
    "hebat",
    "joss",
    "jos",
]
NEGATIVE_TERMS = [
    "susah",
    "sulit",
    "error",
    "eror",
    "gagal",
    "jelek",
    "buruk",
    "parah",
    "ribet",
    "lemot",
    "lambat",
    "crash",
    "tidak bisa",
    "tdk bisa",
    "ga bisa",
    "gak bisa",
    "nggak bisa",
    "gabisa",
    "gk bisa",
    "payah",
    "kecewa",
    "tolong",
    "masalah",
    "login",
    "verifikasi",
    "captcha",
    "bug",
    "gangguan",
    "down",
    "keluar sendiri",
    "muter",
    "lama",
    "kendala",
    "komplain",
    "sampah",
    "mengecewakan",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Mobile JKN Google Play reviews into a labeled CSV dataset."
    )
    parser.add_argument(
        "--output",
        default="data/mobile_jkn_reviews.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=4000,
        help="Target number of rows for each rating-derived sentiment group.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of reviews requested per Google Play page.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Pause between requests in seconds.",
    )
    parser.add_argument(
        "--overfetch-ratio",
        type=float,
        default=1.15,
        help="Fetch extra raw reviews before deduplication and balancing.",
    )
    return parser.parse_args()


def sentiment_for_score(score: int) -> str:
    return LABEL_BY_SCORE[int(score)]


def count_terms(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def weak_sentiment_for_text(text: str) -> str:
    positive_count = count_terms(text, POSITIVE_TERMS)
    negative_count = count_terms(text, NEGATIVE_TERMS)

    if negative_count > positive_count and negative_count >= 1:
        return "negatif"
    if positive_count > negative_count and positive_count >= 1:
        return "positif"
    return "netral"


def fetch_reviews_for_score(
    score: int,
    target: int,
    batch_size: int,
    sleep: float,
) -> list[dict]:
    rows: list[dict] = []
    token = None

    while len(rows) < target:
        batch, token = reviews(
            APP_ID,
            lang=LANG,
            country=COUNTRY,
            sort=Sort.NEWEST,
            count=min(batch_size, target - len(rows)),
            filter_score_with=score,
            continuation_token=token,
        )

        if not batch:
            break

        remaining = target - len(rows)
        rows.extend(batch[:remaining])
        print(f"score={score} collected={len(rows):>5}/{target}")

        if token is None:
            break

        time.sleep(sleep)

    return rows


def normalize_rows(raw_reviews: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_reviews)

    keep_columns = [
        "reviewId",
        "content",
        "score",
        "thumbsUpCount",
        "at",
        "appVersion",
    ]
    df = df[[column for column in keep_columns if column in df.columns]].copy()
    df = df.rename(
        columns={
            "reviewId": "review_id",
            "thumbsUpCount": "thumbs_up_count",
            "appVersion": "app_version",
        }
    )

    df["content"] = df["content"].fillna("").astype(str).str.strip()
    df = df[df["content"].str.len() > 0].copy()
    df["score"] = df["score"].astype(int)
    df["rating_sentiment"] = df["score"].map(sentiment_for_score)
    df["sentiment"] = df["content"].map(weak_sentiment_for_text)
    df["source_app"] = "Mobile JKN"
    df["source_app_id"] = APP_ID
    df["scrape_country"] = COUNTRY
    df["scrape_language"] = LANG

    if "at" in df.columns:
        df["at"] = pd.to_datetime(df["at"], errors="coerce")

    # Keep distinct public reviews even when several users write the same short text.
    df = df.drop_duplicates(subset=["review_id"])
    df = df.sort_values(
        ["rating_sentiment", "at"],
        ascending=[True, False],
        na_position="last",
    )
    return df.reset_index(drop=True)


def balance_dataset(df: pd.DataFrame, target_per_class: int) -> pd.DataFrame:
    balanced_parts = []
    for label in ["negatif", "netral", "positif"]:
        label_df = df[df["rating_sentiment"] == label].copy()
        if len(label_df) < target_per_class:
            raise ValueError(
                f"Kelas rating {label} hanya memiliki {len(label_df)} data, "
                f"kurang dari target {target_per_class}."
            )
        balanced_parts.append(label_df.head(target_per_class))

    balanced = pd.concat(balanced_parts, ignore_index=True)
    return balanced.sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_class_fetch_target = int(args.target_per_class * args.overfetch_ratio)
    score_targets = {
        1: per_class_fetch_target // 2,
        2: per_class_fetch_target - (per_class_fetch_target // 2),
        3: per_class_fetch_target,
        4: per_class_fetch_target // 2,
        5: per_class_fetch_target - (per_class_fetch_target // 2),
    }

    all_rows: list[dict] = []
    for score, target in score_targets.items():
        all_rows.extend(fetch_reviews_for_score(score, target, args.batch_size, args.sleep))

    df = normalize_rows(all_rows)
    balanced = balance_dataset(df, args.target_per_class)
    balanced.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nDataset saved:", output_path.resolve())
    print("Shape:", balanced.shape)
    print("\nWeak sentiment distribution:")
    print(balanced["sentiment"].value_counts().sort_index())
    print("\nRating sentiment distribution:")
    print(balanced["rating_sentiment"].value_counts().sort_index())
    print("\nScore distribution:")
    print(balanced["score"].value_counts().sort_index())


if __name__ == "__main__":
    main()
