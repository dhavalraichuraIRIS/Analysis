from typing import List, Dict
import pandas as pd
import numpy as np
import unicodedata
from datetime import datetime

# Class to handle file reading operations for CSV and Excel files
class FileProcessor:
    # Initialize with the file path
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    # Read the file and return a DataFrame
    def read_file(self) -> pd.DataFrame:
        # Check if the file is a CSV and read it
        if self.file_path.endswith(".csv"):
            return pd.read_csv(self.file_path)
        # Check if the file is an Excel file and read it
        elif self.file_path.endswith((".xlsx", ".xls")):
            return pd.read_excel(self.file_path)
        # Raise an error for unsupported file formats
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")

# Function to normalize text by converting to lowercase and removing accents
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Convert text to lowercase
    text = text.lower()
    # Normalize text to remove accents
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
    # Remove any remaining non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove extra spaces
    text = " ".join(text.split())
    return text

# Process DataFrame rows into a list of dictionaries with normalized text
# Process DataFrame rows into a list of dictionaries with normalized text
def process_data_dict(df: pd.DataFrame, column_mapping: Dict[str, str]) -> List[Dict]:
    processed_data = []

    def parse_rating(val) -> int:
        if pd.isna(val) or val is None:
            return 0
        if isinstance(val, (int, float)):
            return int(val)
        val_str = str(val).strip().lower()
        if val_str.isdigit():
            return int(val_str)
        if val_str in ("pos", "positive", "true", "yes"):
            return 5
        if val_str in ("neg", "negative", "false", "no"):
            return 1
        return 0

    for _, row in df.iterrows():
        get_value = lambda val, default: (
            "" if pd.isna(val) or val is None else str(val).strip()
        )
        get_date = lambda val: (
            val.strftime("%Y-%m-%d")
            if isinstance(val, datetime)
            else get_value(val, "")
        )

        processed_row = {
            "DateOfReview": get_date(row[column_mapping["DateOfReview"]]),
            "Rating": parse_rating(row[column_mapping["Rating"]]),
            "Review": normalize_text(get_value(row[column_mapping["Review"]], "")),
            "Title": normalize_text(get_value(row[column_mapping["Title"]], "")),
        }
        processed_data.append(processed_row)

    return processed_data


# Calculate the distribution of ratings from a list of reviews
def get_rating_distribution(reviews):
    rating_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for review in reviews:
        rating = review.get("Rating", 0)
        if rating in rating_counts:
            rating_counts[rating] += 1
        else:
            rating_counts[0] += 1

    # Remove the '0' bucket if all reviews actually had valid 1-5 ratings
    if rating_counts[0] == 0:
        del rating_counts[0]

    total_reviews = sum(rating_counts.values())

    # Safe division that never throws ZeroDivisionError
    distribution_percentages = {
        rating: round((count / total_reviews * 100), 2) if total_reviews > 0 else 0.0
        for rating, count in rating_counts.items()
    }

    valid_rated_reviews = sum(count for r, count in rating_counts.items() if r > 0)
    average_rating = (
        sum(rating * count for rating, count in rating_counts.items() if rating > 0)
        / valid_rated_reviews
        if valid_rated_reviews > 0
        else 0.0
    )

    return {
        "total_reviews": total_reviews,
        "rating_counts": rating_counts,
        "rating_percentages": distribution_percentages,
        "average_rating": round(average_rating, 2),
    }