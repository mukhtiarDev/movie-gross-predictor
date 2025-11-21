import pandas as pd

# Load your CSV
df = pd.read_csv("movies.csv")

# 1. --- Check for duplicate rows ---
duplicates = df[df.duplicated()]
print("Duplicate rows:")
print(duplicates)


# 2. --- Check for missing values in each column ---
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing)


# 3. --- Check for invalid date format in 'released' column ---
# Your format looks like: "June 13, 1980 (United States)"
# This is NOT a standard YYYY-MM-DD, so we'll try parsing it.
df["parsed_released"] = pd.to_datetime(df["released"], errors="coerce")

invalid_dates = df[df["parsed_released"].isna() & df["released"].notna()]
print("\nRows with invalid date format in 'released':")
print(invalid_dates)

