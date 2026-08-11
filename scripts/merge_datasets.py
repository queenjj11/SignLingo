import pandas as pd
import os

files = {
    "ASL": "data/processed/asl_letters.csv",
    "ISL": "data/processed/isl_letters.csv",
    "BSL": "data/processed/bsl_letters.csv",
}

dfs = []

for language, path in files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["language"] = language
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)

os.makedirs("data/processed", exist_ok=True)

merged.to_csv(
    "data/processed/multilingual_letters.csv",
    index=False
)

print("===================================")
print("Merged Successfully!")
print("===================================")
print(f"Total samples : {len(merged)}")
print(f"Languages     : {merged['language'].unique()}")
print(f"Classes       : {merged['label'].nunique()}")
print("Saved to      : data/processed/multilingual_letters.csv")