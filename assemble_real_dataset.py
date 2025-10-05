# assemble_real_dataset.py
import pandas as pd
import os
os.makedirs("data", exist_ok=True)

# load OpenPhish if available
openphish_path = "data/openphish.csv"
if not pd.io.common.file_exists(openphish_path):
    raise SystemExit("Run collect_openphish.py first to fetch data/openphish.csv")

phish = pd.read_csv(openphish_path, dtype=str)
phish = phish.dropna(subset=["url"]).drop_duplicates().reset_index(drop=True)
phish["label"] = 1

# benign
benign = pd.read_csv("benign_urls.csv", dtype=str)
benign = benign.dropna(subset=["url"]).drop_duplicates().reset_index(drop=True)

# unify
combined = pd.concat([phish[["url","label"]], benign[["url","label"]]], ignore_index=True, sort=False)
# normalize minimal: strip whitespace
combined["url"] = combined["url"].astype(str).str.strip()
combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
combined.to_csv("phishing_combined.csv", index=False)
print("Saved phishing_combined.csv with", len(combined), "rows")
