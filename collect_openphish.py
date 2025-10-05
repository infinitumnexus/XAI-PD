# collect_openphish.py
import requests, pandas as pd, os
os.makedirs("data", exist_ok=True)
OUT = "data/openphish.csv"
print("Downloading OpenPhish feed...")
r = requests.get("https://openphish.com/feed.txt", timeout=30)
r.raise_for_status()
urls = [u.strip() for u in r.text.splitlines() if u.strip()]
df = pd.DataFrame({"url": urls})
df.to_csv(OUT, index=False)
print("Saved", len(df), "rows to", OUT)
