# debug_model.py
import joblib, numpy as np
from urllib.parse import urlparse
import tldextract
import json
import os

# load artifacts
MODEL = joblib.load("results/rf_best.joblib")
SCALER = joblib.load("results/scaler.joblib") if os.path.exists("results/scaler.joblib") else None
FEATURE_COLS = joblib.load("results/feature_cols.joblib")

# your extract_features function (copy the same one used by app.py)
def extract_features(url):
    if not url.startswith("http"):
        url = "http://" + url
    parsed = urlparse(url)
    ext = tldextract.extract(parsed.netloc)
    feats = {
        "url_len": len(url),
        "host_len": len(parsed.netloc),
        "path_len": len(parsed.path),
        "num_dots": url.count("."),
        "num_hyphen": url.count("-"),
        "num_underscore": url.count("_"),
        "num_digits": sum(c.isdigit() for c in url),
        "has_https": int(parsed.scheme == "https"),
        "has_at": int("@" in url),
        "has_query": int("?" in url),
        "subdomain_len": len(ext.subdomain),
        "tld_len": len(ext.suffix),
        "path_depth": parsed.path.count("/"),
        "entropy": 0.0,            # keep simple if not computed
        "keyword_count": 0,
        "has_ip": 0,
        "domain_age_days": 0,
        "is_long_host": int(len(parsed.netloc) > 50),
    }
    X = np.array([feats.get(c, 0) for c in FEATURE_COLS], dtype=float).reshape(1, -1)
    return X, feats

# list some example URLs (include ones you know are phishing)
urls = [
    "http://google.com",
    "http://phishy-login.com/login",    # your earlier example
    # add a phishing URL from your OpenPhish file for test
]

print("MODEL classes:", getattr(MODEL, "classes_", "N/A"))
for u in urls:
    X, feats = extract_features(u)
    X_in = SCALER.transform(X) if SCALER is not None else X
    pred = MODEL.predict(X_in)
    try:
        prob = MODEL.predict_proba(X_in)[:,1]
    except Exception:
        prob = None
    print("\nURL:", u)
    print("features (ordered):", [ (c, float(feats.get(c,0))) for c in FEATURE_COLS ])
    print("vector:", X_in.tolist())
    print("predict:", pred, " prob(>class1):", prob)
