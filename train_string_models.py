# train_string_models.py (compact)
import pandas as pd, numpy as np
from urllib.parse import urlparse
import tldextract, re, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

def extract_features(url):
    u = str(url).strip()
    if not u.startswith("http"):
        u = "http://"+u
    parsed = urlparse(u)
    ext = tldextract.extract(parsed.netloc)
    return {
        "url_len": len(u),
        "host_len": len(parsed.netloc),
        "path_len": len(parsed.path),
        "num_dots": u.count("."),
        "num_hyphen": u.count("-"),
        "num_digits": sum(c.isdigit() for c in u),
        "has_https": int(parsed.scheme=="https"),
        "subdomain_len": len(ext.subdomain),
    }

df = pd.read_csv("phishing_combined.csv", dtype=str)
df = df.dropna(subset=["url","label"])
df["label"] = df["label"].astype(int)
X = pd.DataFrame([extract_features(u) for u in df["url"]])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# XGBoost
xclf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200)
xclf.fit(X_train, y_train)
joblib.dump(xclf, "results/xgb_model.joblib")
yhat = xclf.predict(X_test)
print("XGBoost\n", classification_report(y_test, yhat))
try:
    print("AUC:", roc_auc_score(y_test, xclf.predict_proba(X_test)[:,1]))
except: pass

# LightGBM
lclf = lgb.LGBMClassifier(n_estimators=200)
lclf.fit(X_train, y_train)
joblib.dump(lclf, "results/lgb_model.joblib")
yhat2 = lclf.predict(X_test)
print("LightGBM\n", classification_report(y_test, yhat2))
try:
    print("AUC:", roc_auc_score(y_test, lclf.predict_proba(X_test)[:,1]))
except: pass

# SHAP for XGBoost
explainer = shap.TreeExplainer(xclf)
sample = X_test if X_test.shape[0]<=500 else X_test.sample(500, random_state=42)
shap_vals = explainer.shap_values(sample)
shap.summary_plot(shap_vals, sample, feature_names=sample.columns, show=False)
plt.tight_layout()
plt.savefig("results/shap_string_summary.png", dpi=200)
print("Saved SHAP to results/shap_string_summary.png")
