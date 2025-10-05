# phishing_research.py
"""
Research-level phishing detection + XAI pipeline (Windows-ready).
Place phishing_combined.csv in the same folder and run:
    python phishing_research.py
Outputs saved under ./results/
"""

import os
import re
import math
import joblib
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tldextract
from urllib.parse import urlparse

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

# XAI & utilities
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
from scipy.stats import entropy

warnings.filterwarnings("ignore")
plt.rcParams.update({'figure.max_open_warning': 0})

# ----------------------
# Configuration
# ----------------------
DATAFILE = "phishing_combined.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ----------------------
# Utilities / Features
# ----------------------
KEYWORD_LIST = [
    "login", "verify", "account", "secure", "update", "bank", "confirm", "password",
    "invoice", "payment", "billing", "paypal", "apple", "microsoft", "free", "offer"
]

IP_REGEX = re.compile(r'^(?:http[s]?://)?\d{1,3}(?:\.\d{1,3}){3}')

def has_ip(url):
    return int(bool(IP_REGEX.search(url)))

def token_entropy(s):
    if not s:
        return 0.0
    # character-level entropy
    probs = np.array(list(map(lambda x: s.count(x), set(s))), dtype=float)
    probs /= probs.sum()
    return float(-(probs * np.log2(probs)).sum())

def count_suspicious_keywords(url):
    u = url.lower()
    return sum(1 for k in KEYWORD_LIST if k in u)

def path_depth(parsed):
    p = parsed.path if parsed and hasattr(parsed, 'path') else ""
    if not p or p == "/":
        return 0
    return len([seg for seg in p.split("/") if seg])

def extract_domain_age(domain):
    """
    Try to get domain creation year via python-whois, but fail-safe.
    Returns age in days (float) or np.nan.
    """
    try:
        import whois
    except Exception:
        return np.nan
    try:
        w = whois.whois(domain)
        # whois creation_date can be list
        c = w.creation_date
        if isinstance(c, list) and len(c):
            c = c[0]
        if c is None:
            return np.nan
        if isinstance(c, str):
            # try parse
            c = pd.to_datetime(c, errors='coerce')
        delta = (pd.Timestamp.now(tz=None) - pd.to_datetime(c)).days
        return float(delta) if not pd.isna(delta) else np.nan
    except Exception:
        return np.nan

def extract_features_from_url(url):
    """
    Returns a dict of features for a single URL string.
    """
    url = str(url).strip()
    try:
        parsed = urlparse(url if url.startswith("http") else ("http://" + url))
    except Exception:
        parsed = urlparse("http://" + url)
    ext = tldextract.extract(parsed.netloc)
    domain = ext.domain or ""
    subdomain = ext.subdomain or ""
    suffix = ext.suffix or ""
    features = {}
    features['url_len'] = len(url)
    features['host_len'] = len(parsed.netloc)
    features['path_len'] = len(parsed.path)
    features['num_dots'] = url.count('.')
    features['num_hyphen'] = url.count('-')
    features['num_underscore'] = url.count('_')
    features['num_digits'] = sum(1 for c in url if c.isdigit())
    features['has_https'] = int(parsed.scheme == 'https')
    features['has_at'] = int('@' in url)
    features['has_query'] = int(parsed.query != "")
    features['subdomain_len'] = len(subdomain)
    features['tld_len'] = len(suffix)
    features['path_depth'] = path_depth(parsed)
    features['entropy'] = token_entropy(url)
    features['keyword_count'] = count_suspicious_keywords(url)
    features['has_ip'] = has_ip(url)
    # optional domain age (days)
    domain_full = parsed.netloc.lower().strip(":/")
    features['domain_age_days'] = extract_domain_age(domain_full)
    features['is_long_host'] = int(len(parsed.netloc) > 25)
    return features

# ----------------------
# Load & prepare data
# ----------------------
print("Loading dataset:", DATAFILE)
if not Path(DATAFILE).exists():
    raise FileNotFoundError(f"{DATAFILE} not found in current folder. Put it here and re-run.")

df = pd.read_csv(DATAFILE, dtype=str)
# Drop rows with missing URL or label, coerce label to int
df = df.dropna(subset=['url'])
if 'label' not in df.columns:
    raise ValueError("CSV must contain a 'label' column with 0 (benign) or 1 (phish).")
df = df.dropna(subset=['label']).reset_index(drop=True)
df['label'] = df['label'].astype(int)

print("Initial rows:", len(df))
print("Class distribution:\n", df['label'].value_counts())

# Extract features (this can take time if domain_age is enabled)
print("Extracting features (this can take ~seconds per row if WHOIS is used)...")
feat_rows = []
t0 = time.time()
for i, u in enumerate(df['url'].astype(str).tolist()):
    feats = extract_features_from_url(u)
    feats['url'] = u
    feats['label'] = int(df.loc[i,'label'])
    feat_rows.append(feats)
    # lightweight progress message
    if (i+1) % 50 == 0:
        print(f"  processed {i+1} rows...")
t1 = time.time()
print(f"Feature extraction done in {t1-t0:.1f}s. Produced {len(feat_rows)} feature rows.")

feat_df = pd.DataFrame(feat_rows).reset_index(drop=True)

# Some features may be NaN (domain_age). It's OK; we'll impute with median.
impute_cols = ['domain_age_days']
for c in impute_cols:
    if c in feat_df.columns:
        med = feat_df[c].median()
        feat_df[c] = feat_df[c].fillna(med)

# Final feature matrix X and label y
feature_cols = [c for c in feat_df.columns if c not in ('url','label')]
X = feat_df[feature_cols].astype(float)
y = feat_df['label'].astype(int)

print("\nFeature columns:", feature_cols)
print("Example features:\n", X.head())

# ----------------------
# Standardize continuous features (for LR)
# ----------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Save scaler for reproducibility
joblib.dump(scaler, RESULTS_DIR / "scaler.joblib")

# ----------------------
# Cross-validated evaluation (Stratified K-Fold)
# ----------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
}
scoring = ['accuracy', 'precision', 'recall', 'f1']

cv_results = {}
for name, model in models.items():
    print(f"\nRunning Stratified CV for {name} ...")
    if name == "LogisticRegression":
        res = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring, return_train_score=False)
    else:
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    summary = {k: (np.mean(v), np.std(v)) for k,v in res.items() if k.startswith('test_')}
    cv_results[name] = summary
    print(f"{name} CV results (mean ± std):")
    for metric, (mean, std) in summary.items():
        print(f"  {metric.replace('test_','')}: {mean:.3f} ± {std:.3f}")

# ----------------------
# Train-test split for final model + hyperparam search for RF
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

print("\nTrain/test sizes:", X_train.shape, X_test.shape)
# Grid search for RF (small grid)
rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
param_grid = {"n_estimators":[100,200], "max_depth":[5,10,None]}
gs = GridSearchCV(rf, param_grid, scoring='f1', cv=3, n_jobs=-1)
gs.fit(X_train, y_train)
best_rf = gs.best_estimator_
print("Best RF params:", gs.best_params_)
joblib.dump(gs, RESULTS_DIR / "rf_gridsearch.joblib")
joblib.dump(best_rf, RESULTS_DIR / "rf_best.joblib")

# Fit LR baseline
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
lr.fit(scaler.transform(X_train), y_train)  # LR on scaled features
joblib.dump(lr, RESULTS_DIR / "logistic.joblib")

# Evaluate on test set
yhat_rf = best_rf.predict(X_test)
yhat_lr = lr.predict(scaler.transform(X_test))

def print_eval(y_true, y_pred, tag="model"):
    print(f"\nEvaluation for {tag}:")
    print(classification_report(y_true, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_true, best_rf.predict_proba(X_test)[:,1]))
    except Exception:
        pass

print_eval(y_test, yhat_rf, "RandomForest")
print_eval(y_test, yhat_lr, "LogisticRegression")

# ----------------------
# Explainability: SHAP for RF & LIME for LR
# ----------------------
print("\nComputing SHAP (TreeExplainer) for RandomForest (may take seconds)...")
explainer = shap.TreeExplainer(best_rf)
# explain a sample subset if large
X_shap = X_test if X_test.shape[0] <= 500 else X_test.sample(500, random_state=RANDOM_SEED)
shap_values = explainer.shap_values(X_shap)
shap.summary_plot(shap_values, X_shap, feature_names=X.columns, show=False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=200)
plt.close()
print("Saved SHAP summary at:", RESULTS_DIR / "shap_summary.png")

# LIME for logistic regression (local explanation)
print("\nGenerating one LIME explanation for LR on a random test sample...")
lime_exp = None
try:
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=scaler.transform(X_train),
        feature_names=list(X.columns),
        class_names=["benign","phish"],
        discretize_continuous=True
    )
    idx = random.randrange(0, X_test.shape[0])
    exp = lime_explainer.explain_instance(scaler.transform(X_test.iloc[idx:idx+1])[0],
                                          lr.predict_proba, num_features=8)
    exp.save_to_file(str(RESULTS_DIR / "lime_lr_sample.html"))
    print("Saved LIME HTML to:", RESULTS_DIR / "lime_lr_sample.html")
    lime_exp = exp
except Exception as e:
    print("LIME failed:", e)

# ----------------------
# Explanation metrics
# ----------------------
print("\nComputing surrogate fidelity (Decision Tree surrogate for RF predictions)...")
surrogate = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
surrogate.fit(X_train, best_rf.predict(X_train))
pred_sur = surrogate.predict(X_test)
fidelity = (pred_sur == best_rf.predict(X_test)).mean()
print(f"Surrogate fidelity (agreement on test set): {fidelity:.3f}")
joblib.dump(surrogate, RESULTS_DIR / "surrogate_tree.joblib")

# Stability: perturb each test URL by removing a random character and recomputing feature importances ranking (simple)
def feature_ranking_by_absshap(shap_vals, feature_names):
    """Return features sorted by mean absolute SHAP value."""
    mean_abs = np.mean(np.abs(shap_vals), axis=1).mean(axis=0) if isinstance(shap_vals, list) else np.mean(np.abs(shap_vals), axis=0)
    # shap for binary may be list -> handle both
    if isinstance(mean_abs, np.ndarray):
        idxs = np.argsort(-mean_abs)
    else:
        idxs = np.argsort(-np.array(mean_abs))
    return [feature_names[i] for i in idxs]

try:
    base_rank = feature_ranking_by_absshap(shap_values, list(X.columns))
    # a simple perturbation: shuffle url string for one test sample and recompute features+shap for that one sample
    pert_idx = 0
    sample_url = feat_df.loc[X_test.index[pert_idx], "url"]
    pert_url = sample_url.replace("/", "", 1) if "/" in sample_url else sample_url + "x"
    pert_feats = pd.Series(extract_features_from_url(pert_url))[feature_cols].astype(float).values.reshape(1, -1)
    # recompute shap for single item approximated by explainer (fast path)
    pert_shap_vals = explainer.shap_values(pd.DataFrame(pert_feats, columns=X.columns))
    pert_rank = feature_ranking_by_absshap(pert_shap_vals, list(X.columns))
    # compute top-5 overlap
    overlap = len(set(base_rank[:5]).intersection(set(pert_rank[:5])))
    stability_score = overlap / 5.0
    print(f"Stability (top-5 feature overlap under simple perturbation): {stability_score:.3f}")
except Exception as e:
    print("Stability check skipped (error):", e)

# ----------------------
# Save metadata & artifacts
# ----------------------
meta = {
    "timestamp": datetime.utcnow().isoformat(),
    "n_rows": int(len(df)),
    "class_counts": df['label'].value_counts().to_dict(),
    "feature_cols": feature_cols,
    "rf_best_params": gs.best_params_,
    "surrogate_fidelity": float(fidelity)
}
with open(RESULTS_DIR / "meta.json", "w", encoding="utf8") as f:
    json.dump(meta, f, indent=2)

joblib.dump(feature_cols, RESULTS_DIR / "feature_cols.joblib")
joblib.dump(X.columns.tolist(), RESULTS_DIR / "feature_order.joblib")
print("\nAll artifacts saved to", RESULTS_DIR.resolve())

print("\nDone.")
