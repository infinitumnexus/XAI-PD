import pandas as pd
import re
import tldextract
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -------------------------
# 1. Load dataset
# -------------------------
DATASET = "phishing_combined.csv"

print(f"Loading {DATASET} ...")
df = pd.read_csv(DATASET)

# Expect columns: url, label
if "url" not in df.columns or "label" not in df.columns:
    raise ValueError("CSV must have columns: url,label")

print(f"Dataset size: {len(df)} rows")
print(df.head())

# -------------------------
# 2. Feature extraction
# -------------------------
def extract_features(url):
    try:
        length = len(url)
        num_dots = url.count(".")
        num_digits = sum(c.isdigit() for c in url)
        num_special = len(re.findall(r"[^A-Za-z0-9]", url))
        has_https = 1 if url.startswith("https") else 0
        # Extract domain
        ext = tldextract.extract(url)
        domain = ext.domain if ext.domain else ""
        subdomain_len = len(ext.subdomain)
        path_len = len(ext.suffix) + len(ext.domain)
        return [length, num_dots, num_digits, num_special, has_https, subdomain_len, path_len]
    except Exception:
        return [0,0,0,0,0,0,0]

df_features = df["url"].apply(extract_features)
X = pd.DataFrame(df_features.tolist(), columns=[
    "length", "dots", "digits", "special_chars", "https", "subdomain_len", "path_len"
])
y = df["label"]

print("\nSample extracted features:")
print(X.head())

# -------------------------
# 3. Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------
# 4. Train model
# -------------------------
print("\nTraining RandomForest...")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# 5. Explainability with SHAP
# -------------------------
print("\nGenerating SHAP values...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Plot feature importance (global)
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.savefig("shap_summary.png")
print("Saved SHAP summary plot to shap_summary.png")

# -------------------------
# 6. Explainability with LIME
# -------------------------
print("\nGenerating LIME explanation for one sample...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=["benign","phish"],
    discretize_continuous=True
)

i = 0  # explain first test sample
exp = lime_explainer.explain_instance(
    X_test.iloc[i].values,
    clf.predict_proba,
    num_features=5
)

exp.save_to_file("lime_explanation.html")
print("Saved LIME explanation for one sample to lime_explanation.html")
