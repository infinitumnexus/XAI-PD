# app.py (with debug printing + logging)
from flask import Flask, request, jsonify, render_template, send_file
import joblib, os, re, math, json, time, pathlib
import numpy as np
from urllib.parse import urlparse
import tldextract
import shap
from lime import lime_tabular
from datetime import datetime

app = Flask(__name__)

# --- Load artifacts ---
MODEL_PATH = "results/rf_best.joblib"
SCALER_PATH = "results/scaler.joblib"
FEATURE_COLS_PATH = "results/feature_cols.joblib"

MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
FEATURE_COLS = joblib.load(FEATURE_COLS_PATH) if os.path.exists(FEATURE_COLS_PATH) else [
    "url_len","host_len","path_len","num_dots","num_hyphen","num_underscore","num_digits",
    "has_https","has_at","has_query","subdomain_len","tld_len","path_depth","entropy",
    "keyword_count","has_ip","domain_age_days","is_long_host"
]

FEATURE_LOG_PATH = pathlib.Path("results/feature_logs.jsonl")
FEATURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def shannon_entropy(s):
    prob = [float(s.count(c)) / len(s) for c in set(s)] if s else []
    return -sum([p * math.log(p, 2) for p in prob]) if prob else 0

def extract_features(url):
    """Compute all features expected by the model (best-effort)."""
    if not url.startswith("http"):
        url = "http://" + url
    parsed = urlparse(url)
    ext = tldextract.extract(parsed.netloc)

    # WHOIS omitted here for speed/stability in real-time; set 0 by default
    domain_age_days = 0

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
        "entropy": shannon_entropy(parsed.netloc),
        "keyword_count": sum(url.lower().count(k) for k in ["login","secure","account","update","bank","verify","confirm","signin","pay"]),
        "has_ip": int(bool(re.match(r"^\d+\.\d+\.\d+\.\d+$", parsed.netloc))),
        "domain_age_days": domain_age_days,
        "is_long_host": int(len(parsed.netloc) > 50),
    }

    # Vector in FEATURE_COLS order
    vec = np.array([feats.get(c, 0) for c in FEATURE_COLS], dtype=float)
    return vec, feats

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/shap.png")
def shap_png():
    path = "results/shap_summary.png"
    return send_file(path, mimetype="image/png") if os.path.exists(path) else ("No SHAP image", 404)

# Primary API route (keeps backward compatibility)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    return _predict_impl(request)

# Alias route matching frontend
@app.route("/predict", methods=["POST"])
def predict():
    return _predict_impl(request)

def _log_feature_vector(url, feats, vec_raw, vec_scaled=None):
    entry = {
        "ts": time.time(),
        "url": url,
        "features": {k: float(feats.get(k, 0)) for k in FEATURE_COLS},
        "vector_raw": [float(x) for x in vec_raw.flatten().tolist()],
        "vector_scaled": [float(x) for x in vec_scaled.flatten().tolist()] if vec_scaled is not None else None
    }
    try:
        with FEATURE_LOG_PATH.open("a", encoding="utf8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print("[FEATURE LOG WRITE ERROR]", e)

def _print_feature_vector(feats, vec_raw, vec_scaled=None):
    print("----- FEATURE VECTOR (raw) -----")
    for name, val in zip(FEATURE_COLS, vec_raw.flatten().tolist()):
        print(f"{name}: {val}")
    print("vector shape:", vec_raw.shape)
    if vec_scaled is not None:
        print("----- FEATURE VECTOR (scaled) -----")
        for name, val in zip(FEATURE_COLS, vec_scaled.flatten().tolist()):
            print(f"{name}: {val}")

def _predict_impl(req):
    try:
        data = req.get_json(force=True) if req.is_json else req.form.to_dict()
        url = data.get("url","").strip()
    except Exception:
        return jsonify({"error":"Unable to parse request"}), 400

    if not url:
        return jsonify({"error":"No URL provided"}), 400

    # build features & vector
    Xvec, feats = extract_features(url)
    Xvec2 = Xvec.reshape(1, -1)

    # Debug: print & log the raw vector before scaling
    vec_scaled = None
    try:
        if SCALER is not None:
            vec_scaled = SCALER.transform(Xvec2)
    except Exception as e:
        print("[DEBUG] scaler.transform failed:", e)

    # Print to console
    _print_feature_vector(feats, Xvec2, vec_scaled)

    # Append to results/feature_logs.jsonl
    try:
        _log_feature_vector(url, feats, Xvec2, vec_scaled)
    except Exception as e:
        print("[DEBUG] failed to write feature log:", e)

    # Use scaled vector if available
    X_in = vec_scaled if vec_scaled is not None else Xvec2

    # prediction
    try:
        pred = int(MODEL.predict(X_in)[0])
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    prob = None
    try:
        if hasattr(MODEL, "predict_proba"):
            # robustly find index of class '1' if possible
            probs = MODEL.predict_proba(X_in)[0]
            try:
                idx_phish = list(MODEL.classes_).index(1)
                prob = float(probs[idx_phish])
            except ValueError:
                # fallback: if classes_ doesn't contain 1, use second column if exists
                prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    except Exception:
        prob = None

    # SHAP top features (best-effort; keep lightweight)
    shap_info = []
    try:
        explainer = shap.TreeExplainer(MODEL)
        shap_vals = explainer.shap_values(X_in)
        arr = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        absvals = np.abs(arr).mean(axis=0)
        idxs = np.argsort(-absvals)[:6]
        for i in idxs:
            shap_info.append({"feature": FEATURE_COLS[i], "value": float(arr[0,i])})
    except Exception as e:
        print("[DEBUG] SHAP failed:", e)
        shap_info = []

    # LIME explanation (best-effort)
    lime_info = []
    try:
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.zeros((1,len(FEATURE_COLS))),
            feature_names=FEATURE_COLS,
            class_names=["benign","phish"],
            discretize_continuous=True
        )
        exp = explainer.explain_instance(Xvec.flatten(), MODEL.predict_proba, num_features=6)
        for feat, weight in exp.as_list():
            name = feat.split()[0]
            lime_info.append({"feature": name, "weight": weight})
    except Exception as e:
        print("[DEBUG] LIME failed:", e)
        lime_info = []

    print(f"[PREDICT] url={url} pred={pred} prob={prob}")
    return jsonify({
        "url": url,
        "prediction": pred,
        "probability": prob,
        "features": feats,
        "shap": shap_info,
        "lime": lime_info
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
