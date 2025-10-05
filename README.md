# XAI-PD
This repository contains a complete implementation of a phishing website detection system built with Python, scikit-learn, Flask, and Explainable AI (SHAP & LIME).  It includes a trained Random Forest classifier, feature extraction pipeline, interactive web interface, and XAI visualizations that explain why each URL is classified as Legitimate or Phishing.

---

Project Overview

Phishing attacks are among the most common forms of cybercrime.  
This project develops a machine learning–based phishing URL detection model and integrates it into a web application that allows users to check suspicious links in real time.

The model uses handcrafted URL-based features to predict malicious intent, while explainability modules (SHAP and LIME) help visualize which factors most influenced the prediction.

---

Features

18 Feature Extractors — analyzes URL structure (lengths, symbols, entropy, domain age, etc.)  
Random Forest Classifier — trained on a labeled phishing/legitimate dataset  
Explainable AI Integration — SHAP & LIME explanations per prediction  
Web Application (Flask) — user-friendly interface for URL testing  
Logging & Debugging — prints feature vectors and scaled values  
Model Persistence — serialized `.joblib` model, scaler, and feature column files  
Real vs Toy Modes — supports both demo (toy) and full-scale research dataset training  
Browser-ready Deployment — can be hosted locally or via any cloud service  

---

Machine Learning Architecture

1. Feature Engineering
Each URL is transformed into a numerical vector of 18 features:

| Feature | Description |
|----------|--------------|
| `url_len` | Total URL length |
| `host_len` | Length of hostname |
| `path_len` | Length of path component |
| `num_dots` | Number of `.` in URL |
| `num_hyphen` | Count of `-` |
| `num_underscore` | Count of `_` |
| `num_digits` | Number of digits |
| `has_https` | 1 if HTTPS used |
| `has_at` | 1 if `@` present |
| `has_query` | 1 if query string present |
| `subdomain_len` | Length of subdomain |
| `tld_len` | Length of TLD (.com, .org, etc.) |
| `path_depth` | Depth of URL path |
| `entropy` | Shannon entropy of domain |
| `keyword_count` | Number of suspicious words like 'login', 'verify' |
| `has_ip` | 1 if domain contains IP address |
| `domain_age_days` | Domain age in days (via WHOIS or simulated) |
| `is_long_host` | 1 if hostname is abnormally long |

2. Model
A RandomForestClassifier was trained with cross-validation and saved as `rf_best.joblib`.  
The associated `scaler.joblib` and `feature_cols.joblib` ensure consistent preprocessing.

3. Explainable AI
- SHAP (TreeExplainer) identifies per-feature impact.
- LIME approximates local decision boundaries for interpretability.

---

Getting Started

1. Clone the Repository
```bash
git clone https://github.com/infinitumnexus/XAI-PD.git
cd phishing_project




