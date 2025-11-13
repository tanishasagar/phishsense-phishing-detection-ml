# phishsense-phishing-detection-ml
Intelligent phishing website detection system built using machine learning techniques for URL and feature-based classification.
# 🛡️ PhishSense: Phishing Detection using Machine Learning

**PhishSense** is an intelligent phishing website detection system built using machine learning techniques for URL and feature-based classification.  
It analyzes patterns in URLs and website attributes to accurately distinguish between legitimate and phishing websites.

---

## 📘 Overview

Phishing attacks continue to be one of the most common and effective cyber threats.  
This project leverages machine learning algorithms to detect phishing websites automatically by learning from URL patterns, domain information, and other extracted features.

The goal of **PhishSense** is to help improve web safety by providing a reliable, data-driven detection mechanism.

---

## 🧠 Project Objectives

- Identify key characteristics that differentiate phishing from legitimate websites.  
- Build machine learning models capable of classifying websites with high accuracy.  
- Evaluate model performance across multiple feature sets and algorithms.  
- Provide a modular codebase that allows further improvement or API deployment.

---

## 📚 Data Collection

This project uses publicly available datasets from trusted security sources:

- **PhishTank:**  
  Verified phishing URLs contributed by the security community.  
  (Used as the primary phishing label source)

- **Kaggle Phishing URL Dataset:**  
  Contains a large collection of phishing and legitimate URLs along with extracted features.


**Key details:**
- **Source:**  Phishtank and Kaggle
- **Format:** CSV  
- **Columns include:**  
  - `URL`
  - `Having_IP_Address`
  - `URL_Length`
  - `Having_At_Symbol`
  - `Prefix_Suffix`
  - `HTTPS_Token`
  - `Shortening_Service`
  - `Label` (1 = Phishing, 0 = Legitimate)

> Note: Full datasets are included in the `/data/` directory for experimentation and training.

---

## ⚙️ Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3.11 |
| **Libraries** | pandas, numpy, scikit-learn, joblib, matplotlib |
| **Development** | Powershell Terminal, Notepad |
| **Version Control** | Git & GitHub |
| **CI/CD** | GitHub Actions |

---

## 🚀 Getting Started


## 🧩 Repository Structure
```

phishsense-phishing-detection-ml/
│
├─ data/
│  ├─ raw/
│  │  ├─ all_urls.xlsx
│  │  ├─ Benign_list_big_final.xlsx
│  │  └─ phishtank.xlsx
│  │
│  └─ processed/
│     ├─ all_combined.xlsx
│     ├─ all_features.xlsx
│     └─ sample_features.xlsx
│
├─ models/
│  ├─ baseline_model.joblib
│  ├─ feature_names.json
│  └─ phishsense_pipeline.joblib
│
├─ notebooks/
│  └─ (your .ipynb EDA/notebook files)
│
├─ src/
│  ├─ __pycache__/            # compiled bytecode (should be gitignored)
│  ├─ api.py
│  ├─ feature_extractor.py
│  └─ train.py
│
├─ logs/
│  └─ predictions.txt
│
├─ phish-sense/               
│  ├─ data/
│  ├─ models/
│  └─ src/
│
├─ venv/                      
│  ├─ Include/
│  ├─ Lib/
│  ├─ Scripts/
│  └─ share/
│
├─ add_labels.py
├─ blacklist.txt
├─ data_ingest.py
├─ merge_datasets.py
├─ quick_retrain.py
├─ requirements.txt
├─ retrain_phishsense.py
├─ sample_and_merge.py
├─ test_load.py
├─ train_run_log.txt
├─ whitelist.txt
├─ .gitignore
└─ README.md


