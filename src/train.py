# src/train.py
import glob, os, joblib, pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def load_features(pattern="data/processed/*.csv"):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No CSVs found in data/processed/. Place feature CSV(s) there.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def prepare_xy(df):
    if 'label' not in df.columns:
        raise ValueError("Processed CSV must contain a 'label' column (1=phish,0=legit).")
    X = df.drop(columns=[c for c in ['url','domain','label'] if c in df.columns])
    X = X.fillna(-1)
    # keep numeric features only for model
    X = X.select_dtypes(include=[np.number])
    y = df['label'].values
    return X, y

def train_models(X_train, X_test, y_train, y_test):
    print("Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    proba_rf = rf.predict_proba(X_test)[:,1]
    print("\nRandomForest results:")
    print(classification_report(y_test, preds_rf, digits=4))
    print("ROC-AUC RF:", roc_auc_score(y_test, proba_rf))

    print("\nTraining XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                        n_estimators=200, random_state=42, n_jobs=4)
    xgb.fit(X_train, y_train)
    preds_xgb = xgb.predict(X_test)
    proba_xgb = xgb.predict_proba(X_test)[:,1]
    print("\nXGBoost results:")
    print(classification_report(y_test, preds_xgb, digits=4))
    print("ROC-AUC XGB:", roc_auc_score(y_test, proba_xgb))

    auc_rf = roc_auc_score(y_test, proba_rf)
    auc_xgb = roc_auc_score(y_test, proba_xgb)
    best = xgb if auc_xgb >= auc_rf else rf
    os.makedirs("models", exist_ok=True)
    joblib.dump(best, "models/baseline_model.joblib")

    # Save feature names (important for API explainability)
    feat_names = X_train.columns.tolist()
    with open("models/feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feat_names, f)
    print("\n✅ Saved best model to models/baseline_model.joblib")
    print("✅ Saved feature names to models/feature_names.json")

if __name__ == "__main__":
    df = load_features()
    X, y = prepare_xy(df)

    # adaptive train/test split to handle very small sample sizes
    n_samples = len(y)
    n_classes = len(np.unique(y))
    desired_test_frac = 0.2
    min_test_needed = n_classes
    test_count = int(np.ceil(desired_test_frac * n_samples))
    if test_count < min_test_needed:
        new_test_frac = min(0.5, max(desired_test_frac, min_test_needed / max(1, n_samples)))
        print(f"[info] small dataset ({n_samples} rows, {n_classes} classes). Using test_size={new_test_frac:.2f}.")
        test_size = new_test_frac
    else:
        test_size = desired_test_frac

    stratify_var = y if int(np.ceil(test_size * n_samples)) >= n_classes else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_var, random_state=42
    )

    train_models(X_train, X_test, y_train, y_test)
