# retrain_phishsense.py
import pandas as pd, numpy as np, re, joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "all_combined.csv"
OUT_MODEL = ROOT / "models" / "phishsense_pipeline.joblib"

if not DATA_PATH.exists():
    raise SystemExit(f"Combined CSV not found at {DATA_PATH}. Run merge_datasets.py first.")

def canonicalize(u):
    if pd.isna(u): return ""
    s = str(u).strip()
    s = s.split('#')[0]
    s = re.sub(r'(\?|&)(utm_[^=]+|fbclid|gclid)=[^&]*', '', s)
    s = re.sub(r'\?&|&&', '?', s)
    return s.lower()

class URLNumericTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        out = []
        for u in X:
            s = str(u)
            length = len(s)
            digits = sum(ch.isdigit() for ch in s)
            dots = s.count('.')
            hyphens = s.count('-')
            slashes = s.count('/')
            at_sign = 1 if '@' in s else 0
            has_ip = 1 if re.search(r'://\d+\.\d+\.\d+\.\d+', s) else 0
            out.append([length, digits, dots, hyphens, slashes, at_sign, has_ip])
        return np.array(out)

def main():
    df = pd.read_csv(DATA_PATH)
    df['url'] = df['url'].apply(canonicalize)
    df = df[df['label'].notnull()]
    X = df['url'].values
    y = df['label'].astype(int).values

    # optional sampling: keep negatives at most 3x positives to avoid huge imbalance on low-memory machines
    from collections import Counter
    cnt = Counter(y)
    n_pos = int(cnt.get(1,0))
    n_neg = int(cnt.get(0,0))
    if n_neg > max(1000, n_pos * 3):
        neg_idx = np.where(y==0)[0]
        pos_idx = np.where(y==1)[0]
        np.random.seed(42)
        keep_neg = np.random.choice(neg_idx, size=min(len(neg_idx), max(1000, n_pos*3)), replace=False)
        keep_idx = np.concatenate([pos_idx, keep_neg])
        X = X[keep_idx]
        y = y[keep_idx]
        print("Downsampled negatives for training. New counts:", Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), max_features=3000)

    union = FeatureUnion([
        ('tfidf', Pipeline([('tf', tfidf)])),
        ('num', Pipeline([('numfeat', URLNumericTransformer()), ('scale', StandardScaler())]))
    ])

    clf_pipeline = Pipeline([('union', union), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])

    print("Training on", len(X_train), "samples...")
    clf_pipeline.fit(X_train, y_train)

    pred = clf_pipeline.predict(X_test)
    proba = clf_pipeline.predict_proba(X_test)[:,1]

    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_pipeline, OUT_MODEL)
    print("Saved model to:", OUT_MODEL)

if __name__ == "__main__":
    main()
