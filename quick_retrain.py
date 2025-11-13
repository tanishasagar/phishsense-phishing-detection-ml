# quick_retrain.py  -- fast balanced retrain for demo
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import joblib
import json

ROOT = Path(__file__).resolve().parent
models_dir = ROOT / "models"
models_dir.mkdir(exist_ok=True)
# File paths - update if yours are different
phishtank_p = ROOT / "dataset" / "phishtank.csv"
benign_p = ROOT / "dataset" / "Benign_list_big_final.csv"
allurls_p = ROOT / "dataset" / "all_urls.csv"

def load_urls(path, url_col_candidates=None):
    if not path.exists():
        return pd.DataFrame(columns=["url"])
    try:
        df = pd.read_csv(path, low_memory=False)
        # try common column names
        for c in (["url","URL","domain","link","url_raw"] if url_col_candidates is None else url_col_candidates):
            if c in df.columns:
                return pd.DataFrame({"url": df[c].astype(str).str.strip()})
        # fallback: if only one column
        if df.shape[1] == 1:
            return pd.DataFrame({"url": df.iloc[:,0].astype(str).str.strip()})
        # otherwise try to find first column with http
        for c in df.columns:
            if df[c].astype(str).str.contains("http", na=False).any():
                return pd.DataFrame({"url": df[c].astype(str).str.strip()})
        return pd.DataFrame()
    except Exception:
        try:
            # try reading as plain text list
            s = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            return pd.DataFrame({"url": [x.strip() for x in s if x.strip()]})
        except Exception:
            return pd.DataFrame()

# load
df_phish = load_urls(phishtank_p)
df_ben = load_urls(benign_p)
df_all = load_urls(allurls_p)

print("Counts:", len(df_phish), len(df_ben), len(df_all))

# create labels and combine
df_phish = df_phish.drop_duplicates().assign(label=1)
df_ben = df_ben.drop_duplicates().assign(label=0)
df_all = df_all.drop_duplicates().assign(label=0)  # treat all_urls as benign if unlabeled

# combine
df = pd.concat([df_phish, df_ben, df_all], ignore_index=True).drop_duplicates("url")
# quick clean
df = df[df['url'].notna() & (df['url'].str.len()>3)]

# sample balanced: take up to N per class
N = 2000  # change to 5000/10000 if you have time/memory
ph = df[df.label==1]
bg = df[df.label==0]
n = min(len(ph), len(bg), N)
if n < 100:
    print("Not enough data for a balanced sample; using available counts")
    n = min(len(ph), len(bg))
train_df = pd.concat([ph.sample(n, random_state=42), bg.sample(n, random_state=42)], ignore_index=True).sample(frac=1, random_state=42)

X = train_df['url'].values
y = train_df['label'].values

# build TF-IDF on character ngrams for URLs + simple numeric features (length, digits)
def numeric_feats(urls):
    out = []
    import re
    for u in urls:
        s = str(u)
        out.append([len(s), sum(ch.isdigit() for ch in s), s.count('.'), s.count('-'), int(bool(re.search(r'://\\d+\\.\\d+\\.\\d+\\.\\d+', s)))])
    return np.array(out)

num_transformer = FunctionTransformer(lambda x: numeric_feats(x.ravel()), validate=False)

pre = ColumnTransformer(transformers=[
    ("tfidf", TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), max_features=2000), 0),
    ("num", num_transformer, 0)
], remainder='drop')

# classifier + calibration
clf = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs')
cal = CalibratedClassifierCV(clf, cv=3, method='isotonic')

pipe = Pipeline([
    ("pre", pre),
    ("clf", cal)
])

print("Training on", len(X), "samples")
pipe.fit(X.reshape(-1,1), y)

# save
joblib.dump(pipe, models_dir / "phishsense_pipeline_quick.joblib")
# save feature names (we'll save minimal)
with open(models_dir / "feature_names_quick.json","w",encoding="utf-8") as fh:
    json.dump(["url_tfidf","num_feats"], fh)
print("Saved quick pipeline to", models_dir / "phishsense_pipeline_quick.joblib")
