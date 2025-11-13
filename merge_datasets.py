# merge_datasets.py
import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT / "all_combined.csv"
COLL_PATH = OUT / "collisions_for_review.csv"

def canonicalize(u):
    if pd.isna(u): return ""
    s = str(u).strip()
    s = s.split('#')[0]
    s = re.sub(r'(\?|&)(utm_[^=]+|fbclid|gclid)=[^&]*', '', s)
    s = re.sub(r'\?&|&&', '?', s)
    return s.lower()

def load_file(path, assume_label=None):
    df = pd.read_csv(path, dtype=str, encoding='utf-8', engine='python', on_bad_lines='skip')
    if 'url' in df.columns:
        urls = df['url'].astype(str)
    else:
        urls = df.iloc[:,0].astype(str)
        urls.name = 'url'
        df = urls.to_frame()
    df = df[['url']].copy()
    df['url'] = df['url'].apply(canonicalize)
    if assume_label is not None:
        df['label'] = assume_label
    else:
        if 'label' in df.columns:
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        else:
            df['label'] = None
    return df

def main():
    ph = load_file(RAW / "phishtank.csv", assume_label=1) if (RAW / "phishtank.csv").exists() else None
    bn = load_file(RAW / "Benign_list_big_final.csv", assume_label=0) if (RAW / "Benign_list_big_final.csv").exists() else None
    all_urls = load_file(RAW / "all_urls.csv", assume_label=None) if (RAW / "all_urls.csv").exists() else None

    frames = [p for p in (ph, bn, all_urls) if p is not None]
    if not frames:
        raise SystemExit("No input CSVs found in data/raw/ - please add them.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[combined['url'].str.strip().astype(bool)]

    # find collision URLs (same url but different labels)
    dup = combined[combined.duplicated(subset=['url'], keep=False)].copy()
    collisions = dup.groupby('url')['label'].nunique().reset_index()
    collisions = collisions[collisions['label']>1]
    collision_count = len(collisions)
    if collision_count:
        conflicting_rows = combined[combined['url'].isin(collisions['url'])]
        conflicting_rows.to_csv(COLL_PATH, index=False)
        print(f"WARNING: {collision_count} collision URLs saved to {COLL_PATH}. Please review labels.")
    else:
        if COLL_PATH.exists():
            COLL_PATH.unlink()

    combined = combined.drop_duplicates(subset=['url'], keep='first').reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)
    print("Saved combined CSV to:", OUT_PATH)
    print("Rows in combined:", len(combined))

if __name__ == "__main__":
    main()
