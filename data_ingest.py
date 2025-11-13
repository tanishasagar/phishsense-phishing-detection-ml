# data_ingest.py
import pandas as pd
import time
from src.feature_extractor import extract_features
from pathlib import Path
from tqdm import tqdm

INPUT_CSV = "data/raw/all_urls.csv"   # must have columns: url,label
OUTPUT_CSV = "data/processed/all_features.csv"
RATE_LIMIT = 0.3  # seconds between requests; set to 0.5-1.0 if you see many failures

def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        u = str(r['url']).strip()
        if not u:
            rows.append({'url': u, 'label': int(r.get('label', 0))})
            continue
        try:
            feats = extract_features(u)
        except Exception:
            feats = {}
        feats['label'] = int(r.get('label', 0))
        rows.append(feats)
        time.sleep(RATE_LIMIT)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print("Wrote", OUTPUT_CSV)

if __name__ == "__main__":
    main()
