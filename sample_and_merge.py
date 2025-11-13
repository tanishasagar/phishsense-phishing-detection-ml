# sample_and_merge.py
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT = RAW_DIR / "all_urls.csv"

# filenames on your machine (as listed)
phish_file = RAW_DIR / "phishtank.csv"
benign_file = RAW_DIR / "Benign_list_big_final.csv"

# sample sizes (change if you want)
N_PHISH = 500
N_BENIGN = 500

def load_csv_guess_header(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    candidates = [c for c in df.columns if c.lower() in ("url","link","website","domain","address")]
    if candidates:
        col = candidates[0]
    else:
        col = df.columns[0]
    return df[[col]].rename(columns={col: "url"})

def main():
    print("Loading phishing file:", phish_file)
    ph = load_csv_guess_header(phish_file)
    ph = ph.dropna().drop_duplicates().reset_index(drop=True)
    if len(ph) > N_PHISH:
        ph = ph.sample(N_PHISH, random_state=42)
    ph["label"] = 1

    print("Loading benign file:", benign_file)
    be = load_csv_guess_header(benign_file)
    be = be.dropna().drop_duplicates().reset_index(drop=True)
    if len(be) > N_BENIGN:
        be = be.sample(N_BENIGN, random_state=42)
    be["label"] = 0

    combined = pd.concat([ph, be], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    combined.to_csv(OUT, index=False)
    print("Wrote:", OUT, "rows:", len(combined))

if __name__ == "__main__":
    main()
