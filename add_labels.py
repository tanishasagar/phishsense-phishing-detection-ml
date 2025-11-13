# add_labels.py — add a label column to the sample CSV
import pandas as pd
df = pd.read_csv("data/processed/sample_features.csv")

# Replace these labels if you want different mapping for sample rows
df['label'] = [0, 0, 1, 1]

df.to_csv("data/processed/sample_features.csv", index=False)
print("Added label column to data/processed/sample_features.csv")
