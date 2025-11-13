import pandas as pd
import os

# Make Path to Dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "FakeNewsNet.csv")

# Load CSV Data
try:
    df = pd.read_csv(DATA_DIR)
    print("[DATASET] - Dataset found, loading...")
except FileNotFoundError:
    print("[DATASET ERROR] - Dataset file not found in directory")

# Show dataset structure
print("[DATASET] - Loading dataset structure...")
print("\n-------------- DATASET STRUCTURE --------------\n")
print(df.info())

# Show first 5 rows for preview
print("\n-------------- DATASET PREVIEW --------------\n")
print(df.head())

print("\n -------------- LABEL DISTRIBUTION -------------- \n")
if "title" in df.columns:
    print(df["title"].value_counts())
else:
    print("[DATASET ERROR] - 'title' column not found in dataset")