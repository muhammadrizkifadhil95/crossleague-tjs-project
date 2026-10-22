import pandas as pd
import os

# Get the path to the TJS_Project_V2 folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the absolute path to your master dataset
DATA_PATH = os.path.join(BASE_DIR, 'data_extraction', 'data', 'final_df_AAA_v2.csv')

# Load the data
df = pd.read_csv(DATA_PATH)

# Pitcher counter results for your report
print(f"Name of file: {DATA_PATH}")
print(f"Total rows in file: {len(df)}")
print(f"Unique pitchers in file: {df['pitcher'].nunique()}")
print(f"Injury (Target 1) cases: {df[df['target'] == 1]['pitcher'].nunique()}")
print(f"Healthy (Target 0) cases: {df[df['target'] == 0]['pitcher'].nunique()}")

print(pd.read_csv(DATA_PATH, nrows=0).columns.tolist())