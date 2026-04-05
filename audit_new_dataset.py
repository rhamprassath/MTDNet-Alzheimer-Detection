import os
import glob
import pandas as pd

base_dir = r"d:\al project\Final dataset for the published paper"
folders = ["1-Healthy", "2-Mild", "3-Moderate", "4-Sever"]

print("--- NEW DATASET COLUMN AUDIT ---")
for folder in folders:
    path = os.path.join(base_dir, folder)
    files = glob.glob(os.path.join(path, "*.csv"))
    print(f"\nFolder: {folder} ({len(files)} files)")
    
    col_counts = {}
    for f in files:
        try:
            # Read only first line to get column count
            df = pd.read_csv(f, nrows=0)
            count = len(df.columns)
            col_counts[count] = col_counts.get(count, 0) + 1
        except Exception as e:
            print(f"Error reading {os.path.basename(f)}: {e}")
            
    for count, freq in col_counts.items():
        print(f"  - {count} columns: {freq} files")
