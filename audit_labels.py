import numpy as np
import glob
import os

processed_dir = r"d:\al project\processed_data"
files = glob.glob(os.path.join(processed_dir, "*.npz"))

print(f"Auditing {len(files)} files...")
freqs = {}

for f in files:
    try:
        data = np.load(f)
        label = int(data['y'])
        freqs[label] = freqs.get(label, 0) + 1
        if label == 2:
            print(f"FOUND LABEL 2: {os.path.basename(f)}")
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("\nLabel Frequencies:")
for l, count in freqs.items():
    name = "HC (0)" if l == 0 else "AD (1)" if l == 1 else "FTD (2)"
    print(f"  {name}: {count} files")
