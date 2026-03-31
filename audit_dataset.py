"""Audit all 65 subjects in the dataset."""
import numpy as np
import glob
import os

dirs = {
    'V1 (1s segments)': r'd:\al project\processed_data',
    'V2 (4s segments)': r'd:\al project\processed_v2',
}

for name, d in dirs.items():
    files = sorted(glob.glob(os.path.join(d, "*.npz")))
    print(f"\n{'='*60}")
    print(f"  DATASET: {name} ({d})")
    print(f"  Total Files: {len(files)}")
    print(f"{'='*60}")
    
    print(f"\n{'ID':<14} {'Group':<6} {'Segments':<10} {'Shape'}")
    print("-" * 55)
    
    ad_count, hc_count = 0, 0
    ad_segs, hc_segs = 0, 0
    
    for f in files:
        data = np.load(f)
        x = data['x']
        y = int(data['y'])
        label = "AD" if y == 1 else "HC"
        basename = os.path.basename(f).replace('.npz', '')
        
        if y == 1:
            ad_count += 1
            ad_segs += len(x)
        else:
            hc_count += 1
            hc_segs += len(x)
        
        print(f"{basename:<14} {label:<6} {len(x):<10} {x.shape}")
    
    print(f"\n--- SUMMARY ---")
    print(f"AD subjects: {ad_count} ({ad_segs} segments)")
    print(f"HC subjects: {hc_count} ({hc_segs} segments)")
    print(f"Total: {ad_count + hc_count} subjects ({ad_segs + hc_segs} segments)")

if __name__ == "__main__":
    pass
