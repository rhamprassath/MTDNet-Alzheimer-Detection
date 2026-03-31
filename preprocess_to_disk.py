import os
import numpy as np
import pandas as pd
from data_loader import EEGDataLoader
import gc

def main():
    root = r"d:\al project\ADFTD"
    tsv = r"d:\al project\ADFTD_git\participants.tsv"
    out_dir = r"d:\al project\processed_data"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    loader = EEGDataLoader(root, tsv)
    file_df = loader.get_subject_files()
    full_df = file_df.merge(loader.participants, on='participant_id')
    
    print(f"Total subjects to process: {len(full_df)}")
    
    for idx, row in full_df.iterrows():
        out_file = os.path.join(out_dir, f"{row['participant_id']}.npz")
        
        if os.path.exists(out_file):
            print(f"Skipping {row['participant_id']}, already exists.")
            continue
            
        print(f"Processing {row['participant_id']}...")
        segments = loader.load_and_preprocess(row['file_path'])
        
        if segments is not None:
            # Save segments and metadata
            np.savez(out_file, x=segments, y=loader.label_map[row['Group']], group=idx)
            print(f"Saved {len(segments)} segments for {row['participant_id']}.")
        
        gc.collect()

if __name__ == "__main__":
    main()
