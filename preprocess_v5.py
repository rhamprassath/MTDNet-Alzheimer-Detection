import os
import glob
import pandas as pd
import numpy as np
import torch
from utils import bandpass_filter, z_score_normalize, segment_signal

# Settings
BASE_DIR = r"d:\al project\Final dataset for the published paper"
FOLDERS = {
    "1-Healthy": 0, # Healthy Control
    "2-Mild": 1,    # Alzheimer's Disease
    "3-Moderate": 1,
    "4-Sever": 1
}
FS = 128 # 128Hz for clinical CSVs
SEGMENT_LEN = 4 # Target 4s for V4 LSTM
OUTPUT_FILE = "v5_train_data.npz"

def process_all():
    print(f"--- V5 PREPROCESSOR: MASTER CLINICAL DATASET (Low Memory) ---")
    
    X_list = []
    y_list = []
    
    # 50+ Experience Tip: Avoid exhausting RAM by processing subject-by-subject.
    # Convert each subject to float32 numpy array immediately.

    for folder, label in FOLDERS.items():
        path = os.path.join(BASE_DIR, folder)
        files = glob.glob(os.path.join(path, "*.csv"))
        print(f"\nProcessing {folder} ({len(files)} subjects)...")
        
        for f in files:
            try:
                # Use a fast, low-memory reader
                df = pd.read_csv(f, engine='c', low_memory=True)
                
                # Numeric only
                df = df.select_dtypes(include=[np.number])
                
                if df.shape[1] < 19:
                     continue
                         
                # Extract 19 EEG channels
                data = df.iloc[:, :19].values.T.astype(np.float32) # (19, samples)
                
                # Apply Architect Overhaul (Filtering)
                data = bandpass_filter(data, 0.5, 45.0, FS)
                
                # Segment this subject (Overlapping 50% stride)
                segments = segment_signal(data, FS, segment_len_sec=SEGMENT_LEN, stride_sec=2)
                
                # Diversity Limit: cap at 40 segments per subject for model balance
                if len(segments) > 40:
                     idx = np.random.choice(len(segments), 40, replace=False)
                     segments = [segments[i] for i in idx]
                
                # Robust Normalization and Labeling
                if segments:
                    subj_block = np.array([z_score_normalize(s) for s in segments]).astype(np.float32)
                    X_list.append(subj_block)
                    y_list.append(np.full(len(segments), label, dtype=np.int64))
                    print(f"    {os.path.basename(f)}: +{len(segments)} segments")
                        
            except Exception as e:
                print(f"    Error processing {os.path.basename(f)}: Skipping ({e})")
                
    if not X_list:
        print("ERROR: No segments processed!")
        return

    # Consolidate memory efficiently
    print("\nConsolidating master dataset...")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"\n--- SUCCESS ---")
    print(f"Total Master Segments: {len(X)}")
    print(f"Class Distribution: HC={len(y[y==0])}, AD={len(y[y==1])}")
    print(f"Saving to {OUTPUT_FILE}...")
    np.savez(OUTPUT_FILE, x=X, y=y)

if __name__ == "__main__":
    process_all()
