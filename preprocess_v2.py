"""
Enhanced Preprocessing Pipeline V2
Key improvements:
- 4-second segments (512 samples at 128Hz) for richer temporal context
- Overlapping windows (50% overlap) for data augmentation
- Filters only AD and HC subjects strictly
"""
import os
import numpy as np
import pandas as pd
from data_loader import EEGDataLoader
import gc

SEGMENT_LEN_SEC = 4      # 4 seconds (was 1)
OVERLAP_RATIO = 0.5       # 50% overlap
FS_TARGET = 128

def preprocess_subject(loader, file_path, segment_len_sec=SEGMENT_LEN_SEC, overlap=OVERLAP_RATIO):
    """Load and preprocess a single subject with overlapping segments."""
    try:
        import mne
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        raw.filter(0.5, 45.0, verbose=False)
        raw.resample(FS_TARGET, verbose=False)
        
        data = raw.get_data()  # (n_channels, n_samples)
        n_samples = data.shape[1]
        
        samples_per_seg = int(FS_TARGET * segment_len_sec)
        step = int(samples_per_seg * (1 - overlap))
        
        segments = []
        start = 0
        while start + samples_per_seg <= n_samples:
            seg = data[:, start:start + samples_per_seg]
            # Z-score normalization per channel
            seg = (seg - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True) + 1e-8)
            segments.append(seg)
            start += step
        
        del raw, data
        gc.collect()
        
        return np.array(segments, dtype=np.float32) if segments else None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def main():
    root = r"d:\al project\ADFTD"
    tsv = r"d:\al project\ADFTD_git\participants.tsv"
    out_dir = r"d:\al project\processed_v2"
    
    os.makedirs(out_dir, exist_ok=True)
    
    loader = EEGDataLoader(root, tsv, fs_target=FS_TARGET, segment_len_sec=SEGMENT_LEN_SEC)
    file_df = loader.get_subject_files()
    full_df = file_df.merge(loader.participants, on='participant_id')
    
    # STRICT: Only AD and HC
    full_df = full_df[full_df['Group'].isin(['A', 'C'])]
    
    print(f"Total AD/HC subjects to process: {len(full_df)}")
    
    ad_count = len(full_df[full_df['Group'] == 'A'])
    hc_count = len(full_df[full_df['Group'] == 'C'])
    print(f"AD: {ad_count}, HC: {hc_count}")
    
    for idx, row in full_df.iterrows():
        out_file = os.path.join(out_dir, f"{row['participant_id']}.npz")
        
        if os.path.exists(out_file):
            print(f"Skipping {row['participant_id']}, already exists.")
            continue
        
        print(f"Processing {row['participant_id']} (Group: {row['Group']})...")
        segments = preprocess_subject(loader, row['file_path'])
        
        if segments is not None:
            label = loader.label_map[row['Group']]
            np.savez(out_file, x=segments, y=label, group=idx)
            print(f"  -> Saved {len(segments)} segments (4s, 50% overlap) for {row['participant_id']}.")
        
        gc.collect()

if __name__ == "__main__":
    main()
