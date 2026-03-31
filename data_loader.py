import os
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

class EEGDataLoader:
    def __init__(self, data_root, participants_tsv, fs_target=128, segment_len_sec=1):
        self.data_root = data_root
        self.fs_target = fs_target
        self.segment_len_sec = segment_len_sec
        self.participants = pd.read_csv(participants_tsv, sep='\t')
        
        # Map Group to labels: A (AD) -> 1, C (HC) -> 0, F (FTD) -> 2 (optional)
        self.label_map = {'A': 1, 'C': 0, 'F': 2}
        
    def get_subject_files(self):
        """
        Scan for .set files in the derivatives folder.
        """
        files = []
        derivatives_path = os.path.join(self.data_root, 'derivatives')
        if not os.path.exists(derivatives_path):
            # Fallback to root if derivatives directory isn't found
            derivatives_path = self.data_root
            
        for root, dirs, filenames in os.walk(derivatives_path):
            for f in filenames:
                if f.endswith('.set') and 'task-eyesclosed' in f:
                    # Extract participant_id from filename (e.g., sub-001)
                    pid = f.split('_')[0]
                    files.append({'participant_id': pid, 'file_path': os.path.join(root, f)})
        return pd.DataFrame(files)

    def load_and_preprocess(self, file_path):
        """
        Load .set file, filter, resample, and segment.
        """
        try:
            # Load raw data
            # Use preload=True to avoid size issues with shared mem in some envs
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            raw.filter(0.5, 45.0, verbose=False)
            raw.resample(self.fs_target, verbose=False)
            
            data = raw.get_data() # (n_channels, n_samples)
            n_samples = data.shape[1]
            
            # Segmenting
            samples_per_seg = int(self.fs_target * self.segment_len_sec)
            n_segments = n_samples // samples_per_seg
            
            segments = []
            for i in range(n_segments):
                seg = data[:, i*samples_per_seg : (i+1)*samples_per_seg]
                # Z-score normalization
                seg = (seg - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True) + 1e-8)
                segments.append(seg)
                
            # Cleanup raw data to free memory
            del raw
            del data
            
            return np.array(segments, dtype=np.float32)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    import gc
    def prepare_dataset(self, binary_ad_hc=True):
        """
        Full pipeline: scan -> load -> split.
        """
        file_df = self.get_subject_files()
        full_df = file_df.merge(self.participants, on='participant_id')
        
        if binary_ad_hc:
            full_df = full_df[full_df['Group'].isin(['A', 'C'])]
            
        all_x = []
        all_y = []
        all_groups = []
        
        print(f"Loading {len(full_df)} subjects...")
        
        for idx, row in full_df.iterrows():
            segments = self.load_and_preprocess(row['file_path'])
            if segments is not None:
                all_x.append(segments)
                label = self.label_map[row['Group']]
                all_y.extend([label] * len(segments))
                all_groups.extend([idx] * len(segments))
            
            # Force Memory Cleanup hourly or after each subject
            import gc
            gc.collect()
            
        if not all_x:
            return np.array([]), np.array([]), np.array([])
            
        X = np.concatenate(all_x, axis=0)
        y = np.array(all_y)
        groups = np.array(all_groups)
        
        return X, y, groups

if __name__ == "__main__":
    # Test with available data
    root = r"d:\al project\ADFTD"
    tsv = r"d:\al project\ADFTD_git\participants.tsv"
    
    loader = EEGDataLoader(root, tsv)
    files = loader.get_subject_files()
    print(f"Found {len(files)} .set files.")
    if len(files) > 0:
        sample = loader.load_and_preprocess(files.iloc[0]['file_path'])
        print(f"Sample segments shape: {sample.shape}")
