import os
import glob
import numpy as np
from predict import AlzheimerPredictor

def test_ensemble_on_real_data():
    data_dir = r"d:\al project\processed_v2"
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if not files:
        print("No files found in processed_v2")
        return
        
    predictor = AlzheimerPredictor()
    
    correct = 0
    total = len(files)
    
    print(f"\n--- EXPERT VERIFICATION: TESTING ENSEMBLE ON {total} REAL SUBJECTS ---")
    
    for f in files:
        data = np.load(f)
        segments = data['x']
        y_true = int(data['y'])
        
        # Reconstruct rough raw EEG from shape (n_segments, 19, 512)
        # Assuming 128Hz, each segment is 4s.
        raw_eeg = np.transpose(segments, (1, 0, 2)).reshape(19, -1)
        
        # Silence prints
        import sys, os as os_mod
        old_stdout = sys.stdout
        sys.stdout = open(os_mod.devnull, 'w')
        
        result = predictor.diagnose(raw_eeg, fs=128, strategy='average')
        
        sys.stdout = old_stdout
        
        y_pred_str = result['diagnosis']
        if y_pred_str == "Healthy Control (HC)":
            y_pred = 0
        elif y_pred_str == "Mild Cognitive Impairment (MCI)":
            # Count MCI as leaning AD but technically its own class, let's treat it as AD for binary accuracy
            y_pred = 1
        else:
            y_pred = 1
            
        is_correct = (y_pred == y_true)
        if is_correct: correct +=1
        
        status = "✅ CORRECT" if is_correct else "❌ FAILED"
        print(f"Subject {os.path.basename(f)}: True={predictor.class_names[y_true][:3]} | Pred={y_pred_str} -> {status} (Conf: {result['confidence']})")
        
    print(f"\nFINAL ENSEMBLE CLINICAL ACCURACY: {correct}/{total} ({correct/total*100:.2f}%)")

if __name__ == "__main__":
    test_ensemble_on_real_data()
