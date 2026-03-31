import os
import torch
import numpy as np
import glob
from model import MTDNet
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

# Configuration
CONFIG = {
    'processed_dir': r"d:\al project\processed_data",
    'model_path': "mtdnet_best_npz.pth",
    'n_channels': 19,
    'n_features': 128,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def load_all_npz(data_dir):
    all_x = []
    all_y = []
    all_subject_ids = []
    
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    print(f"Testing on {len(files)} subjects...")
    
    for f in files:
        data = np.load(f)
        x = data['x']
        y = int(data['y'])
        
        all_x.append(x)
        all_y.extend([y] * len(x))
        all_subject_ids.append((os.path.basename(f), len(x), y))
        
    X = np.concatenate(all_x, axis=0)
    y = np.array(all_y)
    return X, y, all_subject_ids

def main():
    print("--- MTDNet Comprehensive Model Testing (Memory Efficient) ---")
    
    if not os.path.exists(CONFIG['model_path']):
        print(f"ERROR: Model checkpoint not found at {CONFIG['model_path']}")
        return

    # Load Model
    model = MTDNet(n_channels=CONFIG['n_channels'], n_features=CONFIG['n_features']).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    model.eval()
    
    files = glob.glob(os.path.join(CONFIG['processed_dir'], "*.npz"))
    print(f"Testing on {len(files)} subjects...")
    
    y_true_all = []
    y_pred_all = []
    subject_consensus = []
    
    with torch.no_grad():
        for f in files:
            data = np.load(f)
            x = data['x']
            y = int(data['y'])
            
            x_tensor = torch.from_numpy(x).float().to(CONFIG['device'])
            
            # Prediction
            outputs = model(x_tensor)
            _, predicted = torch.max(outputs.data, 1)
            preds = predicted.cpu().numpy()
            
            y_true_all.extend([y] * len(preds))
            y_pred_all.extend(preds)
            
            # Subject-level Consensus
            unique, counts = np.unique(preds, return_counts=True)
            winner = unique[np.argmax(counts)]
            subject_consensus.append((os.path.basename(f), y, winner))
            
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print(f"Macro F1 Score:   {f1:.4f}")
    
    print("\n--- CONFUSION MATRIX ---")
    print(cm)
    print("\nLegend: [[TN, FP], [FN, TP]] for binary classification (HC=0, AD=1)")
    
    print("\n--- CLASSIFICATION REPORT ---")
    target_names = ["Healthy Control (HC)", "Alzheimer's (AD)"]
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Subject-level analysis (Expert Rigor)
    print("\n--- SUBJECT-LEVEL DIAGNOSTIC ANALYSIS (Consensus) ---")
    correct_subjects = 0
    for sid, label, winner in subject_consensus:
        status = "CORRECT" if winner == label else "FAILED"
        if winner == label: correct_subjects += 1
        print(f"Subject {sid}: True={label}, Pred={winner} -> {status}")
        
    print(f"\nSubject-Level Accuracy: {correct_subjects}/{len(subject_consensus)} ({correct_subjects/len(subject_consensus)*100:.2f}%)")

if __name__ == "__main__":
    main()
