"""
Comprehensive Subject-Level Testing - Tests ALL available model variants.
Uses score averaging (more robust than majority voting) for patient-level diagnosis.
"""
import os
import torch
import numpy as np
import glob
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Test multiple models
MODELS_TO_TEST = [
    {
        'name': 'V1 (Original)',
        'path': 'mtdnet_best_npz.pth',
        'data_dir': r'd:\al project\processed_data',
        'model_class': 'MTDNet',
        'n_features': 128
    },
    {
        'name': 'V4 (4s segments)',
        'path': 'mtdnet_v4_best.pth',
        'data_dir': r'd:\al project\processed_v2',
        'model_class': 'MTDNetV4',
        'n_features': 32
    },
]

def test_model(model_config):
    name = model_config['name']
    model_path = model_config['path']
    data_dir = model_config['data_dir']
    
    if not os.path.exists(model_path):
        print(f"  SKIP: {model_path} not found")
        return None
    if not os.path.exists(data_dir):
        print(f"  SKIP: {data_dir} not found")
        return None
    
    device = torch.device("cpu")
    
    # Load the right model class
    if model_config['model_class'] == 'MTDNet':
        from model import MTDNet
        model = MTDNet(n_channels=19, n_features=model_config['n_features']).to(device)
    elif model_config['model_class'] == 'MTDNetV4':
        from train_v4 import MTDNetV4
        model = MTDNetV4(n_channels=19, n_features=model_config['n_features']).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    
    subject_results = []
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for f in files:
            data = np.load(f)
            x = data['x']
            y = int(data['y'])
            
            x_tensor = torch.from_numpy(x).float().to(device)
            outputs = model(x_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Score Averaging (more robust than majority voting)
            avg_probs = np.mean(probs, axis=0)
            pred_label = np.argmax(avg_probs)
            confidence = avg_probs[pred_label]
            
            # Also get majority voting result
            _, preds_mv = torch.max(outputs, 1)
            preds_mv = preds_mv.cpu().numpy()
            unique, counts = np.unique(preds_mv, return_counts=True)
            mv_label = unique[np.argmax(counts)]
            mv_conf = counts[np.argmax(counts)] / len(preds_mv)
            
            subject_results.append({
                'id': os.path.basename(f),
                'true': y,
                'avg_pred': pred_label,
                'avg_conf': confidence,
                'mv_pred': mv_label,
                'mv_conf': mv_conf
            })
            
            all_y_true.append(y)
            all_y_pred.append(pred_label)
    
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'name': name,
        'acc': acc,
        'f1': f1,
        'cm': cm,
        'subjects': subject_results,
        'n_subjects': len(files)
    }


def main():
    print("=" * 70)
    print("  COMPREHENSIVE MODEL COMPARISON - Subject-Level Diagnostics")
    print("=" * 70)
    
    results = []
    
    for cfg in MODELS_TO_TEST:
        print(f"\n--- Testing {cfg['name']} ---")
        r = test_model(cfg)
        if r:
            results.append(r)
    
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    
    for r in results:
        print(f"\n{'='*40}")
        print(f"Model: {r['name']}")
        print(f"Subjects: {r['n_subjects']}")
        print(f"Subject-Level Accuracy: {r['acc']:.4f} ({int(r['acc']*r['n_subjects'])}/{r['n_subjects']})")
        print(f"Subject-Level F1 (Macro): {r['f1']:.4f}")
        print(f"Confusion Matrix:")
        print(f"  TN(HC→HC)={r['cm'][0,0]}, FP(HC→AD)={r['cm'][0,1]}")
        print(f"  FN(AD→HC)={r['cm'][1,0]}, TP(AD→AD)={r['cm'][1,1]}")
        
        print(f"\nSubject Details (Score Averaging):")
        correct = 0
        for s in r['subjects']:
            status = "✓" if s['avg_pred'] == s['true'] else "✗"
            label_name = "AD" if s['true'] == 1 else "HC"
            pred_name = "AD" if s['avg_pred'] == 1 else "HC"
            if s['avg_pred'] == s['true']:
                correct += 1
            print(f"  {status} {s['id']}: True={label_name}, Pred={pred_name} (conf={s['avg_conf']:.2%})")
        
        print(f"\nCorrect: {correct}/{r['n_subjects']} ({correct/r['n_subjects']:.2%})")
    
    # Determine best model
    if results:
        best = max(results, key=lambda x: x['f1'])
        print(f"\n{'='*70}")
        print(f"  WINNER: {best['name']} with F1={best['f1']:.4f}, Acc={best['acc']:.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
