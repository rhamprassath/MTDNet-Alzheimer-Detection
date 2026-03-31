"""
MTDNet V2 Training Script - Expert-Level Configuration
Key improvements:
- Uses MTDNet V2 (bidirectional LSTM, dropout, deeper head)
- 4-second segments for richer temporal context
- Gradient clipping for training stability
- Cosine Annealing LR scheduler
- Label smoothing in CrossEntropy
- Comprehensive logging every epoch
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model_v2 import MTDNetV2
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
import glob

CONFIG = {
    'processed_dir': r"d:\al project\processed_v2",
    'n_channels': 19,
    'n_features': 64,
    'batch_size': 64,
    'lr': 0.0005,
    'epochs': 150,
    'dropout': 0.3,
    'grad_clip': 1.0,
    'label_smoothing': 0.1,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'model_save_path': "mtdnet_v2_best.pth"
}

class NPZDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.from_numpy(x_data).float()
        self.y = torch.from_numpy(y_data).long()
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_all_npz(data_dir):
    all_x, all_y, all_groups = [], [], []
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    print(f"Loading {len(files)} subjects from disk...")
    for i, f in enumerate(files):
        data = np.load(f)
        x = data['x']
        y = int(data['y'])
        all_x.append(x)
        all_y.extend([y] * len(x))
        all_groups.extend([i] * len(x))
    X = np.concatenate(all_x, axis=0)
    y = np.array(all_y)
    groups = np.array(all_groups)
    return X, y, groups

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())
    
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return total_loss / len(loader), acc, f1

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return total_loss / len(loader), acc, f1

def main():
    print("=" * 60)
    print("  MTDNet V2 Training - Expert Configuration")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    
    if not os.path.exists(CONFIG['processed_dir']):
        print(f"ERROR: No preprocessed data at {CONFIG['processed_dir']}")
        return
    
    X, y, groups = load_all_npz(CONFIG['processed_dir'])
    n_subjects = len(np.unique(groups))
    n_ad = np.sum(y == 1)
    n_hc = np.sum(y == 0)
    print(f"Dataset: {len(X)} segments from {n_subjects} subjects")
    print(f"  AD segments: {n_ad} | HC segments: {n_hc}")
    print(f"  Segment shape: {X[0].shape}")
    
    # Subject-independent split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    print(f"Train: {len(train_idx)} segments | Val: {len(val_idx)} segments")
    
    train_ds = NPZDataset(X[train_idx], y[train_idx])
    val_ds = NPZDataset(X[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], num_workers=0)
    
    # Model
    model = MTDNetV2(
        n_channels=CONFIG['n_channels'],
        n_features=CONFIG['n_features'],
        n_classes=2,
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {params:,}")
    
    # Class Weights
    labels_unique, counts = np.unique(y[train_idx], return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * 2.0
    class_weights = torch.FloatTensor(weights).to(CONFIG['device'])
    print(f"Class Weights: {weights} (Labels: {labels_unique})")
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=CONFIG['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_f1 = 0
    patience_counter = 0
    patience = 20
    
    print("\n" + "-" * 60)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train F1':>8} | {'Val F1':>8} | {'Val Acc':>8} | {'LR':>10}")
    print("-" * 60)
    
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, CONFIG['device'], CONFIG['grad_clip']
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, CONFIG['device'])
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log every epoch for transparency
        print(f"{epoch+1:>5} | {train_loss:>10.4f} | {train_f1:>8.4f} | {val_f1:>8.4f} | {val_acc:>8.4f} | {current_lr:>10.6f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f"  >>> NEW BEST checkpoint saved! Val F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}. Best Val F1: {best_f1:.4f}")
            break
    
    # Final evaluation with best model
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION ON BEST MODEL")
    print("=" * 60)
    
    model.load_state_dict(torch.load(CONFIG['model_save_path'], map_location=CONFIG['device']))
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(CONFIG['device']), batch_y.to(CONFIG['device'])
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=['HC (0)', 'AD (1)']))
    print(f"FINAL Val Accuracy: {accuracy_score(all_targets, all_preds):.4f}")
    print(f"FINAL Val F1 (Macro): {f1_score(all_targets, all_preds, average='macro'):.4f}")
    print(f"\nTraining complete. Best model saved to: {CONFIG['model_save_path']}")

if __name__ == "__main__":
    main()
