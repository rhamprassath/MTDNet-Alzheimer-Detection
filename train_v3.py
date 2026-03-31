"""
MTDNet V3 Training - Anti-Overfitting Expert Configuration
Key changes from V2:
- Simpler unidirectional LSTM (prevents overfitting on small datasets)
- Higher dropout (0.5)
- Mixup data augmentation
- Larger batch size (128) for smoother gradients
- Stronger weight decay (5e-3)
- OneCycleLR scheduler (proven for fast convergence)
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
import glob

CONFIG = {
    'processed_dir': r"d:\al project\processed_v2",
    'n_channels': 19,
    'n_features': 32,       # Reduced from 64
    'batch_size': 128,      # Larger batch for stability
    'lr': 0.002,            # Higher initial LR with OneCycle
    'epochs': 80,
    'dropout': 0.5,         # Aggressive dropout
    'grad_clip': 0.5,
    'weight_decay': 5e-3,   # Stronger L2 regularization
    'mixup_alpha': 0.4,     # Mixup augmentation
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'model_save_path': "mtdnet_v3_best.pth"
}


class MTDNetV3(nn.Module):
    """
    MTDNet V3 - Optimized for generalization.
    Simpler than V2 but with stronger regularization.
    """
    def __init__(self, n_channels=19, n_features=32, n_classes=2, dropout=0.5):
        super(MTDNetV3, self).__init__()
        
        # Scale 1: Coarse (kernel=10, stride=10)
        self.scale1 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=10, stride=10),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Scale 2: Medium (kernel=5, stride=5 + AvgPool)
        self.scale2 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=5, stride=5),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        # Scale 3: Fine (kernel=2, stride=2 + AvgPool)
        self.scale3 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=2, stride=2),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AvgPool1d(kernel_size=5, stride=5)
        )
        
        # Unidirectional LSTM (simpler, less overfitting)
        self.lstm = nn.LSTM(
            input_size=n_features * 3,
            hidden_size=32,
            num_layers=1,       # Single layer
            batch_first=True,
            dropout=0
        )
        self.lstm_dropout = nn.Dropout(dropout)
        
        # Simple but effective classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        f1 = self.scale1(x)
        f2 = self.scale2(x)
        f3 = self.scale3(x)
        
        # Temporal alignment
        min_len = min(f1.size(2), f2.size(2), f3.size(2))
        f1 = f1[:, :, :min_len]
        f2 = f2[:, :, :min_len]
        f3 = f3[:, :, :min_len]
        
        features = torch.cat([f1, f2, f3], dim=1)
        features = features.transpose(1, 2)
        
        out, _ = self.lstm(features)
        last_out = self.lstm_dropout(out[:, -1, :])
        
        return self.classifier(last_out)


class NPZDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.from_numpy(x_data).float()
        self.y = torch.from_numpy(y_data).long()
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation: blends pairs of samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_all_npz(data_dir):
    all_x, all_y, all_groups = [], [], []
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    print(f"Loading {len(files)} subjects from disk...")
    for i, f in enumerate(files):
        data = np.load(f)
        x = data['x']
        y_val = int(data['y'])
        all_x.append(x)
        all_y.extend([y_val] * len(x))
        all_groups.extend([i] * len(x))
    X = np.concatenate(all_x, axis=0)
    y = np.array(all_y)
    groups = np.array(all_groups)
    return X, y, groups


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, use_mixup=True):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        if use_mixup:
            mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, CONFIG['mixup_alpha'])
            optimizer.zero_grad()
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        
        loss.backward()
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
    print("  MTDNet V3 Training - Anti-Overfitting Expert Config")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    
    if not os.path.exists(CONFIG['processed_dir']):
        print(f"ERROR: No data at {CONFIG['processed_dir']}")
        return
    
    X, y, groups = load_all_npz(CONFIG['processed_dir'])
    n_subjects = len(np.unique(groups))
    print(f"Dataset: {len(X)} segments from {n_subjects} subjects")
    print(f"  AD: {np.sum(y == 1)} | HC: {np.sum(y == 0)}")
    print(f"  Segment shape: {X[0].shape}")
    
    # Subject-independent split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    
    train_ds = NPZDataset(X[train_idx], y[train_idx])
    val_ds = NPZDataset(X[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], num_workers=0)
    
    model = MTDNetV3(
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
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    # OneCycleLR: proven for fast convergence with good generalization
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['lr'],
        epochs=CONFIG['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_f1 = 0
    patience_counter = 0
    patience = 25
    
    print("\n" + "-" * 70)
    print(f"{'Ep':>3} | {'TrLoss':>7} | {'TrF1':>6} | {'VlF1':>6} | {'VlAcc':>6} | {'LR':>10} | Status")
    print("-" * 70)
    
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion,
            CONFIG['device'], CONFIG['grad_clip'], use_mixup=True
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, CONFIG['device'])
        
        current_lr = optimizer.param_groups[0]['lr']
        
        status = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            status = f"*** BEST ({best_f1:.4f}) ***"
        else:
            patience_counter += 1
        
        print(f"{epoch+1:>3} | {train_loss:>7.4f} | {train_f1:>6.4f} | {val_f1:>6.4f} | {val_acc:>6.4f} | {current_lr:>10.6f} | {status}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}. Best Val F1: {best_f1:.4f}")
            break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION")
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
    print(f"FINAL Accuracy: {accuracy_score(all_targets, all_preds):.4f}")
    print(f"FINAL F1 (Macro): {f1_score(all_targets, all_preds, average='macro'):.4f}")
    print(f"\nModel saved to: {CONFIG['model_save_path']}")


if __name__ == "__main__":
    main()
