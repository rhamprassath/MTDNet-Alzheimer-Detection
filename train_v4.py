"""
MTDNet V4 Training - Proven Strategy with Enhanced Architecture
Key approach:
- Uses the V3 architecture (simpler LSTM, strong dropout)
- ReduceLROnPlateau scheduler (proven in V1)
- Standard training (no Mixup) for cleaner signal learning
- Higher initial LR (0.001)
- 4-second segments from processed_v2
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
    'n_features': 32,
    'batch_size': 64,
    'lr': 0.001,
    'epochs': 100,
    'dropout': 0.4,
    'grad_clip': 1.0,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'model_save_path': "mtdnet_v4_best.pth"
}


class MTDNetV4(nn.Module):
    """Balanced model: not too complex, not too simple."""
    def __init__(self, n_channels=19, n_features=32, n_classes=2, dropout=0.4):
        super(MTDNetV4, self).__init__()
        
        self.scale1 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=10, stride=10),
            nn.BatchNorm1d(n_features), nn.ReLU(), nn.Dropout(dropout)
        )
        self.scale2 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=5, stride=5),
            nn.BatchNorm1d(n_features), nn.ReLU(), nn.Dropout(dropout),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        self.scale3 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=2, stride=2),
            nn.BatchNorm1d(n_features), nn.ReLU(), nn.Dropout(dropout),
            nn.AvgPool1d(kernel_size=5, stride=5)
        )
        
        self.lstm = nn.LSTM(
            input_size=n_features * 3,
            hidden_size=24,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.lstm_drop = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(24, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        f1 = self.scale1(x)
        f2 = self.scale2(x)
        f3 = self.scale3(x)
        
        min_len = min(f1.size(2), f2.size(2), f3.size(2))
        f1, f2, f3 = f1[:,:,:min_len], f2[:,:,:min_len], f3[:,:,:min_len]
        
        features = torch.cat([f1, f2, f3], dim=1).transpose(1, 2)
        out, _ = self.lstm(features)
        last_out = self.lstm_drop(out[:, -1, :])
        return self.classifier(last_out)


class NPZDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.from_numpy(x_data).float()
        self.y = torch.from_numpy(y_data).long()
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


def load_all_npz(data_dir):
    all_x, all_y, all_groups = [], [], []
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    print(f"Loading {len(files)} subjects...")
    for i, f in enumerate(files):
        data = np.load(f)
        x = data['x']
        y_val = int(data['y'])
        all_x.append(x)
        all_y.extend([y_val] * len(x))
        all_groups.extend([i] * len(x))
    return np.concatenate(all_x), np.array(all_y), np.array(all_groups)


def train_epoch(model, loader, optimizer, criterion, device, clip):
    model.train()
    loss_sum, preds, targets = 0, [], []
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        loss_sum += loss.item()
        preds.extend(torch.max(out, 1)[1].cpu().numpy())
        targets.extend(by.cpu().numpy())
    return loss_sum / len(loader), accuracy_score(targets, preds), f1_score(targets, preds, average='macro')


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, preds, targets = 0, [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = criterion(out, by)
            loss_sum += loss.item()
            preds.extend(torch.max(out, 1)[1].cpu().numpy())
            targets.extend(by.cpu().numpy())
    return loss_sum / len(loader), accuracy_score(targets, preds), f1_score(targets, preds, average='macro')


def main():
    print("=" * 60)
    print("  MTDNet V4 - Proven Strategy + Enhanced Architecture")
    print("=" * 60)
    
    X, y, groups = load_all_npz(CONFIG['processed_dir'])
    n_sub = len(np.unique(groups))
    print(f"Dataset: {len(X)} segs, {n_sub} subjects (AD:{np.sum(y==1)}, HC:{np.sum(y==0)})")
    
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    tr_idx, vl_idx = next(gss.split(X, y, groups))
    print(f"Train: {len(tr_idx)} | Val: {len(vl_idx)}")
    
    tr_loader = DataLoader(NPZDataset(X[tr_idx], y[tr_idx]), batch_size=CONFIG['batch_size'], shuffle=True)
    vl_loader = DataLoader(NPZDataset(X[vl_idx], y[vl_idx]), batch_size=CONFIG['batch_size'])
    
    model = MTDNetV4(CONFIG['n_channels'], CONFIG['n_features'], 2, CONFIG['dropout']).to(CONFIG['device'])
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Class weights
    _, counts = np.unique(y[tr_idx], return_counts=True)
    w = 1.0 / counts; w = w / w.sum() * 2.0
    cw = torch.FloatTensor(w).to(CONFIG['device'])
    print(f"Class Weights: {w}")
    
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=True)
    
    best_f1 = 0
    no_improve = 0
    
    print(f"\n{'Ep':>3} | {'TrLoss':>7} | {'TrF1':>6} | {'VlF1':>6} | {'VlAcc':>6} | Status")
    print("-" * 55)
    
    for ep in range(CONFIG['epochs']):
        tl, ta, tf = train_epoch(model, tr_loader, optimizer, criterion, CONFIG['device'], CONFIG['grad_clip'])
        vl, va, vf = validate(model, vl_loader, criterion, CONFIG['device'])
        scheduler.step(vf)
        
        s = ""
        if vf > best_f1:
            best_f1 = vf
            no_improve = 0
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            s = f"*** BEST ({best_f1:.4f}) ***"
        else:
            no_improve += 1
        
        print(f"{ep+1:>3} | {tl:>7.4f} | {tf:>6.4f} | {vf:>6.4f} | {va:>6.4f} | {s}")
        
        if no_improve >= 20:
            print(f"Early stop at {ep+1}. Best F1: {best_f1:.4f}")
            break
    
    # Final
    print("\n" + "=" * 60)
    model.load_state_dict(torch.load(CONFIG['model_save_path'], map_location=CONFIG['device']))
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for bx, by in vl_loader:
            bx = bx.to(CONFIG['device'])
            ps.extend(torch.max(model(bx), 1)[1].cpu().numpy())
            ts.extend(by.numpy())
    print(classification_report(ts, ps, target_names=['HC', 'AD']))
    print(f"Final Acc: {accuracy_score(ts, ps):.4f} | F1: {f1_score(ts, ps, average='macro'):.4f}")
    print(f"Saved: {CONFIG['model_save_path']}")

if __name__ == "__main__":
    main()
