"""
MTDNet V5 Training - Master Clinical Ensemble
Key strategy:
- Trains on the V5 Overhaul Dataset (Winsorizing + Polyphase Resampling)
- Handles Extreme Class Imbalance (HC:69, AD:1200)
- Optimized for the new high-resolution clinical records
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

CONFIG = {
    'data_path': "v5_train_data.npz",
    'n_channels': 19,
    'n_features': 32,
    'batch_size': 32, # Smaller batch for more stable gradient in imbalanced data
    'lr': 0.0005,      # Lower LR for fine-tuning on high-res signal
    'epochs': 150,
    'dropout': 0.5,
    'grad_clip': 1.0,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'model_save_path': "mtdnet_v5_best.pth"
}

class MTDNetV5(nn.Module):
    """High-Stability Clinical Architecture."""
    def __init__(self, n_channels=19, n_features=32, n_classes=2, dropout=0.5):
        super(MTDNetV5, self).__init__()
        
        self.scale1 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=10, stride=5),
            nn.BatchNorm1d(n_features), nn.ReLU(), nn.Dropout(dropout)
        )
        self.scale2 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=5, stride=2),
            nn.BatchNorm1d(n_features), nn.ReLU(), nn.Dropout(dropout),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        
        self.lstm = nn.LSTM(
            input_size=n_features * 2,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.lstm_drop = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        f1 = self.scale1(x)
        f2 = self.scale2(x)
        
        # Consistent time-step alignment
        min_len = min(f1.size(2), f2.size(2))
        f1, f2 = f1[:,:,:min_len], f2[:,:,:min_len]
        
        features = torch.cat([f1, f2], dim=1).transpose(1, 2)
        out, _ = self.lstm(features)
        last_out = self.lstm_drop(out[:, -1, :])
        return self.classifier(last_out)

class V5Dataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.from_numpy(x_data).float()
        self.y = torch.from_numpy(y_data).long()
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

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
    print("  MTDNet V5 - Master Clinical Ingestion Training")
    print("=" * 60)
    
    data = np.load(CONFIG['data_path'])
    X, y = data['x'], data['y']
    print(f"Dataset Size: {len(X)} segments")
    print(f"Class Dist: HC={np.sum(y==0)}, AD={np.sum(y==1)}")
    
    # 80/20 Stratified Split
    tr_x, vl_x, tr_y, vl_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    tr_loader = DataLoader(V5Dataset(tr_x, tr_y), batch_size=CONFIG['batch_size'], shuffle=True)
    vl_loader = DataLoader(V5Dataset(vl_x, vl_y), batch_size=CONFIG['batch_size'])
    
    model = MTDNetV5(CONFIG['n_channels'], CONFIG['n_features'], 2, CONFIG['dropout']).to(CONFIG['device'])
    
    # HEAVY WEIGHTING for Imbalance Fix (HC Weight = 1200/69 ≈ 17x)
    counts = np.array([np.sum(tr_y==0), np.sum(tr_y==1)])
    weights = 1.0 / counts; weights = weights / weights.sum() * 2.0
    cw = torch.FloatTensor(weights).to(CONFIG['device'])
    print(f"Active Class Weights: {weights}")
    
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
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
        
        if no_improve >= 30: # Patient Master training
            print(f"Early stop at {ep+1}.")
            break
            
    print("\nTraining Finalized. Saved: " + CONFIG['model_save_path'])

if __name__ == "__main__":
    main()
