import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model import MTDNet
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
import glob

# Configuration
CONFIG = {
    'processed_dir': r"d:\al project\processed_data",
    'n_channels': 19,
    'n_features': 128, # fs
    'batch_size': 32,
    'lr': 0.001,
    'epochs': 100,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'model_save_path': "mtdnet_best_npz.pth"
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
    all_x = []
    all_y = []
    all_groups = []
    
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    print(f"Loading {len(files)} subjects from disk...")
    
    for i, f in enumerate(files):
        data = np.load(f)
        x = data['x']
        y = int(data['y'])
        g = int(data['group'])
        
        all_x.append(x)
        all_y.extend([y] * len(x))
        all_groups.extend([i] * len(x)) # Use file index as group id
        
    X = np.concatenate(all_x, axis=0)
    y = np.array(all_y)
    groups = np.array(all_groups)
    return X, y, groups

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())
        
    acc = accuracy_score(all_targets, all_preds)
    return total_loss / len(loader), acc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
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
    print(f"--- MTDNet Training Profile (Disk-Optimized) ---")
    
    if not os.path.exists(CONFIG['processed_dir']):
        print(f"ERROR: No preprocessed data found at {CONFIG['processed_dir']}")
        return
        
    X, y, groups = load_all_npz(CONFIG['processed_dir'])
    print(f"Dataset: {len(X)} segments from {len(np.unique(groups))} subjects.")

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    train_ds = NPZDataset(X[train_idx], y[train_idx])
    val_ds = NPZDataset(X[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])
    
    model = MTDNet(n_channels=CONFIG['n_channels'], n_features=CONFIG['n_features']).to(CONFIG['device'])
    
    # Calculate Class Weights
    labels_unique, counts = np.unique(y[train_idx], return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * 2.0
    class_weights = torch.FloatTensor(weights).to(CONFIG['device'])
    print(f"Using Class Weights: {weights} (Labels: {labels_unique})")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_f1 = 0
    
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, CONFIG['device'])
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} Acc: {val_acc:.4f}")
            
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f"Checkpoint saved with Val F1: {best_f1:.4f}")
            
    print(f"Training Finalized. Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
