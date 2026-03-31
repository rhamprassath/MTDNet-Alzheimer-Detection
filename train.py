import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import MTDNet
from utils import preprocess_eeg
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Expert Configuration
CONFIG = {
    "n_channels": 19,
    "n_features": 32,
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 80,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_save_path": "mtdnet_best.pth",
    "fs": 128
}

class ADFTDDataset(Dataset):
    """
    Production-grade Dataset for ADFTD Ingestion.
    Expects directory structure:
    root/HC/*.npy or *.csv
    root/AD/*.npy or *.csv  
    """
    def __init__(self, segments, labels):
        self.segments = segments
        self.labels = labels

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # (Channels, Samples)
        x = torch.from_numpy(self.segments[idx]).float()
        y = torch.tensor(self.labels[idx]).long()
        return x, y

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    preds, targets = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        targets.extend(y.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(targets, preds)
    return epoch_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, targets = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            running_loss += loss.item() * x.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            targets.extend(y.cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    return val_loss, acc, f1

from data_loader import EEGDataLoader
from sklearn.model_selection import GroupShuffleSplit

# ... (CONFIG remains same)

def main():
    print(f"--- Starting MTDNet Expert Training Profile ---")
    print(f"Device: {CONFIG['device']}")
    
    # Data Root & TSV
    root = r"d:\al project\ADFTD"
    tsv = r"d:\al project\ADFTD_git\participants.tsv"
    
    if not os.path.exists(root) or not os.path.exists(tsv):
        print("ERROR: Dataset path not discovered. Please run download_ds.py first.")
        return

    loader = EEGDataLoader(root, tsv)
    
    print("Loading Dataset from OpenNeuro/ADFTD...")
    try:
        X, y, groups = loader.prepare_dataset(binary_ad_hc=True)
        print(f"Loaded {len(X)} segments from {len(np.unique(groups))} subjects.")
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return
    
    if len(X) == 0:
        print("Dataset empty. Ensure files are downloaded and derivatives exist.")
        return

    # Expert Splitting: Subject-Independent (GroupShuffleSplit)
    # We split by subjects (groups), not by individual segments.
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"TRAIN: {len(X_train)} segments ({len(np.unique(groups[train_idx]))} subjects)")
    print(f"VAL:   {len(X_val)} segments ({len(np.unique(groups[val_idx]))} subjects)")
    
    train_ds = ADFTDDataset(X_train, y_train)
    val_ds = ADFTDDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])
    
    # Model Initialization
    model = MTDNet(n_channels=CONFIG['n_channels'], n_features=CONFIG['n_features']).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    best_f1 = 0
    
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, CONFIG['device'])
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")
            
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            
    print(f"Training Complete. Best Validation F1: {best_f1:.4f}")
    print(f"Checkpoint saved to: {CONFIG['model_save_path']}")

if __name__ == "__main__":
    main()
