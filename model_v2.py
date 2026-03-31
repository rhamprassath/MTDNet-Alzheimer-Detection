import torch
import torch.nn as nn

class MTDNetV2(nn.Module):
    """
    MTDNet V2 - Enhanced Multiscale Temporal Deep Network.
    Improvements over V1:
    - Increased LSTM hidden units (32 -> better temporal modeling)
    - Dropout regularization (prevents overfitting)
    - Deeper classification head with residual-style connection
    - Bidirectional LSTM for richer temporal context
    """
    def __init__(self, n_channels=19, n_features=64, n_classes=2, dropout=0.3):
        super(MTDNetV2, self).__init__()
        
        # Scale 1: Coarse temporal (Kernel 10, Stride 10)
        self.scale1 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=10, stride=10, padding=0),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Scale 2: Medium temporal (Kernel 5, Stride 5 + AvgPool)
        self.scale2 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=5, stride=5, padding=0),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        
        # Scale 3: Fine temporal (Kernel 2, Stride 2 + AvgPool)
        self.scale3 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AvgPool1d(kernel_size=5, stride=5)
        )
        
        # Bidirectional LSTM for richer temporal context
        self.lstm = nn.LSTM(
            input_size=n_features * 3,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Classification Head (enhanced)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2, 64),  # *2 for bidirectional
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # x: (batch, n_channels, n_samples)
        f1 = self.scale1(x)
        f2 = self.scale2(x)
        f3 = self.scale3(x)
        
        # Temporal alignment: ensure same sequence length
        min_len = min(f1.size(2), f2.size(2), f3.size(2))
        f1 = f1[:, :, :min_len]
        f2 = f2[:, :, :min_len]
        f3 = f3[:, :, :min_len]
        
        # Concatenate along feature dimension
        features = torch.cat([f1, f2, f3], dim=1)  # (B, n_features*3, T)
        features = features.transpose(1, 2)  # (B, T, n_features*3)
        
        # Bidirectional LSTM
        out, (h_n, c_n) = self.lstm(features)
        
        # Use last time step
        last_out = out[:, -1, :]
        
        # Classification
        logits = self.classifier(last_out)
        return logits


if __name__ == "__main__":
    # Test with 4-second segment at 128Hz = 512 samples
    model = MTDNetV2(n_channels=19, n_features=64, n_classes=2)
    dummy_input = torch.randn(8, 19, 512)
    output = model(dummy_input)
    params = sum(p.numel() for p in model.parameters())
    print(f"MTDNet V2 Parameters: {params:,}")
    print(f"Output shape: {output.shape}")
    
    # Also test with 1-second segment (128 samples) for backward compatibility
    dummy_short = torch.randn(8, 19, 128)
    output_short = model(dummy_short)
    print(f"Short segment output: {output_short.shape}")
