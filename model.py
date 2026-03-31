import torch
import torch.nn as nn

class MTDNet(nn.Module):
    """
    Multiscale Temporal Deep Network (MTDNet) for Alzheimer's Disease Classification from EEG.
    As described in Zini et al. (2026).
    """
    def __init__(self, n_channels=19, n_features=32, n_classes=2):
        super(MTDNet, self).__init__()
        
        # Scale 1: Kernel 10, Stride 10
        self.scale1 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=10, stride=10, padding=0),
            nn.BatchNorm1d(n_features),
            nn.ReLU()
        )
        
        # Scale 2: Kernel 5, Stride 5 + AvgPool(2,2)
        self.scale2 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=5, stride=5, padding=0),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        
        # Scale 3: Kernel 2, Stride 2 + AvgPool(5,5)
        self.scale3 = nn.Sequential(
            nn.Conv1d(n_channels, n_features, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=5, stride=5)
        )
        
        # LSTM Layer
        # Input dim is n_features * 3 (concatenated scales)
        self.lstm = nn.LSTM(
            input_size=n_features * 3, 
            hidden_size=16, 
            num_layers=2, 
            batch_first=True
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, n_channels, n_samples)
        
        # Multiscale Feature Extraction
        f1 = self.scale1(x) # (B, n_features, T/10)
        f2 = self.scale2(x) # (B, n_features, T/5/2) = (B, n_features, T/10)
        f3 = self.scale3(x) # (B, n_features, T/2/5) = (B, n_features, T/10)
        
        # Concatenate features along the feature dimension
        # Shape: (B, n_features * 3, Seq_Len)
        features = torch.cat([f1, f2, f3], dim=1)
        
        # Reshape for LSTM: (Batch, Seq_Len, Features)
        features = features.transpose(1, 2)
        
        # LSTM
        # out shape: (B, Seq_Len, hidden_dim)
        # h_n shape: (num_layers, B, hidden_dim)
        out, (h_n, c_n) = self.lstm(features)
        
        # Take the last time step output
        last_out = out[:, -1, :]
        
        # Final Classification
        logits = self.classifier(last_out)
        return logits

if __name__ == "__main__":
    # Test with dummy data
    # (batch_size, n_channels, n_samples)
    # 1 second segment at 128Hz = 128 samples
    model = MTDNet(n_channels=19, n_features=32, n_classes=2)
    dummy_input = torch.randn(8, 19, 128)
    output = model(dummy_input)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Output shape: {output.shape}")
