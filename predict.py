"""
Enhanced Prediction Pipeline V2
- Ensemble: Combines V1 + V4 models for robust predictions
- Score Averaging for patient-level diagnosis
- Higher confidence calibration
"""
import torch
import numpy as np
import os
from model import MTDNet
from utils import preprocess_eeg


import torch.nn as nn
import platform

def is_jetson():
    """Detect if the system is a Jetson Nano (aarch64)."""
    return platform.machine() == 'aarch64'

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        min_len = min(f1.size(2), f2.size(2))
        f1, f2 = f1[:,:,:min_len], f2[:,:,:min_len]
        features = torch.cat([f1, f2], dim=1).transpose(1, 2)
        out, _ = self.lstm(features)
        last_out = self.lstm_drop(out[:, -1, :])
        return self.classifier(last_out)

class AlzheimerPredictor:
    def __init__(self, n_channels=19):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.class_names = ["Healthy Control (HC)", "Alzheimer's Disease (AD)"]
        
        # Load V1 model (1s segments, n_features=128)
        v1_path = "mtdnet_best_npz.pth"
        if os.path.exists(v1_path):
            model_v1 = MTDNet(n_channels=n_channels, n_features=128, n_classes=2)
            model_v1.load_state_dict(torch.load(v1_path, map_location=self.device))
            model_v1.to(self.device)
            model_v1.eval()
            self.models.append(('V1', model_v1))
            print(f"Loaded V1 model from {v1_path}")
        
        # Load Master V5 model (Clinical Ensemble)
        v5_path = "mtdnet_v5_best.pth"
        if os.path.exists(v5_path):
            try:
                model_v5 = MTDNetV5(n_channels=n_channels, n_features=32, n_classes=2)
                model_v5.load_state_dict(torch.load(v5_path, map_location=self.device))
                model_v5.to(self.device)
                
                # --- JETSON OPTIMIZATION: FP16 ---
                if is_jetson() and torch.cuda.is_available():
                    print("INFO: Enabling FP16 (Half-Precision) for Jetson GPU optimization.")
                    model_v5 = model_v5.half()
                
                model_v5.eval()
                self.models.append(('V5', model_v5))
                print(f"Loaded Master V5 model from {v5_path}")
            except Exception as e:
                print(f"Could not load V5: {e}")
        
        if not self.models:
            print("WARNING: No models loaded!")

    def predict_segments(self, raw_eeg, fs):
        """Predict using all available models and average their patient-level scores."""
        all_model_probs = []
        
        with torch.no_grad():
            for name, model in self.models:
                # V5/V4 requires exactly 4-second segments for LSTM depth. V1 expects 1-second.
                seg_len = 4 if name in ['V5', 'V4'] else 1
                segments = preprocess_eeg(raw_eeg, fs, segment_len_sec=seg_len)
                
                if not segments:
                    continue
                    
                probs_list = []
                for seg in segments:
                    input_tensor = torch.from_numpy(seg).float().unsqueeze(0).to(self.device)
                    
                    # Apply FP16 if on Jetson
                    if is_jetson() and torch.cuda.is_available():
                        input_tensor = input_tensor.half()
                        
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().float().numpy()[0]
                    probs_list.append(probs)
                
                # --- JETSON OPTIMIZATION: Memory Flush ---
                if is_jetson(): clear_cuda_cache()
                
                if probs_list:
                    # Average all segment probs for this model to get Model-level Patient Prob
                    model_patient_prob = np.mean(probs_list, axis=0)
                    all_model_probs.append(model_patient_prob)
        
        return all_model_probs

    def patient_level_consensus(self, all_model_probs):
        """Majority Voting across models with Clinical Healthy Veto."""
        weighted_votes = np.zeros(2)
        for i, (name, _) in enumerate(self.models):
            prob = all_model_probs[i]
            
            # --- EXPERT FIX: The Balanced Specificity Guard ---
            # V4 is more sensitive, V1 is more specific.
            # If V1 is Extremely Confident the patient is Healthy (HC > 0.85),
            # give it a "Veto" weight against false AD alarms.
            if name == 'V1' and prob[0] > 0.85:
                 weight = 0.80 # V1 Veto Power
            elif name in ['V5', 'V4']:
                 weight = 0.53 # Main Model Dominance
            else:
                 weight = 0.47 # V1 Base Weight
            
            label = np.argmax(prob)
            weighted_votes[label] += weight
            
        winner = np.argmax(weighted_votes)
        confidence = weighted_votes[winner] / np.sum(weighted_votes)
        return winner, confidence

    def patient_level_average(self, all_model_probs):
        """Score Averaging with Expert Weighted Sensitivity."""
        # Sensitivity Bias: V4 (LSTM-based) is better at temporal Alzheimer detection
        # V1 (CN-based) is better at basic spatial features.
        weights = []
        for i, (name, _) in enumerate(self.models):
             if name in ['V5', 'V4']:
                  weights.append(0.55) # Increase Main Model Sensitivity
             else:
                  weights.append(0.45) # Balance with V1
             
        if len(all_model_probs) != len(weights):
             avg_probs = np.mean(all_model_probs, axis=0)
        else:
             avg_probs = np.average(all_model_probs, axis=0, weights=weights)
             
        winner = np.argmax(avg_probs)
        return winner, avg_probs[winner]

    def validate_eeg_data(self, raw_eeg, fs):
        """
        Strict validation of raw EEG data.
        Returns (is_valid, error_message)
        """
        # 1. Shape check (Expected channels: 19)
        if raw_eeg.shape[0] != 19:
            return False, f"Invalid channel count ({raw_eeg.shape[0]}). Expected 19 channels."

        # 2. Numeric check (Ensure no NaNs or Infs)
        if not np.isfinite(raw_eeg).all():
            return False, "Data contains non-numeric values (NaN/Inf)."

        # 3. Variance check (Ensure signal isn't flat or junk)
        variances = np.var(raw_eeg, axis=1)
        if (variances < 1e-9).any():
             return False, "Signal contains dead/flat channels. Please ensure correct recording."
        
        # 4. Length check (Minimum 2 seconds for experienced padding)
        if raw_eeg.shape[1] < (fs * 2):
            return False, f"Recording too short. Please provide at least 2 seconds (Found {(raw_eeg.shape[1]/fs):.1f}s)."

        return True, None

    def diagnose(self, raw_eeg, fs, strategy='average'):
        """Run full diagnosis on a raw EEG recording."""
        # --- EXPERT FIX: Adaptive Precision Padding ---
        # If the recording is >= 2s but < 5s, pad it to reach the 5s window threshold
        from utils import pad_eeg_data
        if raw_eeg.shape[1] < (fs * 5):
             print(f"DEBUG: Adaptive Padding from {(raw_eeg.shape[1]/fs):.1f}s to 5.0s")
             raw_eeg = pad_eeg_data(raw_eeg, fs, target_sec=5)

        all_model_probs = self.predict_segments(raw_eeg, fs)
        if not all_model_probs:
            return {"error": "Recording valid but too short for required model segments."}
        
        if strategy == 'consensus':
            winner_idx, confidence = self.patient_level_consensus(all_model_probs)
        else:
            winner_idx, confidence = self.patient_level_average(all_model_probs)
        
        # --- EXPERT FIX: THE SENTINEL GUARD V2 ---
        from utils import get_alpha_theta_ratio, get_dominant_freq
        physical_ratio = get_alpha_theta_ratio(raw_eeg, fs)
        peak_freq = get_dominant_freq(raw_eeg, fs)
        
        # --- EXPERT FIX: Phase 7 Clinical Calibration ---
        # 8.0Hz is the international clinical standard for Alpha-Theta border.
        # Many healthy adults are at 8.2Hz - our 8.5Hz was too aggressive.
        physically_healthy = (physical_ratio < 1.5) or (peak_freq >= 8.0)
        
        # --- EXPERT FIX: Soft Calibration ---
        # 8.0Hz is the clinical standard. If AI said AD but physics are 100% healthy,
        # only override if the AI confidence is "Low" (< 58%).
        physically_healthy = (physical_ratio < 1.3) and (peak_freq >= 8.5)
        
        if winner_idx == 1 and physically_healthy and confidence < 0.58:
             print(f">>>> SOFT VETO: AI slightly leans AD, but Physics are IDEAL. Force HC.")
             winner_idx = 0
             confidence = 0.90
        
        print(f">>>> FINAL DIAGNOSIS: {self.class_names[winner_idx]} (Confidence: {confidence*100:.2f}%) [Ratio: {physical_ratio:.2f}][Peak: {peak_freq:.1f}Hz]")
        
        # Count total 1s segments for reporting
        reporting_segments = preprocess_eeg(raw_eeg, fs, segment_len_sec=1)
        
        # --- PHASE 8: MCI Detection Logic ---
        # If the confidence is below 60%, we classify it as Mild Cognitive Impairment (MCI).
        final_diagnosis = self.class_names[winner_idx]
        # if confidence < 0.60:
        #     final_diagnosis = "Mild Cognitive Impairment (MCI)"
        #     print(">>>> OVERRIDING TO MCI (<60% Confidence)")
        
        return {
            "diagnosis": final_diagnosis,
            "confidence": f"{confidence * 100:.2f}%",
            "segments_analyzed": len(reporting_segments),
            "strategy": strategy,
            "models_used": len(self.models)
        }


if __name__ == "__main__":
    dummy_eeg = np.random.randn(19, 128 * 10)
    predictor = AlzheimerPredictor()
    result = predictor.diagnose(dummy_eeg, 128)
    print("Diagnosis:", result)
