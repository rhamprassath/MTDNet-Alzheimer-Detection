import os
import glob
import time
import tempfile
import mne
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import resample
from pydantic import BaseModel
from typing import Optional
import numpy as np
from predict import AlzheimerPredictor
import json
from huggingface_hub import HfApi

app = FastAPI(title="MTDNet Alzheimer's Detection API")

# Setup CORS for the mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor
predictor = AlzheimerPredictor()

STANDARD_10_20 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

BRAIN_HEALTH_TIPS = {
    "Healthy Control (HC)": [
        "Stay Mentally Active: Engage in puzzles, reading, or learning new skills daily.",
        "Physical Exercise: Regular aerobic exercise improves blood flow to the brain.",
        "Healthy Diet: Follow a Mediterranean diet rich in antioxidants and Omega-3.",
        "Quality Sleep: Aim for 7–9 hours of sleep to help brain detoxification."
    ],
    "Alzheimer's Disease (AD)": [
        "Social Engagement: Regular interaction helps slow cognitive decline.",
        "Cognitive Training: Use memory-focused apps or games daily.",
        "Stress Management: Practice mindfulness or light yoga to reduce cortisol levels.",
        "Consistent Routine: Maintain a structured daily schedule to reduce confusion."
    ],
    "Mild Cognitive Impairment (MCI)": [
        "Monitor Diet: Focus heavily on B-vitamins and Omega-3 rich foods.",
        "Medical Consultation: Regular check-ins with a neurologist to monitor progression.",
        "Daily Physical Activity: Regular walks to boost neurogenesis.",
        "Reduce Toxins: Limit alcohol and quit smoking entirely to protect brain cells."
    ]
}

# Data directory
processed_dir = os.path.join(os.path.dirname(__file__), "processed_v2")
os.makedirs(processed_dir, exist_ok=True)

def seed_demo_subjects():
    """Auto-generate realistic synthetic demo subjects if the database is empty."""
    existing = glob.glob(os.path.join(processed_dir, "*.npz"))
    if existing:
        print(f"INFO: Database has {len(existing)} subjects. Skipping seed.")
        return

    print("INFO: Database empty. Auto-seeding demo subjects...")
    FS = 128
    DURATION = 30  # seconds of EEG per subject
    SEG_LEN = 4    # 4-second segments
    N_SEGS = DURATION // SEG_LEN
    N_CH = 19
    SAMPLES = SEG_LEN * FS

    # Subject definitions: (id, label, dominant_freq_hz, noise_scale)
    subjects = [
        # Healthy Controls - strong alpha (8-12Hz), low noise
        ("HC_Subject_01", 0, 10.0, 0.3),
        ("HC_Subject_02", 0, 9.5,  0.25),
        ("HC_Subject_03", 0, 11.0, 0.2),
        # Mild Cognitive Impairment - mixed theta/alpha
        ("MCI_Subject_01", 1, 6.5, 0.6),
        ("MCI_Subject_02", 1, 7.0, 0.55),
        ("MCI_Subject_03", 1, 5.5, 0.65),
        # Alzheimer's Disease - dominant theta/delta, high noise
        ("AD_Subject_01", 1, 4.0, 1.1),
        ("AD_Subject_02", 1, 3.5, 1.2),
        ("AD_Subject_03", 1, 5.0, 0.9),
    ]

    t = np.linspace(0, SEG_LEN, SAMPLES)
    for sub_id, label, dom_freq, noise_sc in subjects:
        segs = []
        for _ in range(N_SEGS):
            seg = np.zeros((N_CH, SAMPLES), dtype=np.float32)
            for ch in range(N_CH):
                phase = np.random.uniform(0, 2 * np.pi)
                amp = np.random.uniform(0.8, 1.2)
                # Dominant rhythm + harmonics + noise
                sig = (amp * np.sin(2 * np.pi * dom_freq * t + phase)
                       + 0.3 * np.sin(2 * np.pi * (dom_freq * 2) * t + phase)
                       + noise_sc * np.random.randn(SAMPLES))
                # Z-score normalize per channel
                sig = (sig - sig.mean()) / (sig.std() + 1e-8)
                seg[ch] = sig.astype(np.float32)
            segs.append(seg)
        block = np.array(segs, dtype=np.float32)
        out_path = os.path.join(processed_dir, f"{sub_id}.npz")
        np.savez(out_path, x=block, y=label)
        print(f"INFO: Seeded {sub_id} (label={label})")

    print("INFO: Demo subject database ready with 9 subjects.")

# Seed on startup
seed_demo_subjects()

class SimulationRequest(BaseModel):
    simulation_len_seconds: int = 10
    sampling_rate: int = 128
    strategy: str = "average"

class RealDataRequest(BaseModel):
    subject_id: str
    strategy: str = "average"

@app.get("/")
def read_root():
    """Welcome message for the MTDNet Alzheimer's Detection API."""
    return {
        "status": "online",
        "system": "MTDNet Alzheimer's Detection Engine",
        "message": "Welcome! Use /health to check system status or the mobile app for diagnostics."
    }

@app.get("/health")
def health_check():
    """Check if the API and model are loaded correctly."""
    return {"status": "ok", "message": "MTDNet API is running and model is loaded."}

@app.get("/subjects")
def get_subjects():
    """Get a list of available preprocessed subjects."""
    if not os.path.exists(processed_dir):
        return {"subjects": []}
    
    npz_files = glob.glob(os.path.join(processed_dir, "*.npz"))
    subject_ids = [os.path.basename(f).replace('.npz', '') for f in npz_files]
    return {"subjects": subject_ids}

@app.post("/evaluate/simulated")
def evaluate_simulated(request: SimulationRequest):
    """Run a simulated 19-channel EEG stream and evaluate it."""
    expected_channels = 19
    total_samples = request.simulation_len_seconds * request.sampling_rate
    
    # Generate mock data
    t = np.linspace(0, request.simulation_len_seconds, total_samples)
    mock_data = np.zeros((expected_channels, total_samples))
    for ch in range(expected_channels):
        noise = 0.5 * np.sin(2 * np.pi * 5 * t + np.random.randn()) + 0.1 * np.random.randn(total_samples)
        mock_data[ch] = noise

    start_time = time.time()
    try:
        result = predictor.diagnose(mock_data, request.sampling_rate, strategy=request.strategy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    eval_time = time.time() - start_time
    
    return {
        "diagnosis": result["diagnosis"],
        "confidence": result["confidence"],
        "segments_analyzed": result["segments_analyzed"],
        "strategy": result["strategy"],
        "evaluation_time_seconds": round(eval_time, 2)
    }

@app.post("/evaluate/real")
def evaluate_real(request: RealDataRequest):
    """Evaluate preprocessed real subject data."""
    print(f"INFO: [evaluate/real] Request for Subject: {request.subject_id}, Strategy: {request.strategy}")
    subject_file = os.path.join(processed_dir, f"{request.subject_id}.npz")
    if not os.path.exists(subject_file):
        raise HTTPException(status_code=404, detail="Subject data not found.")
    
    # Load data
    data = np.load(subject_file)
    segments = data['x'] # (n_segments, 19, 512)
    y_true = int(data['y'])

    start_time = time.time()
    try:
        # Reconstruct raw EEG from segments for the ensemble predictor
        # Original shape: (n_segments, channels, samples_per_segment)
        # Target shape: (channels, total_samples)
        raw_eeg = np.transpose(segments, (1, 0, 2)).reshape(19, -1)
        
        # Use the high-level diagnose method which handles V1+V4 ensemble
        result = predictor.diagnose(raw_eeg, fs=128, strategy=request.strategy)
        
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
             
        eval_time = time.time() - start_time
        
        # Safely handle MCI mapping for binary accuracy
        pred_diag = result["diagnosis"]
        if pred_diag == "Mild Cognitive Impairment (MCI)":
            pred_idx = 1 # Treat leaning as AD for binary assessment
        else:
            try:
                pred_idx = predictor.class_names.index(pred_diag)
            except ValueError:
                pred_idx = -1
                
        return {
            "diagnosis": pred_diag,
            "confidence": result["confidence"],
            "segments_analyzed": result["segments_analyzed"],
            "strategy": result["strategy"],
            "true_label": predictor.class_names[y_true],
            "evaluation_time_seconds": round(eval_time, 2),
            "is_correct": bool(pred_idx == y_true),
            "tips": BRAIN_HEALTH_TIPS.get(pred_diag, BRAIN_HEALTH_TIPS["Healthy Control (HC)"])
        }
    except Exception as e:
        import traceback
        print(f"REAL SUBJECT EVALUATION ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis Engine Error: {str(e)}")

@app.post("/evaluate/upload")
async def evaluate_upload(
    file: UploadFile = File(...), 
    sampling_rate: Optional[int] = Form(None),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[str] = Form(None),
    patient_notes: Optional[str] = Form(None)
):
    """Evaluate an uploaded raw .edf, .csv, .set, or .npz file with automated HF Persistence."""
    print(f"INFO: [evaluate/upload] Received File: {file.filename}, Size: {file.size if hasattr(file, 'size') else 'Unknown'} bytes")
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    
    # Check file extension before saving/processing
    filename = file.filename.lower()
    allowed_extensions = ['.edf', '.csv', '.set', '.npz']
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {filename}. Please upload .edf, .csv, .set, or .npz.")

    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        start_time = time.time()
        
        try:
            # Load raw data
            if temp_path.endswith('.edf'):
                raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
                # Master-Level Upgrade: Strict Spatial Filtering
                # If >19 channels, try to pick the first 19 (typically EEG before EKG/refs)
                if len(raw.ch_names) > 19:
                    raw.pick(raw.ch_names[:19])
                fs = raw.info['sfreq']
                data = raw.get_data() # (channels, samples)
            elif temp_path.endswith('.csv'):
                df = pd.read_csv(temp_path)
                
                # --- EXPERT FIX 1: Structural CSV Sanitization ---
                drop_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['time', 'date', 'epoch', 'trigger', 'event', 'unnamed', 'id'])]
                if drop_cols:
                     df = df.drop(columns=drop_cols)
                     
                # --- EXPERT FIX 2: Spatial Channel Alignment ---
                # If the CSV columns map to the 10-20 standard system, forcefully re-order them.
                def find_match(target):
                    target_upper = target.upper()
                    for col in df.columns:
                        col_u = col.upper().replace("."," ").replace("-"," ").replace("_"," ").strip()
                        tokens = col_u.split()
                        # Match exact, or if the target is one of the tokens (e.g. "EEG Fp1" matches "Fp1")
                        if target_upper == col_u or target_upper in tokens:
                            return col
                    return None
                    
                matched_cols = [find_match(ch) for ch in STANDARD_10_20]
                matched_cols_filtered = [c for c in matched_cols if c is not None]
                
                # --- EXPERT FIX 2: Spatial Fallback ---
                # If we failed to find 19 matches via name, but the file has EXACTLY 19 columns, 
                # assume they are the 10-20 channels in order as a safety fallback.
                if len(matched_cols_filtered) < 19 and df.shape[1] == 19:
                     print(f"DEBUG: Found {len(matched_cols_filtered)} matches, but file has exactly 19 channels. Using Direct Mode.")
                     matched_cols_filtered = list(df.columns)
                
                print(f"DEBUG: Found {len(matched_cols_filtered)} matches out of 19.")
                if len(matched_cols_filtered) == 19:
                     print(f"DEBUG: Successfully alignment [19/19] channels: {matched_cols_filtered}")
                     df = df[matched_cols_filtered] # Perfectly reconstruct the "scrambled brain"
                else:
                     print(f"WARNING: Only found {len(matched_cols_filtered)} channels. Defaulting to first 19 columns.")

                # Filter for numeric columns only
                numeric_df = df.select_dtypes(include=[np.number])
                
                if numeric_df.shape[1] >= 19:
                     data = numeric_df.iloc[:, :19].values.T
                else:
                     data = numeric_df.values.T
                     
                # Use provided sampling_rate or default to 128
                fs = sampling_rate if sampling_rate is not None else 128
                if sampling_rate is None:
                    print("WARNING: [CSV] No Sampling Rate provided in Form. Defaulting to 128Hz.")
                
            elif temp_path.endswith('.set'):
                raw = mne.io.read_raw_eeglab(temp_path, preload=True, verbose=False)
                if len(raw.ch_names) > 19:
                    raw.pick(raw.ch_names[:19])
                fs = raw.info['sfreq']
                data = raw.get_data()
            elif temp_path.endswith('.npz'):
                # --- EXPERT FIX 3: Training Data Support (.npz) ---
                npz_data = np.load(temp_path, allow_pickle=True)
                keys = list(npz_data.keys())
                data_key = keys[0] # Often 'x' or 'data'
                data = npz_data[data_key]
                if len(data.shape) == 3: # (Batch, CH, Samples)? Take first
                     data = data[0]
                if data.shape[0] != 19 and data.shape[1] == 19:
                     data = data.T
                fs = sampling_rate 
            else:
                 raise HTTPException(status_code=400, detail="Unsupported file format.")
                 
            # --- EXPERT FIX 3: Frequency Resampling & Auto-Sensing ---
            if fs is None:
                 # Default to 128 if all detection fails
                 fs = sampling_rate if sampling_rate is not None else 128
                 print(f"DEBUG: No FS detected. Defaulting to: {fs}Hz")

            if fs != 128:
                print(f"DEBUG: Master-Level Resampling (Polyphase) from {fs}Hz to 128Hz...")
                from scipy.signal import resample_poly
                # Calculate up/down factors
                # target is 128. Current is fs.
                # up = 128, down = int(fs) -> simplify via gcd is handled internally by resample_poly usually, but let's be safe
                import math
                up = 128
                down = int(fs)
                gcd = math.gcd(up, down)
                data = resample_poly(data, up // gcd, down // gcd, axis=1)
                fs = 128
                
            eval_time = time.time() - start_time
            
            result = predictor.diagnose(data, fs, strategy='average')
            
            if "error" in result:
                 raise HTTPException(status_code=400, detail=result["error"])
                 
            # --- PERSIST DATA TO HUGGING FACE ---
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                try:
                    hf_api = HfApi(token=hf_token)
                    repo_id = "RHAMPRASSATH/MTDNet-Alzheimer-Detector"
                    timestamp = int(time.time())
                    safe_name = (patient_name or "unknown").replace(" ", "_").lower()
                    folder = f"patient_data/{safe_name}_{timestamp}"
                    
                    # 1. Upload original file
                    hf_api.upload_file(
                        path_or_fileobj=temp_path,
                        path_in_repo=f"{folder}/{file.filename}",
                        repo_id=repo_id,
                        repo_type="space"
                    )
                    
                    # 2. Upload Metadata JSON
                    meta = {
                        "patient_name": patient_name,
                        "patient_age": patient_age,
                        "patient_notes": patient_notes,
                        "diagnosis": result["diagnosis"],
                        "confidence": result["confidence"],
                        "timestamp": timestamp,
                        "filename": file.filename
                    }
                    meta_path = os.path.join(temp_dir, f"meta_{timestamp}.json")
                    with open(meta_path, "w") as f:
                        json.dump(meta, f)
                    
                    hf_api.upload_file(
                        path_or_fileobj=meta_path,
                        path_in_repo=f"{folder}/metadata.json",
                        repo_id=repo_id,
                        repo_type="space"
                    )
                    print(f"INFO: Successfully persisted data to HF: {folder}")
                    if os.path.exists(meta_path):
                        os.remove(meta_path)
                except Exception as hf_err:
                    print(f"WARNING: Hugging Face Persistence Failed: {hf_err}")
            else:
                print("WARNING: HF_TOKEN not found. Persistence skipped.")

            return {
                "diagnosis": result["diagnosis"],
                "confidence": result["confidence"],
                "segments_analyzed": result["segments_analyzed"],
                "strategy": result["strategy"],
                "evaluation_time_seconds": round(eval_time, 2),
                "tips": BRAIN_HEALTH_TIPS.get(result["diagnosis"], BRAIN_HEALTH_TIPS["Healthy Control (HC)"])
            }
        except Exception as e:
            import traceback
            print(f"DIAGNOSTIC ENGINE CRASH: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Diagnostic Error: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating file: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

