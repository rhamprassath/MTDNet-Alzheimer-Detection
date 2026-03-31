# MTDNet: Alzheimer's Disease Diagnostic System

MTDNet is a clinical-grade EEG analysis system designed for the detection of Alzheimer's Disease (AD) using advanced dry-electrode technology and deep learning ensembles.

## 🚀 Key Features

- **Model Ensemble (V1 + V4)**: Combines high-specificity (V1) and high-sensitivity (V4) models for a balanced diagnostic output.
- **Physical Sentinel Veto**: Implements an 8.0Hz Alpha-peak guard to eliminate false positives in healthy patients.
- **Auto-Sensing Ingestion**: Automatically detects sampling rates and channel layouts from EDF, SET, and CSV files.
- **Mobile-Cloud Synchronization**: Real-time analysis from a React Native mobile application connected to a high-performance FastAPI backend.

## 🛠️ Architecture

- **Backend**: Python, FastAPI, PyTorch, MNE-Python, Scipy.
- **Frontend**: React Native (Expo), Axios, SVG Waveforms.
- **AI Models**: MTDNet (Multiscale Temporal Dependency Network) with optimized Z-score normalization.

## 📁 Project Structure

- `/`: FastAPI backend, model training scripts, and diagnostic logic.
- `/mobile-app`: React Native source code for the clinical application.
- `/knowledge`: Technical documentation and implementation history.

## 🚦 Getting Started

### Backend
1. Install dependencies: `pip install -r requirements.txt`
2. Start the API: `python api.py`

### Mobile App
1. Go to `mobile-app` directory.
2. Install dependencies: `npm install`
3. Start Expo: `npx expo start`

---
*Developed for clinical-grade EEG diagnostics.*
