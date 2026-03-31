import streamlit as st
import numpy as np
import torch
import time
import os
import glob
from predict import AlzheimerPredictor

# --- UI Setup ---
st.set_page_config(page_title="MTDNet - Alzheimer's Detection", page_icon="🧠", layout="wide")

st.title("🧠 MTDNet: Early Alzheimer's Risk Detection")
st.markdown("""
This system uses a **Multiscale Temporal Deep Network (MTDNet)** to analyze EEG signals for early detection of Alzheimer's Disease (AD) versus Healthy Controls (HC).
""")

# Sidebar settings
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Input Source", ["Simulated Stream", "Real Subject Data"])
strategy = st.sidebar.selectbox("Evaluation Strategy", ["average", "consensus"], help="Score Averaging is generally more robust.")

# --- Predictor Setup ---
predictor = AlzheimerPredictor()

# Use V4 data (4-second segments, higher confidence)
processed_dir = r"d:\al project\processed_v2"
if not os.path.exists(processed_dir):
    processed_dir = r"d:\al project\processed_data"  # Fallback to V1
selected_subject = None

if mode == "Real Subject Data":
    if os.path.exists(processed_dir):
        npz_files = glob.glob(os.path.join(processed_dir, "*.npz"))
        if npz_files:
            subject_ids = [os.path.basename(f).replace('.npz', '') for f in npz_files]
            selected_id = st.sidebar.selectbox("Select Subject", subject_ids)
            selected_subject = os.path.join(processed_dir, f"{selected_id}.npz")
        else:
            st.sidebar.warning("No preprocessed data found. Please run preprocessing first.")
    else:
        st.sidebar.warning(f"Directory {processed_dir} not found.")

# --- Simulation & Detection ---
col1, col2 = st.columns([2, 1])

with col1:
    if mode == "Simulated Stream":
        st.subheader("📊 Live EEG Stream Simulation")
        simulation_len = st.slider("Recording Length (seconds)", 5, 60, 10)
        sampling_rate = 128
        
        if st.button("Start New Recording Session"):
            st.info(f"Simulating {simulation_len} seconds of 19-channel EEG...")
            progress_bar = st.progress(0)
            chart_holder = st.empty()
            full_recording = []
            
            for i in range(simulation_len):
                t = np.linspace(0, 1, sampling_rate)
                mock_data = np.zeros((19, sampling_rate))
                for ch in range(19):
                    noise = 0.5 * np.sin(2 * np.pi * 5 * t + np.random.randn()) + 0.1 * np.random.randn(sampling_rate)
                    mock_data[ch] = noise
                full_recording.append(mock_data)
                viz_data = np.concatenate(full_recording[-2:], axis=1) if len(full_recording) > 1 else mock_data
                chart_holder.line_chart(viz_data.T)
                progress_bar.progress((i + 1) / simulation_len)
                time.sleep(0.05)
            
            st.session_state['eeg_data'] = np.concatenate(full_recording, axis=1)
            st.session_state['data_type'] = "simulated"
            st.success("Simulation Complete!")
            
    else: # Real Subject Data
        st.subheader("🧪 Real Subject Data Analysis")
        if selected_subject:
            data = np.load(selected_subject)
            # x is (n_segments, channels, samples)
            segments = data['x']
            y_true = int(data['y'])
            st.info(f"Subject **{selected_id}** loaded. Contains {len(segments)} segments.")
            
            # Show a sample segment
            st.markdown("**Sample EEG Segment (1 second):**")
            st.line_chart(segments[0].T)
            
            if st.button("Prepare for Analysis"):
                st.session_state['segments'] = segments
                st.session_state['y_true'] = y_true
                st.session_state['data_type'] = "real"
                st.success(f"Data for {selected_id} ready for MTDNet analysis.")

with col2:
    st.subheader("🏥 Diagnostic Analysis")
    if 'data_type' in st.session_state:
        if st.button("Analyze with MTDNet"):
            with st.spinner("Processing Multiscale Temporal Features..."):
                if st.session_state['data_type'] == "simulated":
                    result = predictor.diagnose(st.session_state['eeg_data'], 128, strategy=strategy)
                else:
                    # For real data, we already have segments
                    probs, labels = predictor.predict_segments(st.session_state['segments'])
                    if strategy == 'consensus':
                        idx, conf = predictor.patient_level_consensus(labels)
                    else:
                        idx, conf = predictor.patient_level_average(probs)
                    result = {
                        "diagnosis": predictor.class_names[idx],
                        "confidence": f"{conf * 100:.2f}%",
                        "segments_analyzed": len(labels),
                        "strategy": strategy,
                        "true_label": predictor.class_names[st.session_state['y_true']] if 'y_true' in st.session_state else "Unknown"
                    }
                
                # UI Display
                st.metric("Final Diagnosis", result["diagnosis"])
                st.metric("Confidence Level", result["confidence"])
                if "true_label" in result:
                    st.write(f"**Ground Truth (Dataset):** {result['true_label']}")
                st.write(f"Analyzed {result['segments_analyzed']} segments using {strategy.capitalize()}.")
    else:
        st.write("Please load or simulate data first.")

# Footer info
st.divider()
st.markdown("""
**System Architecture Details:**
- **Model**: MTDNet Ensemble (V1 + V4) with Score Averaging.
- **V1**: ~20.5K params, 1s segments | **V4**: ~27.6K params, 4s segments.
- **Processing**: 0.5-45Hz Filter → Segments → Z-Score Normalization.
- **Subject-Level Accuracy**: 89.23% on balanced 65-subject cohort.
""")
