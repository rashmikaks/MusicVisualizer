import streamlit as st
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import time
import tempfile

st.set_page_config(page_title="GPU Music Visualizer", layout="wide")
st.title("🎶 GPU Music Visualizer with Mood Analysis")

uploaded = st.file_uploader("Upload a music file", type=["wav", "mp3"])

if uploaded:
    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded.read())
        tmp_path = tmpfile.name

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read audio file
    audio_data, sr = sf.read(tmp_path, dtype="float32")
    waveform = torch.tensor(audio_data, device=device)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)  # Convert to mono

    st.audio(tmp_path, format="audio/wav")

    # --- 🔹 Mood Analysis ---
    with st.spinner("Analyzing mood..."):
        # Loudness (RMS)
        rms = torch.sqrt(torch.mean(waveform ** 2)).item()

        # Simple tempo estimate (using energy peaks)
        envelope = waveform.abs().cpu().numpy()
        onset_env = np.diff(envelope, prepend=0)
        peaks = np.where(onset_env > onset_env.mean() + 2 * onset_env.std())[0]
        if len(peaks) > 1:
            intervals = np.diff(peaks) / sr
            tempo = 60.0 / np.mean(intervals)  # BPM
        else:
            tempo = 0

        # Mood classification
        if tempo < 90 and rms < 0.02:
            mood = "Calm / Relaxing"
        elif 90 <= tempo <= 130 and rms < 0.05:
            mood = "Focus / Productivity"
        else:
            mood = "Energetic / Stressful"

    st.subheader("Mood Analysis")
    st.write(f"**Tempo (BPM):** {tempo:.2f}")
    st.write(f"**Loudness (RMS):** {rms:.4f}")
    st.success(f"**Predicted Mood:** {mood}")

    # --- 🔹 Controls for visualization ---
    col1, col2 = st.columns(2)
    play = col1.button("▶ Play")
    pause = col2.button("⏸ Pause")

    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "pos" not in st.session_state:
        st.session_state.pos = 0

    if play:
        st.session_state.playing = True
    if pause:
        st.session_state.playing = False

    # --- 🔹 Plot setup ---
    win = 2000  # Number of samples to display
    hop = 512   # Step size

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    line, = ax.plot(np.zeros(win), lw=2, color="lightgrey")
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, win)
    ax.set_title("Waveform", color="white")
    ax.set_xlabel("Samples", color="white")
    ax.set_ylabel("Amplitude", color="white")
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')

    placeholder = st.empty()
    progress_bar = st.progress(0)

    # --- 🔹 Main loop for visualization ---
    while True:
        if st.session_state.playing:
            pos = st.session_state.pos
            if pos + win >= waveform.shape[0]:
                st.session_state.playing = False
                break

            seg = waveform[pos:pos+win].cpu().numpy()
            line.set_ydata(seg)
            ax.set_title(f"Waveform - Playing sample {pos} / {waveform.shape[0]}", color="white")
            progress = int(pos / waveform.shape[0] * 100)
            progress_bar.progress(progress)
            placeholder.pyplot(fig)
            st.session_state.pos += hop

        time.sleep(0.03)

st.caption("Powered by PyTorch (GPU) + Streamlit")
