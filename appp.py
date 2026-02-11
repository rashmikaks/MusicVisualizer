import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pygame
import tempfile
import time
import os
import soundfile as sf

# ------------------ CONFIG ------------------
st.set_page_config(page_title="🎧 GPU Music Visualizer", layout="wide")
st.markdown("""
<style>
body {
  background: radial-gradient(circle at top left, #0f0c29, #302b63, #24243e);
  color: white;
}
h1 {
  text-align: center;
  font-size: 3em;
  background: linear-gradient(90deg, #ff0080, #ff8c00, #40e0d0);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.block-container {
  padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎵 GPU Music Visualizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload your music and explore different visualizations with real-time Spotify-style playback.</p>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Controls")
color_theme = st.sidebar.selectbox("🎨 Color Theme", ["Plasma", "Viridis", "Rainbow", "Inferno", "Cividis"])
fft_window = st.sidebar.slider("FFT Window Size", 512, 4096, 2048, step=512)
sensitivity = st.sidebar.slider("Sensitivity", 0.5, 5.0, 1.0, step=0.1)
st.sidebar.markdown("---")
component_choice = st.sidebar.radio("🧩 Select Component to View",
    ["Waveform", "Spectrogram", "3D Visualizer", "Live Visualizer"],
    index=3)

# ------------------ SESSION STATE ------------------
for key, value in {
    "is_playing": False,
    "paused_at": 0.0,
    "loaded_path": None,
    "duration": 0.0,
    "sr": None,
    "y": None,
    "play_speed": 1.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

try:
    if not pygame.mixer.get_init():
        pygame.mixer.init()
except Exception:
    pass

# ------------------ UPLOAD ------------------
uploaded_file = st.file_uploader("🎶 Upload MP3/WAV file", type=["mp3", "wav"])
st.markdown("---")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_audio_path = tmp.name

    y, sr = librosa.load(uploaded_file, sr=None)
    sf.write(temp_audio_path, y, sr)
    duration = librosa.get_duration(y=y, sr=sr)

    st.audio(uploaded_file)
    st.write(f"**Duration:** {duration:.2f}s | **Sample Rate:** {sr} Hz")

    st.session_state.y = y
    st.session_state.sr = sr
    st.session_state.duration = float(duration)

    if st.session_state.loaded_path != temp_audio_path:
        try:
            pygame.mixer.music.load(temp_audio_path)
            st.session_state.loaded_path = temp_audio_path
            st.session_state.paused_at = 0.0
            st.session_state.is_playing = False
        except Exception as e:
            st.error(f"Could not load audio: {e}")

    # ------------------ COMPONENTS ------------------

    # WAVEFORM
    if component_choice == "Waveform":
        st.subheader("📈 Waveform")
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, color="#00FFFF", alpha=0.8, ax=ax)
        ax.set_facecolor("black")
        ax.set_title("Waveform", color="white")
        ax.set_xticks([]); ax.set_yticks([])
        st.pyplot(fig)

    # SPECTROGRAM
    elif component_choice == "Spectrogram":
        st.subheader("🌈 2D Spectrogram")
        D = np.abs(librosa.stft(y, n_fft=fft_window, hop_length=fft_window // 4))
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                                       y_axis='log', x_axis='time', sr=sr,
                                       cmap=color_theme.lower(), ax=ax)
        ax.set_title("Spectrogram", color="white")
        plt.colorbar(img, ax=ax, format="%+2.f dB")
        st.pyplot(fig)

    # 3D VISUALIZER
    elif component_choice == "3D Visualizer":
        st.subheader("🧊 3D Frequency Surface")
        D = np.abs(librosa.stft(y, n_fft=fft_window, hop_length=fft_window // 4))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=fft_window)
        times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=fft_window // 4)
        D_small = D[:512, ::4] ** sensitivity
        fig3d = go.Figure(data=[go.Surface(
            z=D_small, x=times[::4], y=freqs[:512],
            colorscale=color_theme.lower(), showscale=False)])
        fig3d.update_layout(
            scene=dict(xaxis_title="Time (s)", yaxis_title="Freq (Hz)", zaxis_title="Amplitude"),
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0),
            scene_camera=dict(eye=dict(x=1.3, y=1.3, z=0.7))
        )
        st.plotly_chart(fig3d, use_container_width=True)

#Live visualizer
    elif component_choice == "Live Visualizer":
        st.subheader("🎧 Live Visualizer")
        st.caption("Play, pause, and drag the seek bar to jump to any position. Visual speed adjusts animation only.")

        # Resample for visualization
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)
        vis_sr = 22050
        frame_size = int(vis_sr * 0.03)
        hop = frame_size
        total_frames = len(y_resampled) // hop

        stframe = st.empty()
        time_display = st.empty()
        seek_placeholder = st.empty()
        st.markdown("---")

        # --- CENTERED PLAY/PAUSE BUTTONS ---
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            play_clicked = st.button("▶ Play", use_container_width=True)
            pause_clicked = st.button("⏸ Pause", use_container_width=True)

        # --- SESSION STATE ---
        ss = st.session_state
        ss.setdefault("is_playing", False)
        ss.setdefault("paused_at", 0.0)
        ss.setdefault("start_time", None)
        ss.setdefault("seek_at", 0.0)

        def format_time(seconds):
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"

        # --- PLAY LOGIC ---
        if play_clicked:
            pygame.mixer.music.stop()
            pygame.mixer.music.play(start=ss.paused_at)
            ss.start_time = time.time() - ss.paused_at
            ss.is_playing = True

        # --- PAUSE LOGIC ---
        if pause_clicked and ss.is_playing:
            ss.paused_at = time.time() - ss.start_time
            pygame.mixer.music.stop()
            ss.is_playing = False

        # --- MAIN LOOP ---
        while ss.is_playing:
            current_time = time.time() - ss.start_time
            if current_time >= ss.duration:
                ss.is_playing = False
                pygame.mixer.music.stop()
                st.success("🎶 Playback complete!")
                break

            # --- TIME DISPLAY ---
            time_display.markdown(
                f"<div style='display:flex; justify-content:space-between; color:#FF4B4B; font-weight:bold;'>"
                f"<span>{format_time(current_time)}</span>"
                f"<span>{format_time(ss.duration)}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            # --- INTERACTIVE SEEK BAR ---
            seek_val = seek_placeholder.slider(
                "🔴 Seek Position",
                0.0, float(ss.duration),
                float(current_time),
                step=0.1,
                key=f"seek_{int(current_time*10)}"
            )

            # If user moves slider significantly → seek to that position
            if abs(seek_val - current_time) > 0.3:
                ss.paused_at = seek_val
                pygame.mixer.music.stop()
                pygame.mixer.music.play(start=seek_val)
                ss.start_time = time.time() - seek_val
                continue

            # --- FFT VISUALIZATION ---
            idx = int((current_time / ss.duration) * total_frames)
            idx = max(0, min(idx, total_frames - 1))
            frame = y_resampled[idx * hop:(idx + 1) * hop]

            if len(frame) > 0:
                fft = np.abs(np.fft.rfft(frame))[:100]
                fft = fft / np.max(fft + 1e-6)
                fft = fft ** sensitivity

                fig, ax = plt.subplots(figsize=(8, 3))
                cmap = plt.cm.get_cmap(color_theme.lower())
                colors = cmap(fft)
                ax.bar(np.arange(len(fft)), fft, color=colors, width=0.8)
                ax.set_ylim(0, 1)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_facecolor("black")
                plt.tight_layout()
                stframe.pyplot(fig)
                plt.close(fig)

            time.sleep(0.25)

        # --- PAUSED/STATIC DISPLAY ---
        if not ss.is_playing:
            current_time = ss.paused_at
            time_display.markdown(
                f"<div style='display:flex; justify-content:space-between; color:#FF4B4B; font-weight:bold;'>"
                f"<span>{format_time(current_time)}</span>"
                f"<span>{format_time(ss.duration)}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            seek_val = seek_placeholder.slider(
                "🔴 Seek Position",
                0.0, float(ss.duration),
                float(current_time),
                step=0.1,
                key=f"seek_static_{int(time.time())}"
            )

            # Allow instant seek even when paused
            if abs(seek_val - current_time) > 0.3:
                ss.paused_at = seek_val
                pygame.mixer.music.stop()
                pygame.mixer.music.play(start=seek_val)
                ss.start_time = time.time() - seek_val
                ss.is_playing = True
                st.experimental_rerun()

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(np.arange(100), np.zeros(100), color="gray", width=0.8)
            ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("black")
            plt.tight_layout()
            stframe.pyplot(fig)




# ------------------ FOOTER ------------------
st.markdown("""
<hr>
<div style='text-align:center; color:gray;'>
GPU Music Visualizer by Rahul and Rashmika
</div>
""", unsafe_allow_html=True)