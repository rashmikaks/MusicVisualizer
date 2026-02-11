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

# CONFIG
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

# SIDEBAR  
st.sidebar.header("⚙️ Controls")
color_theme = st.sidebar.selectbox("🎨 Color Theme", ["Plasma", "Viridis", "Rainbow", "Inferno", "Cividis"])
fft_window = st.sidebar.slider("FFT Window Size", 512, 4096, 2048, step=512)
sensitivity = st.sidebar.slider("Sensitivity", 0.5, 5.0, 1.0, step=0.1)
st.sidebar.markdown(" ")
component_choice = st.sidebar.radio("🧩 Select Component to View",
    ["Waveform", "Spectrogram", "3D Visualizer", "Live Visualizer"],
    index=3)

# SESSION STATE  
for key, value in {
    "is_playing": False,
    "paused_at": 0.0,
    "loaded_path": None,
    "duration": 0.0,
    "sr": None,
    "y": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

try:
    if not pygame.mixer.get_init():
        pygame.mixer.init()
except Exception:
    pass

# UPLOAD  
uploaded_file = st.file_uploader("🎶 Upload MP3/WAV file", type=["mp3", "wav"])
st.markdown(" ")

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

    # COMPONENTS  
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

    # LIVE VISUALIZER (with correct playback logic)  
    elif component_choice == "Live Visualizer":
        st.subheader("🎧 Live Visualizer")
        st.caption("Play, pause, and drag the seek bar — visuals stay perfectly in sync with audio.")

        # Current playback position
        pos_ms = pygame.mixer.music.get_pos()
        if pos_ms < 0:
            pos_ms = 0
        current_time = st.session_state.paused_at + (pos_ms / 1000.0 if st.session_state.is_playing else 0.0)
        current_time = float(max(0.0, min(current_time, st.session_state.duration)))

        # SEEK BAR  
        seek_time = st.slider("⏩ Seek", 0.0, float(st.session_state.duration),
                              float(current_time), step=0.05)

        # PLAY/PAUSE/STOP  
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            play_clicked = st.button("▶ Play / Resume", use_container_width=True)
        with c2:
            pause_clicked = st.button("⏸ Pause", use_container_width=True)
        with c3:
            stop_clicked = st.button("⏹ Stop", use_container_width=True)

        # Handle Seek  
        if abs(seek_time - current_time) > 1e-6:
            st.session_state.paused_at = float(seek_time)
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.play(start=st.session_state.paused_at)
                if not st.session_state.is_playing:
                    pygame.mixer.music.pause()
            except Exception:
                try:
                    pygame.mixer.music.play()
                    pygame.mixer.music.set_pos(st.session_state.paused_at)
                    if not st.session_state.is_playing:
                        pygame.mixer.music.pause()
                except Exception:
                    pass

        # Handle Controls  
        if play_clicked:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.play(start=st.session_state.paused_at)
            except Exception:
                try:
                    pygame.mixer.music.play()
                    pygame.mixer.music.set_pos(st.session_state.paused_at)
                except Exception:
                    pass
            st.session_state.is_playing = True

        if pause_clicked:
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms > 0:
                st.session_state.paused_at += pos_ms / 1000.0
            try:
                pygame.mixer.music.pause()
            except Exception:
                pass
            st.session_state.is_playing = False

        if stop_clicked:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            st.session_state.is_playing = False
            st.session_state.paused_at = 0.0

        # Visualization  
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)
        vis_sr = 22050
        frame_size = int(vis_sr * 0.03)
        hop = frame_size
        total_frames = len(y_resampled) // hop
        stframe = st.empty()

        if st.session_state.is_playing:
            for _ in range(total_frames * 3):
                if not st.session_state.is_playing:
                    break
                pos_ms = pygame.mixer.music.get_pos()
                if pos_ms < 0:
                    pos_ms = 0
                current_time = st.session_state.paused_at + (pos_ms / 1000.0)
                if current_time >= st.session_state.duration - 0.02:
                    pygame.mixer.music.stop()
                    st.session_state.is_playing = False
                    break

                idx = int((current_time / st.session_state.duration) * total_frames)
                idx = max(0, min(idx, total_frames - 1))
                frame = y_resampled[idx * hop:(idx + 1) * hop]
                if len(frame) == 0:
                    time.sleep(0.03)
                    continue

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
                time.sleep(0.03)

            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            st.session_state.is_playing = False
            st.success("🎉 Visualization complete!")

        else:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(np.arange(100), np.zeros(100), color="gray", width=0.8)
            ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("black")
            plt.tight_layout()
            stframe.pyplot(fig)

    # Cleanup temp file
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

else:
    st.info("⬆️ Upload an audio file to start visualizing!")

# FOOTER  
st.markdown("""
<hr>
<div style='text-align:center; color:gray;'>
GPU Music Visualizer by Rahul and Rashmika
</div>
""", unsafe_allow_html=True)
