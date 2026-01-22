import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import solve_travel_order

# ============================
# CONFIG
# ============================
MODEL_SIZE = "small"  # small / medium / large-v3
DEVICE = "cuda"  # cuda or cpu
COMPUTE_TYPE = "float16"  # float16 (GPU) or int8 (CPU)

SAMPLE_RATE = 16000
BUFFER_SECONDS = 5
STEP_SECONDS = 1.5

# ============================
# LOAD MODEL (GPU with fallback)
# ============================
print("🔄 Loading Whisper model...")

try:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("✅ GPU model loaded")
except Exception as e:
    print("⚠️ GPU failed, fallback to CPU:", e)
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

# ============================
# HELPERS
# ============================


def format_ts(seconds, vtt=False):
    ms = int(seconds * 1000)
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms = ms % 1000
    sep = "." if vtt else ","
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def write_srt(segments):
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(str(i))
        out.append(f"{format_ts(seg['start'])} --> {format_ts(seg['end'])}")
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out)


def write_vtt(segments):
    out = ["WEBVTT\n"]
    for seg in segments:
        out.append(f"{format_ts(seg['start'], True)} --> {format_ts(seg['end'], True)}")
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out)


# ============================
# FILE TRANSCRIPTION
# ============================


def transcribe_file(audio_path):
    if not audio_path:
        return "No audio", None, None, None

    segments_gen, info = model.transcribe(
        audio_path,
        language=None,  # auto language detection
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        word_timestamps=True,
    )

    segments = []
    full_text = ""

    for seg in segments_gen:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
        full_text += seg.text + " "

    tmp = tempfile.mkdtemp()

    txt_path = os.path.join(tmp, "transcript.txt")
    srt_path = os.path.join(tmp, "subtitles.srt")
    vtt_path = os.path.join(tmp, "subtitles.vtt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text.strip())

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(write_srt(segments))

    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write(write_vtt(segments))

    header = (
        f"🌍 Langue détectée: {info.language} ({info.language_probability:.2f})\n\n"
    )
    analysis = solve_travel_order(full_text.strip())
    combined_text = header + full_text.strip() + "\n\n" + analysis
    return combined_text, txt_path, srt_path, vtt_path


# ============================
# LIVE MICROPHONE
# ============================
audio_queue = queue.Queue()
stop_event = threading.Event()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def live_transcribe():
    stop_event.clear()
    buffer = np.zeros((0, 1), dtype=np.float32)
    last_text = ""

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback
    ):
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.1)
                buffer = np.concatenate([buffer, chunk])

                max_len = SAMPLE_RATE * BUFFER_SECONDS
                if len(buffer) > max_len:
                    buffer = buffer[-max_len:]

                if len(buffer) >= SAMPLE_RATE * STEP_SECONDS:
                    segments, info = model.transcribe(
                        buffer.flatten(), language=None, vad_filter=True
                    )

                    text = ""
                    for s in segments:
                        text += s.text + " "

                    text = text.strip()
                    if text and text != last_text:
                        last_text = text
                        analysis = solve_travel_order(text)
                        yield text + "\n\n" + analysis

                    time.sleep(STEP_SECONDS)

            except queue.Empty:
                continue


def stop_live():
    stop_event.set()
    return "⛔ Live stopped"


# ============================
# UI
# ============================
with gr.Blocks(title="Whisper • GPU • Live • SRT/VTT") as app:
    gr.Markdown(
        """
# 🎤 Whisper — Fast • GPU • Live

### 📂 Fichier audio
✔ Auto langue
✔ VAD
✔ TXT / SRT / VTT

### 🎙️ Micro en direct
✔ Buffer circulaire
✔ Quasi temps réel
✔ GPU / CPU fallback
"""
    )

    # ---- File mode
    audio_file = gr.Audio(type="filepath", label="🎧 Audio file")
    btn_file = gr.Button("🚀 Transcribe file")

    output_file = gr.Textbox(label="📝 Transcription", lines=10)

    with gr.Row():
        txt = gr.File(label="TXT")
        srt = gr.File(label="SRT")
        vtt = gr.File(label="VTT")

    btn_file.click(transcribe_file, audio_file, [output_file, txt, srt, vtt])

    gr.Markdown("## 🎙️ Live microphone")

    live_output = gr.Textbox(label="📝 Live transcription", lines=6)

    start_btn = gr.Button("▶️ Start live")
    stop_btn = gr.Button("⛔ Stop live")

    start_btn.click(live_transcribe, outputs=live_output)
    stop_btn.click(stop_live, outputs=live_output)

# Work around Gradio api_info JSON schema bug
try:
    import gradio.routes as gr_routes

    def _safe_api_info(_serialize: bool = False):
        return {}

    gr_routes.api_info = _safe_api_info
except Exception:
    pass

app.launch(share=True)
