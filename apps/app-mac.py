import os
import queue
import tempfile
import threading
import time
import html
from pathlib import Path

import gradio as gr
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import solve_travel_order

# ============================
# CONFIG
# ============================
MODEL_SIZE = "small"          # small / medium / large-v3
DEVICE = "cuda"               # cuda or cpu
COMPUTE_TYPE = "float16"      # float16 (GPU) or int8 (CPU)

SAMPLE_RATE = 16000
BUFFER_SECONDS = 5
STEP_SECONDS = 1.5

# ============================
# LOAD MODEL (GPU with fallback)
# ============================
print("🔄 Loading Whisper model...")

try:
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    print("✅ GPU model loaded")
except Exception as e:
    print("⚠️ GPU failed, fallback to CPU:", e)
    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

# ============================
# HELPERS
# ============================

def _map_iframe_from_html(document_html: str, *, height_px: int = 520) -> str:
    escaped = html.escape(document_html, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width: 100%; height: {height_px}px; border: 0;" '
        f'loading="lazy"></iframe>'
    )


def _extract_map_error(message: str) -> str:
    marker = "Map generation failed:"
    idx = message.find(marker)
    if idx == -1:
        return ""
    return message[idx:].strip()


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
        out.append(
            f"{format_ts(seg['start'])} --> {format_ts(seg['end'])}"
        )
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out)


def write_vtt(segments):
    out = ["WEBVTT\n"]
    for seg in segments:
        out.append(
            f"{format_ts(seg['start'], True)} --> {format_ts(seg['end'], True)}"
        )
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out)

# ============================
# FILE TRANSCRIPTION
# ============================


def transcribe_file(audio_path):
    if not audio_path:
        return "No audio", "<p>No map</p>"

    segments_gen, info = model.transcribe(
        audio_path,
        language=None,                 # auto language detection
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        word_timestamps=True
    )

    segments = []
    full_text = ""

    for seg in segments_gen:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })
        full_text += seg.text + " "

    tmp = tempfile.mkdtemp()

    map_path = os.path.join(tmp, "trajectory.html")

    header = f"🌍 Langue détectée: {info.language} ({info.language_probability:.2f})\n\n"
    analysis = solve_travel_order(full_text.strip(), map_output_html=map_path)
    combined_text = header + full_text.strip() + "\n\n" + analysis
    try:
        with open(map_path, "r", encoding="utf-8") as f:
            map_html = _map_iframe_from_html(f.read())
    except OSError:
        err = _extract_map_error(analysis)
        map_html = f"<pre>{html.escape(err or 'No map')}</pre>"

    return combined_text, map_html


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
    tmp = tempfile.mkdtemp()
    map_path = os.path.join(tmp, "trajectory.html")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback
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
                        buffer.flatten(),
                        language=None,
                        vad_filter=True
                    )

                    text = ""
                    for s in segments:
                        text += s.text + " "

                    text = text.strip()
                    if text and text != last_text:
                        last_text = text
                        analysis = solve_travel_order(text, map_output_html=map_path)
                        try:
                            with open(map_path, "r", encoding="utf-8") as f:
                                map_html = _map_iframe_from_html(f.read())
                        except OSError:
                            err = _extract_map_error(analysis)
                            map_html = f"<pre>{html.escape(err or 'No map')}</pre>"

                        yield text + "\n\n" + analysis, map_html

                    time.sleep(STEP_SECONDS)

            except queue.Empty:
                continue


def stop_live():
    stop_event.set()
    return "⛔ Live stopped", "<p></p>"


# ============================
# UI
# ============================
with gr.Blocks(title="Whisper • GPU • Live • SRT/VTT") as app:
    gr.Markdown("""
# 🎤 Travel Order Resolver
""")

    # ---- File mode
    audio_file = gr.Audio(type="filepath", label="🎧 Audio file")
    btn_file = gr.Button("🚀 Transcribe file")

    output_file = gr.Textbox(label="📝 Transcription", lines=10)
    map_file = gr.HTML(value="<p></p>")

    btn_file.click(transcribe_file, audio_file, [output_file, map_file])

    # gr.Markdown("## 🎙️ Live microphone")

    # with gr.Row():
    #     live_output = gr.Textbox(label="📝 Live transcription", lines=6)
    #     live_map = gr.HTML(value="<p></p>")

    # start_btn = gr.Button("▶️ Start live")
    # stop_btn = gr.Button("⛔ Stop live")

    # start_btn.click(live_transcribe, outputs=[live_output, live_map])
    # stop_btn.click(stop_live, outputs=[live_output, live_map])

# Work around Gradio api_info JSON schema bug
try:
    import gradio.routes as gr_routes

    def _safe_api_info(_serialize: bool = False):
        return {}

    gr_routes.api_info = _safe_api_info
except Exception:
    pass

app.launch(share=True)
