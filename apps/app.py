import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from faster_whisper import WhisperModel
from utils import (
    extract_departure_and_destinations,
    extract_locations,
    extract_valid_cities,
    format_ts,
)
from src.pipeline import solve_travel_order
from src.nlp.intent import detect_intent, Intent
from src.nlp.phonetic_correction import correct_city_names

# ============================ CONFIG ============================
MODEL_SIZE = "small"  # small / medium / large-v3
DEVICE = "cuda"  # cuda or cpu
COMPUTE_TYPE = "float16"

# ============================ LOAD MODEL ============================
print("🔄 Loading Whisper model...")
try:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("✅ GPU model loaded")
except Exception as e:
    print("⚠️ GPU failed, fallback to CPU:", e)
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe_file(audio_path: str) -> str:
    if not audio_path:
        return "❌ Aucun fichier audio"

    segments_gen, info = model.transcribe(
        audio_path,
        language=None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )

    # Convert generator to list to be able to iterate multiple times
    segments = list(segments_gen)

    output = [
        f"{format_ts(seg.start)} --> {format_ts(seg.end)}\n{seg.text.strip()}\n"
        for seg in segments
    ]

    full_text = "\n".join(output)

    # Extract plain text without timestamps for intent detection
    plain_text = " ".join([seg.text.strip() for seg in segments])

    # Apply phonetic correction for city names
    corrected_text = correct_city_names(plain_text)

    header = (
        f"🌍 Langue détectée: {info.language} ({info.language_probability:.2f})\n\n"
    )

    # Show correction if text was changed
    if corrected_text != plain_text:
        header += f"🔧 Correction appliquée:\n"
        header += f"   Avant: {plain_text}\n"
        header += f"   Après: {corrected_text}\n\n"

    # Detect intent and compute route if applicable
    intent = detect_intent(corrected_text)
    header += f"🤖 Intent détecté: {intent.name}\n\n"

    if intent == Intent.TRIP:
        route_result = solve_travel_order(corrected_text)
        header += f"🚆 {route_result}\n\n"
    elif intent == Intent.NOT_FRENCH:
        header += "❌ Désolé, je ne traite que les demandes en français.\n\n"
    elif intent == Intent.NOT_TRIP:
        header += "❌ Désolé, votre demande n'est pas une demande de voyage.\n"
        header += "   Essayez : 'Je veux aller de Paris à Lyon'\n\n"
    elif intent == Intent.UNKNOWN:
        header += "❌ Désolé, je n'ai pas compris votre demande.\n"
        header += "   Assurez-vous que votre message n'est pas vide.\n\n"

    locations = extract_locations(full_text)
    valid_cities = extract_valid_cities(locations)

    if valid_cities:
        header += "📍 Lieux détectés :\n"
        for city in valid_cities:
            header += (
                f"- {city['name']} (lat: {city['lat']:.5f}, lon: {city['lon']:.5f})\n"
            )
            for k, v in city["address"].items():
                header += f"    {k}: {v}\n"
        header += "\n"

    route_info = extract_departure_and_destinations(full_text, valid_cities)

    if route_info["depart"] or route_info["destinations"]:
        header += "🧭 Itinéraire :\n"
        if route_info["depart"]:
            header += f"- Départ : {route_info['depart']['name']}\n"

        if route_info["destinations"]:
            for idx, dest in enumerate(route_info["destinations"], 1):
                header += (
                    f"  {idx}. Destination : {dest['name']} "
                    f"(lat: {dest['lat']:.5f}, lon: {dest['lon']:.5f})\n"
                )

    if route_info["dates"]:
        header += "📅 Dates détectées : " + ", ".join(route_info["dates"]) + "\n\n"

    return header + full_text


# ============================ UI ============================
with gr.Blocks(title="Whisper GPU • SRT style text") as app:
    gr.Markdown(
        """
# 🎧 Whisper – Transcription avec timestamps
✔ Détection automatique de la langue
✔ GPU / CPU fallback
"""
    )

    audio_file = gr.Audio(type="filepath", label="🎵 Fichier audio")
    btn = gr.Button("🚀 Transcrire")

    output = gr.Textbox(label="📝 Transcription", lines=18)

    btn.click(transcribe_file, audio_file, output)

app.launch()
