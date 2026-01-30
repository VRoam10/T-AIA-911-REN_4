import html
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

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
from src.graph.load_graph import load_graph
from src.graph.dijkstra import dijkstra
from pathlib import Path

# Load graph for direct route calculation
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GRAPH = load_graph(str(DATA_DIR / "stations.csv"), str(DATA_DIR / "edges.csv"))

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import solve_travel_order

# ============================ CONFIG ============================
DEFAULT_DEVICE = "cuda"  # cuda or cpu
DEFAULT_COMPUTE = "float16"

# Liste de modèles testables (ta base + des FR CT2 HF)
MODEL_CHOICES = [
    "small",
    "medium",
    "large-v3",
    "bofenghuang/whisper-large-v2-cv11-french-ct2",
    "brandenkmurray/faster-whisper-large-v3-french-distil-dec16",
]

# Cache pour éviter de recharger un modèle déjà chargé
MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}


def _map_iframe_from_html(document_html: str, *, height_px: int = 520) -> str:
    escaped = html.escape(document_html, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width: 100%; height: {height_px}px; border: 0;" '
        f'loading="lazy"></iframe>'
    )


def _extract_map_error(message: str) -> Optional[str]:
    marker = "Map generation failed:"
    idx = message.find(marker)
    if idx == -1:
        return None
    return message[idx:].strip()


def get_model(model_id: str, device: str, compute_type: str) -> WhisperModel:
    key = (model_id, device, compute_type)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    # Tentative GPU, fallback CPU si souci
    try:
        m = WhisperModel(model_id, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"⚠️ Failed loading {model_id} on {device}/{compute_type}: {e}")
        m = WhisperModel(model_id, device="cpu", compute_type="int8")

    MODEL_CACHE[key] = m
    return m


def transcribe_file(audio_path: str, model_id: str) -> str:
    if not audio_path:
        return "❌ Aucun fichier audio", "<p></p>"

    model = get_model(model_id, DEFAULT_DEVICE, DEFAULT_COMPUTE)

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

    header = f"🧠 Modèle: {model_id}\n🌍 Langue détectée: {info.language} ({info.language_probability:.2f})\n\n"

    # Detect intent and compute route if applicable
    intent = detect_intent(plain_text)
    header += f"🤖 Intent détecté: {intent.name}\n\n"

    if intent == Intent.NOT_FRENCH:
        header += "❌ Désolé, je ne traite que les demandes en français.\n\n"
    elif intent == Intent.NOT_TRIP:
        header += "❌ Désolé, votre demande n'est pas une demande de voyage.\n"
        header += "   Essayez : 'Je veux aller de Paris à Lyon'\n\n"
    elif intent == Intent.UNKNOWN:
        header += "❌ Désolé, je n'ai pas compris votre demande.\n"
        header += "   Assurez-vous que votre message n'est pas vide.\n\n"

    # Extract cities with GPS coordinates and nearest stations
    locations = extract_locations(plain_text)
    valid_cities = extract_valid_cities(locations)

    if valid_cities:
        header += "📍 Lieux détectés :\n"
        for city in valid_cities:
            station_info = ""
            if city.get("station_name"):
                station_info = f" → Gare: {city['station_name']} ({city['station_distance_km']:.1f} km)"
            header += (
                f"- {city['name']} (lat: {city['lat']:.5f}, lon: {city['lon']:.5f}){station_info}\n"
            )
        header += "\n"

    route_info = extract_departure_and_destinations(plain_text, valid_cities)

    # Calculate train route if we have departure and destination
    if intent == Intent.TRIP and route_info["depart"] and route_info["destinations"]:
        dep_station = route_info["depart"].get("station_code")
        arr_station = route_info["destinations"][0].get("station_code") if route_info["destinations"] else None

        if dep_station and arr_station:
            path, train_distance = dijkstra(GRAPH, dep_station, arr_station)
            if path:
                path_str = " -> ".join(path)

                # Calculate distances to/from stations
                dep_to_station = route_info["depart"].get("station_distance_km", 0)
                arr_to_station = route_info["destinations"][0].get("station_distance_km", 0)
                total_distance = train_distance + dep_to_station + arr_to_station

                header += f"🚆 Trajet ferroviaire: {path_str}\n"
                header += f"   Distance train: {train_distance} km\n"
                if dep_to_station > 1:  # Only show if > 1km
                    header += f"   + {route_info['depart']['name']} → {route_info['depart'].get('station_name')}: {dep_to_station:.1f} km\n"
                if arr_to_station > 1:  # Only show if > 1km
                    header += f"   + {route_info['destinations'][0].get('station_name')} → {route_info['destinations'][0]['name']}: {arr_to_station:.1f} km\n"
                header += f"   📊 Distance totale estimée: {total_distance:.1f} km\n\n"
            else:
                header += f"🚆 Aucun trajet trouvé entre {dep_station} et {arr_station}\n\n"
        else:
            header += "🚆 Impossible de trouver les gares correspondantes\n\n"
    elif intent == Intent.TRIP:
        header += "🚆 Impossible de détecter le départ et/ou la destination\n\n"

    if route_info["depart"] or route_info["destinations"]:
        header += "🧭 Itinéraire :\n"
        if route_info["depart"]:
            dep = route_info["depart"]
            station_info = f" (Gare: {dep.get('station_name', 'N/A')})" if dep.get("station_name") else ""
            header += f"- Départ : {dep['name']}{station_info}\n"

        if route_info["destinations"]:
            for idx, dest in enumerate(route_info["destinations"], 1):
                station_info = f" (Gare: {dest.get('station_name', 'N/A')})" if dest.get("station_name") else ""
                header += (
                    f"  {idx}. Destination : {dest['name']}{station_info}\n"
                )

    if route_info.get("dates"):
        header += "📅 Dates détectées : " + ", ".join(route_info["dates"]) + "\n\n"

    tmp = tempfile.mkdtemp()
    map_path = os.path.join(tmp, "trajectory.html")
    analysis = solve_travel_order(full_text.strip(), map_output_html=map_path)

    try:
        with open(map_path, "r", encoding="utf-8") as f:
            map_html = _map_iframe_from_html(f.read())
    except OSError:
        err = _extract_map_error(analysis)
        map_html = f"<pre>{html.escape(err or 'No map')}</pre>"

    return header + full_text + "\n\n" + analysis, map_html


# ============================ UI ============================
with gr.Blocks(title="Whisper • SRT style text") as app:
    gr.Markdown(
        """
# 🎧 Whisper – Transcription avec timestamps
✔ Détection automatique de la langue
✔ Sélection de modèle (base + FR fine-tunés CT2)
"""
    )

    model_dd = gr.Dropdown(MODEL_CHOICES, value="small", label="🧠 Modèle")
    audio_file = gr.Audio(type="filepath", label="🎵 Fichier audio")
    btn = gr.Button("🚀 Transcrire")

    with gr.Row():
        output = gr.Textbox(label="📝 Transcription", lines=18)
        map_view = gr.HTML(value="<p></p>")

    btn.click(transcribe_file, audio_file, [output, map_view])

app.launch()
