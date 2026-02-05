# -*- coding: utf-8 -*-
import html
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from src.asr import (
    benchmark,
    clear_model_cache,
    compute_rtf,
    format_benchmark_table,
    format_ts,
    transcribe_with_progress,
)
from src.graph.dijkstra import dijkstra
from src.graph.load_graph import load_graph
from src.monitoring import clear_torch_cuda_cache, get_gpu_live_stats, log_gpu_memory
from src.nlp.extract_stations import find_nearest_station
from src.nlp.intent import Intent, detect_intent, get_intent_classifier
from src.pipeline import solve_travel_order
from src.strategies import (
    CityStrategy,
    DateStrategy,
    extract_departure_and_destinations,
    run_extraction,
)
from src.viz.map import plot_path

# Load graph for direct route calculation
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GRAPH = load_graph(str(DATA_DIR / "stations.csv"), str(DATA_DIR / "edges.csv"))

# ============================ CONFIG ============================
DEFAULT_DEVICE: str = "cuda"  # cuda or cpu
DEFAULT_COMPUTE: str = "float16"

MODEL_CHOICES: List[str] = [
    "small",
    "medium",
    "large-v3",
    "bofenghuang/whisper-large-v2-cv11-french-ct2",
    "brandenkmurray/faster-whisper-large-v3-french-distil-dec16",
]

CITY_STRATEGIES: List[CityStrategy] = ["legacy_spacy", "hf_ner"]
PIPELINE_STRATEGIES: List[str] = ["rule_based", "legacy_spacy", "hf_ner"]
DATE_STRATEGIES: List[DateStrategy] = ["eds", "hf_ner"]
INTENT_STRATEGIES: List[str] = ["rule_based", "hf_xnli"]

LANG_CHOICES: List[str] = ["auto", "fr", "en"]
BEAM_CHOICES: List[int] = [1, 2, 3, 4, 5]


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


def _plain_text_from_srt(full_text: str) -> str:
    chunks = [c for c in full_text.split("\n\n") if c.strip()]
    lines = []
    for chunk in chunks:
        parts = chunk.split("\n", 1)
        if len(parts) == 2:
            lines.append(parts[1].strip())
    return " ".join(lines).strip()


def _analyze_text(
    plain_text: str,
    *,
    model_label: str,
    city_strategy: CityStrategy,
    date_strategy: DateStrategy,
    pipeline_strategy: str,
    intent_strategy: str = "rule_based",
    audio_meta: str = "",
) -> Tuple[str, str]:
    header = (
        f"🧠 Modèle: {model_label}\n"
        f"{audio_meta}"
        f"🏙️ Stratégie villes: {city_strategy}\n"
        f"📅 Stratégie dates: {date_strategy}\n\n"
    )

    classifier = get_intent_classifier(intent_strategy)
    intent = classifier(plain_text)
    header += f"🤖 Intent model: {intent_strategy} | Intent détecté: {intent.name}\n\n"

    if intent == Intent.NOT_FRENCH:
        header += "❌ Désolé, je ne traite que les demandes en français.\n\n"
    elif intent == Intent.NOT_TRIP:
        header += "❌ Désolé, votre demande n'est pas une demande de voyage.\n"
        header += "   Essayez : 'Je veux aller de Paris à Lyon'\n\n"
    elif intent == Intent.UNKNOWN:
        header += "❌ Désolé, je n'ai pas compris votre demande.\n"
        header += "   Assurez-vous que votre message n'est pas vide.\n\n"

    extracted = run_extraction(
        full_text=plain_text,
        city_strategy=city_strategy,
        date_strategy=date_strategy,
        dates_normalize=True,
    )
    valid_cities = extracted["cities"]

    if valid_cities:
        header += "📍 Lieux détectés :\n"
        for city in valid_cities:
            station_info = ""
            station_data = find_nearest_station(city["lat"], city["lon"])
            if station_data:
                station_code, station_name, station_distance_km = station_data
                city["station_code"] = station_code
                city["station_name"] = station_name
                city["station_distance_km"] = station_distance_km
                station_info = f" → Gare: {station_name} ({station_distance_km:.1f} km)"
            header += f"- {city['name']} (lat: {city['lat']:.5f}, lon: {city['lon']:.5f}){station_info}\n"
        header += "\n"

    route_info = extract_departure_and_destinations(plain_text, valid_cities)
    path_for_map: Optional[List[str]] = None
    dep_station: Optional[str] = None
    arr_station: Optional[str] = None

    if intent == Intent.TRIP and route_info["depart"] and route_info["destinations"]:
        if "station_code" in route_info["depart"]:
            dep_station = route_info["depart"]["station_code"]
        if route_info["destinations"]:
            first_dest = route_info["destinations"][0]
            if "station_code" in first_dest:
                arr_station = first_dest["station_code"]

        if dep_station and arr_station:
            path, train_distance = dijkstra(GRAPH, dep_station, arr_station)
            if path:
                path_for_map = path
                path_str = " -> ".join(path)
                dep_to_station = (
                    route_info["depart"]["station_distance_km"]
                    if "station_distance_km" in route_info["depart"]
                    else 0.0
                )
                first_dest = route_info["destinations"][0]
                arr_to_station = (
                    first_dest["station_distance_km"]
                    if "station_distance_km" in first_dest
                    else 0.0
                )
                total_distance = train_distance + dep_to_station + arr_to_station

                header += f"🚆 Trajet ferroviaire: {path_str}\n"
                header += f"   Distance train: {train_distance} km\n"
                if dep_to_station > 1:
                    station_name = (
                        route_info["depart"]["station_name"]
                        if "station_name" in route_info["depart"]
                        else "N/A"
                    )
                    header += f"   + {route_info['depart']['name']} → {station_name}: {dep_to_station:.1f} km\n"
                if arr_to_station > 1:
                    dest_station_name = (
                        first_dest["station_name"]
                        if "station_name" in first_dest
                        else "N/A"
                    )
                    header += f"   + {dest_station_name} → {first_dest['name']}: {arr_to_station:.1f} km\n"
                header += f"   📊 Distance totale estimée: {total_distance:.1f} km\n\n"
            else:
                header += (
                    f"🚆 Aucun trajet trouvé entre {dep_station} et {arr_station}\n\n"
                )
        else:
            header += "🚆 Impossible de trouver les gares correspondantes\n\n"
    elif intent == Intent.TRIP:
        header += "🚆 Impossible de détecter le départ et/ou la destination\n\n"

    if route_info["depart"] or route_info["destinations"]:
        header += "🧭 Itinéraire :\n"
        if route_info["depart"]:
            dep = route_info["depart"]
            station_info = (
                f" (Gare: {dep['station_name']})" if "station_name" in dep else ""
            )
            header += f"- Départ : {dep['name']}{station_info}\n"

        if route_info["destinations"]:
            for idx, dest in enumerate(route_info["destinations"], 1):
                station_info = (
                    f" (Gare: {dest['station_name']})" if "station_name" in dest else ""
                )
                header += f"  {idx}. Destination : {dest['name']}{station_info}\n"

    dates_norm = extracted.get("dates_norm") or []
    dates_raw = extracted.get("dates_raw") or []
    if dates_norm:
        header += "📅 Dates (ISO) : " + ", ".join(dates_norm) + "\n\n"
    elif dates_raw:
        header += "📅 Dates (raw) : " + ", ".join(dates_raw) + "\n\n"

    tmp = tempfile.mkdtemp()
    map_path = Path(tmp) / "trajectory.html"

    if path_for_map:
        try:
            plot_path(path_for_map, DATA_DIR / "stations.csv", map_path)
            map_html = _map_iframe_from_html(map_path.read_text(encoding="utf-8"))
        except Exception as exc:
            map_html = f"<pre>{html.escape(f'No map: {exc}')}</pre>"
        analysis = solve_travel_order(
            plain_text.strip(),
            nlp_name=pipeline_strategy,
            departure_station=dep_station,
            arrival_station=arr_station,
            generate_map=False,
            map_output_html=map_path,
            intent_strategy=intent_strategy,
        )
    else:
        analysis = solve_travel_order(
            plain_text.strip(),
            nlp_name=pipeline_strategy,
            departure_station=dep_station,
            arrival_station=arr_station,
            map_output_html=map_path,
            intent_strategy=intent_strategy,
        )
        try:
            map_html = _map_iframe_from_html(map_path.read_text(encoding="utf-8"))
        except OSError:
            err = _extract_map_error(analysis)
            map_html = f"<pre>{html.escape(err or 'No map')}</pre>"

    return header + "\n" + analysis, map_html


def transcribe_and_analyze(
    audio_path: str,
    model_id: str,
    lang_choice: str,
    beam_size: int,
    city_strategy: CityStrategy,
    date_strategy: DateStrategy,
    pipeline_strategy: str,
    intent_strategy: str,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, str]:
    if not audio_path:
        return "❌ Aucun fichier audio", "<p></p>"

    log_gpu_memory(prefix="Before transcribe | ", device=DEFAULT_DEVICE)

    full_text, info, elapsed, audio_duration = transcribe_with_progress(
        audio_path=audio_path,
        model_id=model_id,
        device=DEFAULT_DEVICE,
        compute_type=DEFAULT_COMPUTE,
        lang_choice=lang_choice,
        beam_size=beam_size,
        format_ts_fn=format_ts,
        gradio_progress=progress,
    )

    log_gpu_memory(prefix="After transcribe  | ", device=DEFAULT_DEVICE)

    plain_text = _plain_text_from_srt(full_text)
    audio_meta = (
        f"🌍 Langue détectée: {info.language} ({info.language_probability:.2f})\n"
        f"⏱️ Temps: {elapsed:.2f}s | ⚡ RTF: {compute_rtf(elapsed, audio_duration)}\n"
    )

    header, map_html = _analyze_text(
        plain_text,
        model_label=model_id,
        city_strategy=city_strategy,
        date_strategy=date_strategy,
        pipeline_strategy=pipeline_strategy,
        intent_strategy=intent_strategy,
        audio_meta=audio_meta,
    )

    return header + "\n" + full_text, map_html


def analyze_text_input(
    text: str,
    city_strategy: CityStrategy,
    date_strategy: DateStrategy,
    pipeline_strategy: str,
    intent_strategy: str,
) -> Tuple[str, str]:
    if not text or not text.strip():
        return "❌ Texte vide", "<p></p>"

    header, map_html = _analyze_text(
        text.strip(),
        model_label="text",
        city_strategy=city_strategy,
        date_strategy=date_strategy,
        pipeline_strategy=pipeline_strategy,
        intent_strategy=intent_strategy,
        audio_meta="",
    )
    return header, map_html


def run_benchmark(
    audio_path: str,
    model_ids: List[str],
    lang_choice: str,
    beam_size: int,
    progress: gr.Progress = gr.Progress(),
) -> str:
    if not audio_path:
        return "❌ Aucun fichier audio"
    if not model_ids or len(model_ids) < 2:
        return "❌ Sélectionne 2–5 modèles pour le benchmark"
    if len(model_ids) > 5:
        model_ids = model_ids[:5]

    rows = benchmark(
        audio_path=audio_path,
        model_ids=model_ids,
        device=DEFAULT_DEVICE,
        compute_type=DEFAULT_COMPUTE,
        lang_choice=lang_choice,
        beam_size=beam_size,
        gradio_progress=progress,
    )

    rows_sorted = sorted(rows, key=lambda r: r["time_s"])
    table = format_benchmark_table(rows_sorted)
    winner = rows_sorted[0]

    return (
        "## 🔁 Benchmark Results\n\n"
        + table
        + f"\n\n🏁 Fastest: `{winner['model']}` ({winner['time_s']:.2f}s, RTF {winner['rtf']})\n"
        + "🧠 Reminder: speed ≠ accuracy. For real accuracy comparison, add WER/CER with a reference transcript."
    )


def ui_clear_cache() -> str:
    removed = clear_model_cache()
    extra = clear_torch_cuda_cache(device=DEFAULT_DEVICE) or ""
    return f"🧹 Cache cleared: removed {removed} model(s). {extra}".strip()


# ============================ UI ============================
with gr.Blocks(title="Whisper • SRT style text") as app:
    gr.Markdown(
        """
# 🎧 Whisper – Transcription avec timestamps
✔ Détection automatique de la langue
✔ Sélection de modèle (base + FR fine-tunés CT2)
"""
    )

    gpu_stats_md = gr.Markdown(get_gpu_live_stats(device=DEFAULT_DEVICE))
    timer = gr.Timer(2.0)
    timer.tick(
        fn=lambda: get_gpu_live_stats(device=DEFAULT_DEVICE), outputs=gpu_stats_md
    )

    with gr.Row():
        model_dd = gr.Dropdown(MODEL_CHOICES, value="small", label="🧠 Modèle")
        lang_dd = gr.Dropdown(LANG_CHOICES, value="auto", label="🌍 Language")
        beam_dd = gr.Dropdown(BEAM_CHOICES, value=1, label="🎯 Beam size")

    with gr.Row():
        city_strategy_dd = gr.Dropdown(
            CITY_STRATEGIES, value="legacy_spacy", label="🏙️ City strategy"
        )
        pipeline_strategy_dd = gr.Dropdown(
            PIPELINE_STRATEGIES, value="legacy_spacy", label="🧭 Pipeline strategy"
        )
        date_strategy_dd = gr.Dropdown(
            DATE_STRATEGIES, value="eds", label="📅 Date strategy"
        )
        intent_strategy_dd = gr.Dropdown(
            INTENT_STRATEGIES, value="rule_based", label="🤖 Intent model"
        )

    with gr.Row():
        audio_file = gr.Audio(type="filepath", label="🎵 Fichier audio")
        text_input = gr.Textbox(
            label="📝 Texte", lines=3, placeholder="Écris ta demande ici..."
        )

    with gr.Row():
        btn = gr.Button("🚀 Transcrire")
        btn_text = gr.Button("📝 Analyser texte")
        btn_clear = gr.Button("🧹 Clear cache")

    with gr.Row():
        output = gr.Textbox(label="📝 Transcription", lines=18)
        map_view = gr.HTML(value="<p></p>")

    cache_status = gr.Textbox(label="Cache status", lines=2)

    btn.click(
        transcribe_and_analyze,
        inputs=[
            audio_file,
            model_dd,
            lang_dd,
            beam_dd,
            city_strategy_dd,
            date_strategy_dd,
            pipeline_strategy_dd,
            intent_strategy_dd,
        ],
        outputs=[output, map_view],
    )

    btn_text.click(
        analyze_text_input,
        inputs=[
            text_input,
            city_strategy_dd,
            date_strategy_dd,
            pipeline_strategy_dd,
            intent_strategy_dd,
        ],
        outputs=[output, map_view],
    )

    btn_clear.click(fn=ui_clear_cache, outputs=cache_status)

    gr.Markdown("## 🔁 Benchmark (2–5 models)")
    bench_models = gr.CheckboxGroup(
        choices=MODEL_CHOICES,
        value=["small", "bofenghuang/whisper-large-v2-cv11-french-ct2"],
        label="Select 2–5 models",
    )
    btn_bench = gr.Button("🔁 Run benchmark")
    bench_out = gr.Markdown()

    btn_bench.click(
        run_benchmark,
        inputs=[audio_file, bench_models, lang_dd, beam_dd],
        outputs=bench_out,
    )

app.launch()
