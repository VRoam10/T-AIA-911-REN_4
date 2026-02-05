# asr.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from faster_whisper import WhisperModel

logger = logging.getLogger("whisper_app")

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


MODEL_CACHE: Dict[Tuple[str, str, str], WhisperModel] = {}


def format_ts(seconds: float) -> str:
    """
    Format seconds into SRT timestamp HH:MM:SS,mmm.
    """
    ms = int(seconds * 1000)
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms = ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def get_audio_duration_seconds(audio_path: str) -> Optional[float]:
    if sf is None:
        return None
    try:
        info = sf.info(audio_path)
        return float(info.duration)
    except Exception as e:
        logger.warning(f"Could not read audio duration: {e}")
        return None


def get_model(model_id: str, device: str, compute_type: str) -> WhisperModel:
    key = (model_id, device, compute_type)
    if key in MODEL_CACHE:
        logger.info(f"✅ Using cached model: {model_id} ({device}/{compute_type})")
        return MODEL_CACHE[key]

    logger.info(f"🔄 Loading model: {model_id} ({device}/{compute_type})")
    try:
        m = WhisperModel(model_id, device=device, compute_type=compute_type)
        logger.info("✅ Model loaded on requested device")
    except Exception as e:
        logger.warning(f"⚠️ Failed loading on {device}/{compute_type}: {e}")
        logger.info("↩️ Fallback to CPU/int8")
        m = WhisperModel(model_id, device="cpu", compute_type="int8")

    MODEL_CACHE[key] = m
    return m


def clear_model_cache() -> int:
    n = len(MODEL_CACHE)
    MODEL_CACHE.clear()
    return n


def _iter_segments_with_terminal_progress(segments_gen: Iterable[Any]):
    if tqdm is None:
        for seg in segments_gen:
            yield seg
        return
    for seg in tqdm(segments_gen, desc="📝 Transcribing", unit="segment"):
        yield seg


def compute_rtf(elapsed: float, audio_s: Optional[float]) -> str:
    if audio_s is None or audio_s <= 0:
        return "—"
    return f"{(elapsed / audio_s):.3f}"


def format_benchmark_table(rows: List[dict]) -> str:
    lines = []
    lines.append("| Model | Detected lang | Lang prob | Time (s) | Audio (s) | RTF |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        audio_val = r["audio_s"] if r["audio_s"] is not None else "—"
        lines.append(
            f"| `{r['model']}` | {r['lang']} | {r['lang_prob']:.2f} | {r['time_s']:.2f} | "
            f"{audio_val} | {r['rtf']} |"
        )
    return "\n".join(lines)


def transcribe_with_progress(
    audio_path: str,
    model_id: str,
    device: str,
    compute_type: str,
    lang_choice: str,
    beam_size: int,
    format_ts_fn,
    gradio_progress=None,
) -> tuple[str, Any, float, Optional[float]]:
    """
    Returns (full_text, info, elapsed_seconds, audio_duration_seconds)
    - full_text is SRT-like timestamped text
    - info is faster-whisper transcription info
    """
    audio_duration = get_audio_duration_seconds(audio_path)
    model = get_model(model_id, device, compute_type)

    language = None if lang_choice == "auto" else lang_choice

    start = time.time()
    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        beam_size=int(beam_size),
    )

    output_lines: List[str] = []
    last_p = 0.0

    if gradio_progress is not None:
        gradio_progress(0.0, desc="Starting transcription…")

    for seg in _iter_segments_with_terminal_progress(segments_gen):
        output_lines.append(
            f"{format_ts_fn(seg.start)} --> {format_ts_fn(seg.end)}\n{seg.text.strip()}\n"
        )
        if gradio_progress is not None and audio_duration and audio_duration > 0:
            p = min(max(seg.end / audio_duration, 0.0), 1.0)
            if p - last_p >= 0.02 or p >= 0.999:
                gradio_progress(p, desc=f"Transcribing… {int(p*100)}%")
                last_p = p

    elapsed = time.time() - start
    full_text = "\n".join(output_lines)
    return full_text, info, elapsed, audio_duration


def benchmark(
    audio_path: str,
    model_ids: List[str],
    device: str,
    compute_type: str,
    lang_choice: str,
    beam_size: int,
    gradio_progress=None,
) -> List[dict]:
    """
    Benchmark by running transcription for each model and consuming the generator fully.
    Returns list of dict rows.
    """
    audio_duration = get_audio_duration_seconds(audio_path)
    language = None if lang_choice == "auto" else lang_choice
    rows: List[dict] = []

    total = len(model_ids)
    for i, mid in enumerate(model_ids, 1):
        if gradio_progress is not None:
            gradio_progress((i - 1) / total, desc=f"Benchmark {i}/{total}: `{mid}`…")

        model = get_model(mid, device, compute_type)

        start = time.time()
        segments_gen, info = model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            beam_size=int(beam_size),
        )

        for _ in segments_gen:
            pass

        elapsed = time.time() - start
        rows.append(
            {
                "model": mid,
                "lang": info.language,
                "lang_prob": float(info.language_probability),
                "time_s": float(elapsed),
                "audio_s": None if audio_duration is None else round(audio_duration, 2),
                "rtf": compute_rtf(elapsed, audio_duration),
            }
        )

    if gradio_progress is not None:
        gradio_progress(1.0, desc="Benchmark done ✅")

    return rows
