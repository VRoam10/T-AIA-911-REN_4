"""NLP pipeline evaluation test.

Evaluates each (intent_strategy, extraction_strategy) combination as a full
pipeline: intent runs first and gates whether the extractor is called,
mirroring the production flow in apps/app.py. Generates a PDF report with
metrics: precision, recall, F1, confusion matrix, per-category accuracy,
departure/arrival accuracy, and timing percentiles.
"""

import csv
import os
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pytest
from tqdm import tqdm

from src.nlp.extract_stations import StationExtractionResult, extract_stations
from src.nlp.hf_ner import extract_stations_hf
from src.nlp.intent import Intent, detect_intent

try:
    from src.nlp.legacy_spacy.extractor import extract_stations_spacy

    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

_DATASET_NAME = os.environ.get("TOR_EVAL_DATASET", "eval_10k.csv")
CSV_PATH = Path(__file__).resolve().parent / "data" / _DATASET_NAME

try:
    from pathlib import Path as _Path

    _finetuned_ner_model = str(
        _Path(__file__).resolve().parent.parent
        / "training"
        / "models"
        / "ner-camembert"
    )
    _finetuned_intent_model = str(
        _Path(__file__).resolve().parent.parent
        / "training"
        / "models"
        / "intent-camembert"
    )
    _FINETUNED_NER_AVAILABLE = _Path(_finetuned_ner_model).exists()
    _FINETUNED_INTENT_AVAILABLE = _Path(_finetuned_intent_model).exists()
except Exception:
    _FINETUNED_NER_AVAILABLE = False
    _FINETUNED_INTENT_AVAILABLE = False

RESULTS_DIR = Path(__file__).resolve().parent.parent / "test_results"

# ---------------------------------------------------------------------------
# Strategy registries
# ---------------------------------------------------------------------------
StationExtractorFn = Callable[[str], StationExtractionResult]
IntentClassifierFn = Callable[[str], Intent]

EXTRACTION_STRATEGIES: Dict[str, StationExtractorFn] = {
    "rule_based": extract_stations,
    "hf_ner": extract_stations_hf,
}
if _SPACY_AVAILABLE:
    EXTRACTION_STRATEGIES["spacy"] = extract_stations_spacy

if _FINETUNED_NER_AVAILABLE:
    from src.adapters.nlp.finetuned_ner_adapter import FineTunedNERAdapter

    _finetuned_ner = FineTunedNERAdapter(model_path=_finetuned_ner_model)

    def _extract_stations_finetuned(sentence: str) -> StationExtractionResult:
        result = _finetuned_ner.extract(sentence)
        from src.nlp.extract_stations import StationExtractionResult as LegacyResult

        return LegacyResult(
            departure=result.departure,
            arrival=result.arrival,
            error=result.error,
        )

    EXTRACTION_STRATEGIES["finetuned_ner"] = _extract_stations_finetuned

INTENT_STRATEGIES: Dict[str, IntentClassifierFn] = {
    "rule_based": detect_intent,
}

if _FINETUNED_INTENT_AVAILABLE:
    from src.adapters.nlp.finetuned_intent_adapter import FineTunedIntentClassifier

    _finetuned_intent = FineTunedIntentClassifier(model_path=_finetuned_intent_model)

    def _classify_intent_finetuned(sentence: str) -> Intent:
        domain_intent = _finetuned_intent.classify(sentence)
        return Intent[domain_intent.name]

    INTENT_STRATEGIES["finetuned_intent"] = _classify_intent_finetuned


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalCase:
    """Single evaluation case loaded from an eval_*.csv file."""

    case_id: str
    text: str
    intent_gt: str
    departure_gt: str | None
    arrival_gt: str | None


@dataclass
class PipelineTestResult:
    """Single (intent + extraction) pipeline test result."""

    sentence_id: str
    sentence: str
    strategy: str  # "intent_name+ext_name"
    expected_intent: str  # ground truth (intent_gt from CSV)
    predicted_intent: str  # what the intent classifier predicted
    departure: str | None  # what the extractor predicted
    arrival: str | None
    departure_gt: str | None  # ground truth
    arrival_gt: str | None
    execution_time: float
    error: str | None  # None or "intent_filtered"
    is_complete: bool  # extractor ran and got both stations
    departure_correct: bool  # predicted departure == gt (TRIP cases only)
    arrival_correct: bool  # predicted arrival == gt (TRIP cases only)
    passed: bool  # entire pipeline correct end-to-end


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_sentences() -> List[EvalCase]:
    """Load evaluation cases from the committed eval dataset CSV."""
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            EvalCase(
                case_id=row["case_id"],
                text=row["text"],
                intent_gt=row["intent_gt"],
                departure_gt=row.get("departure_gt") or None,
                arrival_gt=row.get("arrival_gt") or None,
            )
            for row in reader
        ]


def _timing_stats(times: List[float]) -> Dict[str, float]:
    """Compute timing statistics: mean, median, p95, min, max, std."""
    if not times:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0, "std": 0}
    s = sorted(times)
    p95_idx = min(int(len(s) * 0.95), len(s) - 1)
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p95": s[p95_idx],
        "min": min(times),
        "max": max(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def _sdiv(n: float, d: float) -> float:
    """Safe division returning 0 when denominator is 0."""
    return n / d if d > 0 else 0.0


# ---------------------------------------------------------------------------
# Pipeline evaluation (intent × extraction matrix)
# ---------------------------------------------------------------------------
def evaluate_pipeline() -> Tuple[List[PipelineTestResult], List[Dict]]:
    """Evaluate each (intent_strategy, extraction_strategy) pair as a pipeline.

    For every sentence: intent runs first. If intent != TRIP the extractor is
    skipped (matches apps/app.py flow). Results from all pairs are collected.
    """
    sentences = _load_sentences()
    all_results: List[PipelineTestResult] = []
    all_metrics: List[Dict] = []

    for intent_name, intent_classifier in INTENT_STRATEGIES.items():
        for ext_name, extractor in EXTRACTION_STRATEGIES.items():
            label = f"{intent_name}+{ext_name}"
            results: List[PipelineTestResult] = []
            times: List[float] = []

            for case in tqdm(sentences, desc=f"Pipeline [{label}]"):
                t0 = time.perf_counter()
                intent = intent_classifier(case.text)

                if intent.name != "TRIP":
                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)
                    results.append(
                        PipelineTestResult(
                            sentence_id=case.case_id,
                            sentence=case.text,
                            strategy=label,
                            expected_intent=case.intent_gt,
                            predicted_intent=intent.name,
                            departure=None,
                            arrival=None,
                            departure_gt=case.departure_gt,
                            arrival_gt=case.arrival_gt,
                            execution_time=elapsed,
                            error="intent_filtered",
                            is_complete=False,
                            departure_correct=False,
                            arrival_correct=False,
                            passed=(case.intent_gt != "TRIP"),
                        )
                    )
                    continue

                ext = extractor(case.text)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)

                is_complete = (
                    ext.departure is not None
                    and ext.arrival is not None
                    and ext.error is None
                )
                dep_correct = (
                    ext.departure is not None and ext.departure == case.departure_gt
                )
                arr_correct = ext.arrival is not None and ext.arrival == case.arrival_gt
                # Pipeline passes for TRIP if both stations correct;
                # for non-TRIP if extractor didn't complete (no false route).
                passed = (case.intent_gt == "TRIP" and dep_correct and arr_correct) or (
                    case.intent_gt != "TRIP" and not is_complete
                )

                results.append(
                    PipelineTestResult(
                        sentence_id=case.case_id,
                        sentence=case.text,
                        strategy=label,
                        expected_intent=case.intent_gt,
                        predicted_intent=intent.name,
                        departure=ext.departure,
                        arrival=ext.arrival,
                        departure_gt=case.departure_gt,
                        arrival_gt=case.arrival_gt,
                        execution_time=elapsed,
                        error=ext.error,
                        is_complete=is_complete,
                        departure_correct=dep_correct,
                        arrival_correct=arr_correct,
                        passed=passed,
                    )
                )

            all_results.extend(results)

            # --- Aggregate metrics ---
            total = len(results)
            passed_ct = sum(r.passed for r in results)
            complete_ct = sum(r.is_complete for r in results)
            filtered_ct = sum(1 for r in results if r.error == "intent_filtered")
            error_ct = sum(
                1
                for r in results
                if r.error is not None and r.error != "intent_filtered"
            )

            # Station accuracy on TRIP cases only
            trip_results = [r for r in results if r.expected_intent == "TRIP"]
            dep_correct_ct = sum(r.departure_correct for r in trip_results)
            arr_correct_ct = sum(r.arrival_correct for r in trip_results)
            both_correct_ct = sum(
                r.departure_correct and r.arrival_correct for r in trip_results
            )

            # Binary P/R/F1: positive class = "correctly handled TRIP"
            tp = sum(1 for r in results if r.expected_intent == "TRIP" and r.passed)
            fp = sum(
                1 for r in results if r.expected_intent != "TRIP" and r.is_complete
            )
            fn = sum(1 for r in results if r.expected_intent == "TRIP" and not r.passed)
            prec = _sdiv(tp, tp + fp)
            rec = _sdiv(tp, tp + fn)
            f1 = _sdiv(2 * prec * rec, prec + rec)

            # Intent confusion matrix
            intent_classes = sorted(
                set(r.expected_intent for r in results)
                | set(r.predicted_intent for r in results)
            )
            cm: Counter = Counter(
                (r.expected_intent, r.predicted_intent) for r in results
            )
            cm_dict: Dict[str, Dict[str, int]] = {
                cls_e: {cls_p: cm.get((cls_e, cls_p), 0) for cls_p in intent_classes}
                for cls_e in intent_classes
            }

            # Per-category accuracy
            categories = sorted(set(r.expected_intent for r in results))
            cat_acc: Dict[str, float] = {
                cat: _sdiv(
                    sum(r.passed for r in results if r.expected_intent == cat),
                    sum(1 for r in results if r.expected_intent == cat),
                )
                for cat in categories
            }

            all_metrics.append(
                {
                    "strategy": label,
                    "intent_strategy": intent_name,
                    "ext_strategy": ext_name,
                    "total": total,
                    "passed": passed_ct,
                    "accuracy": _sdiv(passed_ct, total),
                    "complete_count": complete_ct,
                    "filtered_count": filtered_ct,
                    "error_count": error_ct,
                    "complete_rate": _sdiv(complete_ct, total),
                    "filtered_rate": _sdiv(filtered_ct, total),
                    "error_rate": _sdiv(error_ct, total),
                    "trip_total": len(trip_results),
                    "departure_correct": dep_correct_ct,
                    "arrival_correct": arr_correct_ct,
                    "both_correct": both_correct_ct,
                    "departure_accuracy": _sdiv(dep_correct_ct, len(trip_results)),
                    "arrival_accuracy": _sdiv(arr_correct_ct, len(trip_results)),
                    "both_accuracy": _sdiv(both_correct_ct, len(trip_results)),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "category_accuracy": cat_acc,
                    "confusion_matrix": cm_dict,
                    "intent_classes": intent_classes,
                    "timing": _timing_stats(times),
                }
            )

    return all_results, all_metrics


# ---------------------------------------------------------------------------
# CSV failure export
# ---------------------------------------------------------------------------
def save_failures_csv(results: List[PipelineTestResult], results_dir: Path) -> None:
    """Write one CSV per strategy containing only the failed cases."""
    strategies = sorted({r.strategy for r in results})
    for strategy in strategies:
        failures = [r for r in results if r.strategy == strategy and not r.passed]
        if not failures:
            continue
        csv_path = results_dir / f"failures_{strategy}.csv"
        fieldnames = [
            "sentence_id",
            "sentence",
            "expected_intent",
            "predicted_intent",
            "departure_gt",
            "arrival_gt",
            "departure",
            "arrival",
            "is_complete",
            "departure_correct",
            "arrival_correct",
            "error",
            "execution_time",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(asdict(r) for r in failures)
        print(f"  Failures CSV: {csv_path} ({len(failures)} cases)")


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------
def generate_nlp_pdf(
    results: List[PipelineTestResult],
    metrics: List[Dict],
    pdf_path: Path,
) -> None:
    """Generate a NLP pipeline evaluation PDF report."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        raise ImportError(
            "reportlab is required to generate PDF reports. "
            "Install it with: pip install reportlab"
        )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        title="NLP Pipeline Evaluation",
        subject="Travel Order Resolver - NLP Pipeline Report",
    )
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1f6feb"),
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1f6feb"),
        spaceAfter=12,
        spaceBefore=12,
    )
    subheading_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading3"],
        fontSize=11,
        textColor=colors.HexColor("#555555"),
    )
    header_table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6feb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]
    )

    def kv_table(data: list, col1: float = 2.5, col2: float = 2.5) -> Table:
        t = Table(data, colWidths=[col1 * inch, col2 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        return t

    def timing_table(ts: Dict[str, float]) -> Table:
        return kv_table(
            [
                ["Mean", f"{ts['mean'] * 1000:.2f} ms"],
                ["Median", f"{ts['median'] * 1000:.2f} ms"],
                ["P95", f"{ts['p95'] * 1000:.2f} ms"],
                ["Min", f"{ts['min'] * 1000:.2f} ms"],
                ["Max", f"{ts['max'] * 1000:.2f} ms"],
                ["Std Dev", f"{ts['std'] * 1000:.2f} ms"],
            ]
        )

    def confusion_matrix_table(cm: Dict, classes: List[str]) -> Table:
        cm_data = [["Expected \\ Predicted"] + classes]
        for cls_e in classes:
            row = [cls_e] + [str(cm[cls_e].get(cls_p, 0)) for cls_p in classes]
            cm_data.append(row)
        col_w = [1.5 * inch] + [1.0 * inch] * len(classes)
        t = Table(cm_data, colWidths=col_w)
        cm_styles = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6feb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f0f0f0")),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]
        for i in range(len(classes)):
            cm_styles.append(
                (
                    "BACKGROUND",
                    (i + 1, i + 1),
                    (i + 1, i + 1),
                    colors.HexColor("#d4edda"),
                )
            )
        t.setStyle(TableStyle(cm_styles))
        return t

    # ===== TITLE & SUMMARY =====
    story.append(Paragraph("NLP Pipeline Evaluation", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Summary", heading_style))
    n_sentences = len(results) // max(len(metrics), 1)
    story.append(
        kv_table(
            [
                ["Pipeline Combinations", str(len(metrics))],
                ["Test Sentences", str(n_sentences)],
                ["Dataset", _DATASET_NAME],
                ["Timestamp", time.strftime("%Y-%m-%d %H:%M:%S")],
            ],
            col1=3,
            col2=2,
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    # ===== PER-COMBINATION DETAIL =====
    story.append(Paragraph("Results by Strategy Combination", heading_style))

    for m in metrics:
        story.append(
            Paragraph(
                f"{m['intent_strategy'].upper()} \u2192 {m['ext_strategy'].upper()}",
                styles["Heading3"],
            )
        )

        # Classification metrics
        story.append(Paragraph("Classification Metrics", subheading_style))
        story.append(
            kv_table(
                [
                    [
                        "Overall Accuracy",
                        f"{m['accuracy'] * 100:.2f}% ({m['passed']}/{m['total']})",
                    ],
                    ["Precision (TRIP)", f"{m['precision'] * 100:.2f}%"],
                    ["Recall (TRIP)", f"{m['recall'] * 100:.2f}%"],
                    ["F1 Score (TRIP)", f"{m['f1'] * 100:.2f}%"],
                ]
            )
        )
        story.append(Spacer(1, 0.15 * inch))

        # Per-category accuracy
        story.append(Paragraph("Per-Category Accuracy", subheading_style))
        cat_data = [["Category", "Accuracy"]]
        for cat, acc in sorted(m["category_accuracy"].items()):
            cat_data.append([cat, f"{acc * 100:.2f}%"])
        t = Table(cat_data, colWidths=[2.5 * inch, 2.5 * inch])
        t.setStyle(header_table_style)
        story.append(t)
        story.append(Spacer(1, 0.15 * inch))

        # Intent confusion matrix
        story.append(Paragraph("Intent Confusion Matrix", subheading_style))
        story.append(confusion_matrix_table(m["confusion_matrix"], m["intent_classes"]))
        story.append(Spacer(1, 0.15 * inch))

        # Station extraction accuracy (TRIP sentences only)
        story.append(
            Paragraph(
                f"Station Extraction Accuracy (TRIP only, n={m['trip_total']})",
                subheading_style,
            )
        )
        story.append(
            kv_table(
                [
                    [
                        "Departure correct",
                        f"{m['departure_accuracy'] * 100:.2f}% ({m['departure_correct']}/{m['trip_total']})",
                    ],
                    [
                        "Arrival correct",
                        f"{m['arrival_accuracy'] * 100:.2f}% ({m['arrival_correct']}/{m['trip_total']})",
                    ],
                    [
                        "Both correct",
                        f"{m['both_accuracy'] * 100:.2f}% ({m['both_correct']}/{m['trip_total']})",
                    ],
                ]
            )
        )
        story.append(Spacer(1, 0.15 * inch))

        # Pipeline breakdown
        story.append(Paragraph("Pipeline Breakdown", subheading_style))
        story.append(
            kv_table(
                [
                    [
                        "Filtered by Intent",
                        f"{m['filtered_rate'] * 100:.2f}% ({m['filtered_count']})",
                    ],
                    [
                        "Extractor Errors",
                        f"{m['error_rate'] * 100:.2f}% ({m['error_count']})",
                    ],
                ]
            )
        )
        story.append(Spacer(1, 0.15 * inch))

        # Timing
        story.append(Paragraph("Timing (intent + extraction)", subheading_style))
        story.append(timing_table(m["timing"]))
        story.append(Spacer(1, 0.3 * inch))

    # ===== COMPARISON TABLE (when more than one combination) =====
    if len(metrics) > 1:
        story.append(PageBreak())
        story.append(Paragraph("Strategy Comparison", heading_style))
        cmp_data = [["Metric"] + [m["strategy"] for m in metrics]]
        cmp_data.append(["Accuracy"] + [f"{m['accuracy'] * 100:.1f}%" for m in metrics])
        cmp_data.append(
            ["Precision"] + [f"{m['precision'] * 100:.1f}%" for m in metrics]
        )
        cmp_data.append(["Recall"] + [f"{m['recall'] * 100:.1f}%" for m in metrics])
        cmp_data.append(["F1"] + [f"{m['f1'] * 100:.1f}%" for m in metrics])
        cmp_data.append(
            ["Departure acc."]
            + [f"{m['departure_accuracy'] * 100:.1f}%" for m in metrics]
        )
        cmp_data.append(
            ["Arrival acc."] + [f"{m['arrival_accuracy'] * 100:.1f}%" for m in metrics]
        )
        cmp_data.append(
            ["Intent Filtered"] + [f"{m['filtered_rate'] * 100:.1f}%" for m in metrics]
        )
        cmp_data.append(
            ["Mean Time"] + [f"{m['timing']['mean'] * 1000:.1f} ms" for m in metrics]
        )
        cmp_data.append(
            ["P95 Time"] + [f"{m['timing']['p95'] * 1000:.1f} ms" for m in metrics]
        )
        col_w = [1.8 * inch] + [1.5 * inch] * len(metrics)
        t = Table(cmp_data, colWidths=col_w)
        t.setStyle(header_table_style)
        story.append(t)

    doc.build(story)


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_evaluate_nlp_strategies():
    """Evaluate NLP pipelines and generate PDF + CSV failure reports."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results, metrics = evaluate_pipeline()

    pdf_path = RESULTS_DIR / "nlp_evaluation.pdf"
    generate_nlp_pdf(results, metrics, pdf_path)
    print(f"\n  PDF report saved to: {pdf_path}")

    save_failures_csv(results, RESULTS_DIR)

    for m in metrics:
        print(
            f"  [{m['strategy']}]: "
            f"accuracy={m['accuracy'] * 100:.1f}% "
            f"P={m['precision'] * 100:.1f}% "
            f"R={m['recall'] * 100:.1f}% "
            f"F1={m['f1'] * 100:.1f}% "
            f"dep={m['departure_accuracy'] * 100:.1f}% "
            f"arr={m['arrival_accuracy'] * 100:.1f}% "
            f"filtered={m['filtered_rate'] * 100:.1f}%"
        )

    assert results, "No pipeline results generated"
    assert any(r.passed for r in results), "No pipeline tests passed"
