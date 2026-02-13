"""NLP strategies evaluation test.

Evaluates station extraction and intent classification strategies
independently from path finding. Generates a PDF report with metrics:
precision, recall, F1, per-category accuracy, confusion matrix,
and timing percentiles (mean, median, P95, std).
"""

import csv
import statistics
import time
from collections import Counter
from dataclasses import dataclass
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

CSV_PATH = Path(__file__).resolve().parent / "data" / "generated_sentences.csv"
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
        # Convert domain model to legacy model for test compatibility
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
        # Convert domain Intent to legacy Intent by name
        return Intent[domain_intent.name]

    INTENT_STRATEGIES["finetuned_intent"] = _classify_intent_finetuned

_EXPECTED_TO_INTENT: Dict[str, str] = {
    "CORRECT": "TRIP",
    "NOT_TRIP": "NOT_TRIP",
    "NOT_FRENCH": "NOT_FRENCH",
    "UNKNOWN": "UNKNOWN",
}


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------
@dataclass
class ExtractionTestResult:
    """Single station extraction test result."""

    sentence_id: str
    sentence: str
    strategy: str
    expected_output: str
    execution_time: float
    departure: str | None
    arrival: str | None
    error: str | None
    is_complete: bool
    is_partial: bool
    passed: bool


@dataclass
class IntentTestResult:
    """Single intent classification test result."""

    sentence_id: str
    sentence: str
    strategy: str
    expected_intent: str
    predicted_intent: str
    execution_time: float
    passed: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_sentences() -> List[Tuple[str, str, str]]:
    """Load test sentences from CSV."""
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            (row["sentenceID"], row["sentence"], row["expected_output"])
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
# Extraction evaluation
# ---------------------------------------------------------------------------
def _check_extraction(result: StationExtractionResult, expected: str) -> bool:
    """Check if extraction result matches expected output."""
    if expected == "CORRECT":
        return (
            result.departure is not None
            and result.arrival is not None
            and result.error is None
        )
    elif expected in ("NOT_TRIP", "NOT_FRENCH"):
        return (
            result.error is not None
            or result.departure is None
            or result.arrival is None
        )
    return False


def evaluate_extraction() -> Tuple[List[ExtractionTestResult], List[Dict]]:
    """Evaluate all extraction strategies and compute metrics."""
    sentences = _load_sentences()
    all_results: List[ExtractionTestResult] = []
    all_metrics: List[Dict] = []

    for name, extractor in EXTRACTION_STRATEGIES.items():
        results: List[ExtractionTestResult] = []
        times: List[float] = []

        for sid, sent, expected in tqdm(sentences, desc=f"Extraction [{name}]"):
            t0 = time.perf_counter()
            ext = extractor(sent)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            complete = (
                ext.departure is not None
                and ext.arrival is not None
                and ext.error is None
            )
            partial = (
                not complete
                and ext.error is None
                and (ext.departure is not None or ext.arrival is not None)
            )
            passed = _check_extraction(ext, expected)

            results.append(
                ExtractionTestResult(
                    sentence_id=sid,
                    sentence=sent,
                    strategy=name,
                    expected_output=expected,
                    execution_time=elapsed,
                    departure=ext.departure,
                    arrival=ext.arrival,
                    error=ext.error,
                    is_complete=complete,
                    is_partial=partial,
                    passed=passed,
                )
            )

        all_results.extend(results)

        # Aggregated counts
        total = len(results)
        passed_ct = sum(r.passed for r in results)
        complete_ct = sum(r.is_complete for r in results)
        partial_ct = sum(r.is_partial for r in results)
        error_ct = sum(1 for r in results if r.error is not None)

        # Binary classification: CORRECT as positive (extraction succeeds)
        tp = sum(1 for r in results if r.expected_output == "CORRECT" and r.is_complete)
        fp = sum(1 for r in results if r.expected_output != "CORRECT" and r.is_complete)
        fn = sum(
            1 for r in results if r.expected_output == "CORRECT" and not r.is_complete
        )
        prec = _sdiv(tp, tp + fp)
        rec = _sdiv(tp, tp + fn)
        f1 = _sdiv(2 * prec * rec, prec + rec)

        # Per-category accuracy
        categories = sorted(set(r.expected_output for r in results))
        cat_acc: Dict[str, float] = {}
        for cat in categories:
            cr = [r for r in results if r.expected_output == cat]
            cat_acc[cat] = _sdiv(sum(r.passed for r in cr), len(cr))

        all_metrics.append(
            {
                "strategy": name,
                "total": total,
                "passed": passed_ct,
                "accuracy": _sdiv(passed_ct, total),
                "complete_count": complete_ct,
                "partial_count": partial_ct,
                "error_count": error_ct,
                "complete_rate": _sdiv(complete_ct, total),
                "partial_rate": _sdiv(partial_ct, total),
                "error_rate": _sdiv(error_ct, total),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "category_accuracy": cat_acc,
                "timing": _timing_stats(times),
            }
        )

    return all_results, all_metrics


# ---------------------------------------------------------------------------
# Intent evaluation
# ---------------------------------------------------------------------------
def evaluate_intent() -> Tuple[List[IntentTestResult], List[Dict]]:
    """Evaluate all intent strategies and compute metrics."""
    sentences = _load_sentences()
    all_results: List[IntentTestResult] = []
    all_metrics: List[Dict] = []

    for name, classifier in INTENT_STRATEGIES.items():
        results: List[IntentTestResult] = []
        times: List[float] = []

        for sid, sent, expected in tqdm(sentences, desc=f"Intent [{name}]"):
            exp_intent = _EXPECTED_TO_INTENT.get(expected, "UNKNOWN")

            t0 = time.perf_counter()
            intent = classifier(sent)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            pred_intent = intent.name
            passed = pred_intent == exp_intent

            results.append(
                IntentTestResult(
                    sentence_id=sid,
                    sentence=sent,
                    strategy=name,
                    expected_intent=exp_intent,
                    predicted_intent=pred_intent,
                    execution_time=elapsed,
                    passed=passed,
                )
            )

        all_results.extend(results)

        total = len(results)
        passed_ct = sum(r.passed for r in results)

        # Per-class precision, recall, F1
        intent_classes = sorted(
            set(
                [r.expected_intent for r in results]
                + [r.predicted_intent for r in results]
            )
        )
        per_class: Dict[str, Dict[str, float]] = {}
        for cls in intent_classes:
            c_tp = sum(
                1
                for r in results
                if r.expected_intent == cls and r.predicted_intent == cls
            )
            c_fp = sum(
                1
                for r in results
                if r.expected_intent != cls and r.predicted_intent == cls
            )
            c_fn = sum(
                1
                for r in results
                if r.expected_intent == cls and r.predicted_intent != cls
            )
            p = _sdiv(c_tp, c_tp + c_fp)
            r_val = _sdiv(c_tp, c_tp + c_fn)
            f = _sdiv(2 * p * r_val, p + r_val)
            per_class[cls] = {
                "precision": p,
                "recall": r_val,
                "f1": f,
                "support": c_tp + c_fn,
            }

        # Confusion matrix
        cm: Counter = Counter()
        for r in results:
            cm[(r.expected_intent, r.predicted_intent)] += 1
        cm_dict: Dict[str, Dict[str, int]] = {}
        for cls_e in intent_classes:
            cm_dict[cls_e] = {}
            for cls_p in intent_classes:
                cm_dict[cls_e][cls_p] = cm.get((cls_e, cls_p), 0)

        # Macro and weighted F1
        macro_f1 = (
            statistics.mean([m["f1"] for m in per_class.values()]) if per_class else 0.0
        )
        weighted_f1 = _sdiv(
            sum(m["f1"] * m["support"] for m in per_class.values()),
            total,
        )

        all_metrics.append(
            {
                "strategy": name,
                "total": total,
                "passed": passed_ct,
                "accuracy": _sdiv(passed_ct, total),
                "per_class": per_class,
                "confusion_matrix": cm_dict,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "timing": _timing_stats(times),
            }
        )

    return all_results, all_metrics


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------
def generate_nlp_pdf(
    ext_results: List[ExtractionTestResult],
    ext_metrics: List[Dict],
    int_results: List[IntentTestResult],
    int_metrics: List[Dict],
    pdf_path: Path,
) -> None:
    """Generate a comprehensive NLP evaluation PDF report."""
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
        title="NLP Strategies Evaluation",
        subject="Travel Order Resolver - NLP Performance Report",
    )
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
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
        """Create a key-value style table."""
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
        """Create a timing statistics table."""
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

    # ===== TITLE & SUMMARY =====
    story.append(Paragraph("NLP Strategies Evaluation", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Summary", heading_style))
    n_sentences = len(ext_results) // max(len(ext_metrics), 1)
    story.append(
        kv_table(
            [
                ["Extraction Strategies", str(len(ext_metrics))],
                ["Intent Strategies", str(len(int_metrics))],
                ["Test Sentences", str(n_sentences)],
                ["Timestamp", time.strftime("%Y-%m-%d %H:%M:%S")],
            ],
            col1=3,
            col2=2,
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    # ===== EXTRACTION STRATEGIES =====
    story.append(Paragraph("Station Extraction Evaluation", heading_style))

    for m in ext_metrics:
        story.append(
            Paragraph(f"{m['strategy'].upper()} Extractor", styles["Heading3"])
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
                    ["Precision (CORRECT)", f"{m['precision'] * 100:.2f}%"],
                    ["Recall (CORRECT)", f"{m['recall'] * 100:.2f}%"],
                    ["F1 Score (CORRECT)", f"{m['f1'] * 100:.2f}%"],
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

        # Extraction breakdown
        story.append(Paragraph("Extraction Breakdown", subheading_style))
        story.append(
            kv_table(
                [
                    [
                        "Complete Extractions",
                        f"{m['complete_rate'] * 100:.2f}% ({m['complete_count']})",
                    ],
                    [
                        "Partial Extractions",
                        f"{m['partial_rate'] * 100:.2f}% ({m['partial_count']})",
                    ],
                    [
                        "Errors",
                        f"{m['error_rate'] * 100:.2f}% ({m['error_count']})",
                    ],
                ]
            )
        )
        story.append(Spacer(1, 0.15 * inch))

        # Timing
        story.append(Paragraph("Timing", subheading_style))
        story.append(timing_table(m["timing"]))
        story.append(Spacer(1, 0.3 * inch))

    story.append(PageBreak())

    # ===== EXTRACTION COMPARISON TABLE =====
    if len(ext_metrics) > 1:
        story.append(Paragraph("Extraction Strategy Comparison", heading_style))
        cmp_data = [["Metric"] + [m["strategy"].upper() for m in ext_metrics]]
        cmp_data.append(
            ["Accuracy"] + [f"{m['accuracy'] * 100:.1f}%" for m in ext_metrics]
        )
        cmp_data.append(
            ["Precision"] + [f"{m['precision'] * 100:.1f}%" for m in ext_metrics]
        )
        cmp_data.append(["Recall"] + [f"{m['recall'] * 100:.1f}%" for m in ext_metrics])
        cmp_data.append(["F1"] + [f"{m['f1'] * 100:.1f}%" for m in ext_metrics])
        cmp_data.append(
            ["Mean Time"]
            + [f"{m['timing']['mean'] * 1000:.1f} ms" for m in ext_metrics]
        )
        cmp_data.append(
            ["Median Time"]
            + [f"{m['timing']['median'] * 1000:.1f} ms" for m in ext_metrics]
        )
        cmp_data.append(
            ["P95 Time"] + [f"{m['timing']['p95'] * 1000:.1f} ms" for m in ext_metrics]
        )

        col_w = [1.5 * inch] + [1.5 * inch] * len(ext_metrics)
        t = Table(cmp_data, colWidths=col_w)
        t.setStyle(header_table_style)
        story.append(t)
        story.append(Spacer(1, 0.3 * inch))

    # ===== INTENT STRATEGIES =====
    story.append(Paragraph("Intent Classification Evaluation", heading_style))

    for m in int_metrics:
        story.append(
            Paragraph(f"{m['strategy'].upper()} Classifier", styles["Heading3"])
        )

        # Overall metrics
        story.append(Paragraph("Overall Metrics", subheading_style))
        story.append(
            kv_table(
                [
                    [
                        "Accuracy",
                        f"{m['accuracy'] * 100:.2f}% ({m['passed']}/{m['total']})",
                    ],
                    ["Macro F1", f"{m['macro_f1'] * 100:.2f}%"],
                    ["Weighted F1", f"{m['weighted_f1'] * 100:.2f}%"],
                ]
            )
        )
        story.append(Spacer(1, 0.15 * inch))

        # Per-class precision / recall / F1
        story.append(Paragraph("Per-Class Metrics", subheading_style))
        pc_data = [["Class", "Precision", "Recall", "F1", "Support"]]
        for cls, vals in sorted(m["per_class"].items()):
            pc_data.append(
                [
                    cls,
                    f"{vals['precision'] * 100:.1f}%",
                    f"{vals['recall'] * 100:.1f}%",
                    f"{vals['f1'] * 100:.1f}%",
                    str(vals["support"]),
                ]
            )
        t = Table(
            pc_data,
            colWidths=[
                1.2 * inch,
                1.0 * inch,
                1.0 * inch,
                1.0 * inch,
                0.8 * inch,
            ],
        )
        t.setStyle(header_table_style)
        story.append(t)
        story.append(Spacer(1, 0.15 * inch))

        # Confusion matrix
        cm = m["confusion_matrix"]
        classes = sorted(cm.keys())
        story.append(Paragraph("Confusion Matrix", subheading_style))
        cm_data = [["Expected \\ Predicted"] + classes]
        for cls_e in classes:
            row = [cls_e]
            for cls_p in classes:
                row.append(str(cm[cls_e].get(cls_p, 0)))
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
        # Highlight diagonal cells in green
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
        story.append(t)
        story.append(Spacer(1, 0.15 * inch))

        # Timing
        story.append(Paragraph("Timing", subheading_style))
        story.append(timing_table(m["timing"]))
        story.append(Spacer(1, 0.3 * inch))

    doc.build(story)


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_evaluate_nlp_strategies():
    """Evaluate NLP strategies and generate PDF report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ext_results, ext_metrics = evaluate_extraction()
    int_results, int_metrics = evaluate_intent()

    pdf_path = RESULTS_DIR / "nlp_evaluation.pdf"
    generate_nlp_pdf(ext_results, ext_metrics, int_results, int_metrics, pdf_path)

    print(f"\n  PDF report saved to: {pdf_path}")

    for m in ext_metrics:
        print(
            f"  Extraction [{m['strategy']}]: "
            f"accuracy={m['accuracy'] * 100:.1f}% "
            f"P={m['precision'] * 100:.1f}% "
            f"R={m['recall'] * 100:.1f}% "
            f"F1={m['f1'] * 100:.1f}%"
        )
    for m in int_metrics:
        print(
            f"  Intent [{m['strategy']}]: "
            f"accuracy={m['accuracy'] * 100:.1f}% "
            f"macro-F1={m['macro_f1'] * 100:.1f}%"
        )

    assert ext_results, "No extraction results generated"
    assert int_results, "No intent results generated"
    assert any(r.passed for r in ext_results), "No extraction tests passed"
    assert any(r.passed for r in int_results), "No intent tests passed"
