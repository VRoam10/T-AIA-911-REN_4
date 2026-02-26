"""Strategy evaluation test that measures performance across different NLP and pathfinding approaches.

Evaluates all combinations of intent × NLP × path-finding strategies.
Intent runs first and gates whether extraction/routing is attempted,
mirroring the production flow in apps/app.py. Generates a PDF report
with per-component metrics and comparison bar charts, plus a CSV of
failures per strategy combination.
"""

import csv
import os
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pytest
from tqdm import tqdm

from src.graph.dijkstra import dijkstra
from src.graph.load_graph import Graph, load_graph
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
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATIONS_CSV = DATA_DIR / "stations.csv"
EDGES_CSV = DATA_DIR / "edges.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "test_results"


# ---------------------------------------------------------------------------
# Strategy registries
# ---------------------------------------------------------------------------
StationExtractor = Callable[[str], StationExtractionResult]
PathFinder = Callable[[Graph, str, str], Tuple[list[str], float]]
IntentClassifier = Callable[[str], Intent]

NLP_STRATEGIES: Dict[str, StationExtractor] = {
    "rule_based": extract_stations,
    "hf_ner": extract_stations_hf,
}
if _SPACY_AVAILABLE:
    NLP_STRATEGIES["spacy"] = extract_stations_spacy

INTENT_STRATEGIES: Dict[str, IntentClassifier] = {
    "rule_based": detect_intent,
}

PATH_FINDER_STRATEGIES: Dict[str, PathFinder] = {
    "dijkstra": dijkstra,
}

# Register fine-tuned models if available
_finetuned_ner_model_path = (
    Path(__file__).resolve().parent.parent / "training" / "models" / "ner-camembert"
)
if _finetuned_ner_model_path.exists():
    from src.adapters.nlp.finetuned_ner_adapter import FineTunedNERAdapter

    _finetuned_ner = FineTunedNERAdapter(model_path=str(_finetuned_ner_model_path))

    def _extract_stations_finetuned(sentence: str) -> StationExtractionResult:
        result = _finetuned_ner.extract(sentence)
        return StationExtractionResult(
            departure=result.departure,
            arrival=result.arrival,
            error=result.error,
        )

    NLP_STRATEGIES["finetuned_ner"] = _extract_stations_finetuned

_finetuned_intent_model_path = (
    Path(__file__).resolve().parent.parent / "training" / "models" / "intent-camembert"
)
if _finetuned_intent_model_path.exists():
    from src.adapters.nlp.finetuned_intent_adapter import FineTunedIntentClassifier

    _finetuned_intent = FineTunedIntentClassifier(
        model_path=str(_finetuned_intent_model_path)
    )

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
class TestResult:
    """Individual test result."""

    sentence_id: str
    sentence: str
    intent_strategy: str
    nlp_strategy: str
    path_strategy: str
    expected_intent: str
    predicted_intent: str
    departure_gt: str | None
    arrival_gt: str | None
    intent_execution_time: float
    nlp_execution_time: float
    path_execution_time: float
    total_execution_time: float
    departure: str | None
    arrival: str | None
    path: list[str] | None
    distance: float | None
    error: str | None
    departure_correct: bool
    arrival_correct: bool
    passed: bool


@dataclass
class StrategyMetrics:
    """Aggregated metrics for a strategy combination."""

    intent_strategy: str
    nlp_strategy: str
    path_strategy: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    accuracy: float
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    median_execution_time: float = 0.0
    p95_execution_time: float = 0.0
    std_execution_time: float = 0.0
    # Intent classification metrics
    intent_accuracy: float = 0.0
    intent_avg_time: float = 0.0
    intent_p95_time: float = 0.0
    # NLP extraction metrics (TRIP cases only)
    trip_total: int = 0
    departure_correct: int = 0
    arrival_correct: int = 0
    both_correct: int = 0
    departure_accuracy: float = 0.0
    arrival_accuracy: float = 0.0
    both_accuracy: float = 0.0
    nlp_avg_time: float = 0.0
    nlp_median_time: float = 0.0
    nlp_p95_time: float = 0.0
    # Path finding metrics
    path_total: int = 0  # cases where extraction succeeded
    paths_found: int = 0  # cases where a path was actually returned
    paths_found_rate: float = 0.0
    avg_path_length: float = 0.0  # avg number of stations in path
    avg_distance: float = 0.0  # avg route distance in km
    path_avg_time: float = 0.0
    path_median_time: float = 0.0
    path_p95_time: float = 0.0
    # Binary classification (TRIP as positive)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # Per-category accuracy
    per_category_accuracy: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sentences() -> List[EvalCase]:
    """Load evaluation cases from a committed eval dataset CSV."""
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


def _safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0


def _compute_p95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * 0.95), len(s) - 1)]


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
def run_pipeline(
    case: EvalCase,
    intent_name: str,
    nlp_name: str,
    path_name: str,
    graph: Graph,
) -> Tuple[
    str,
    float,
    float,
    float,
    str | None,
    str | None,
    list[str] | None,
    float | None,
    str | None,
    bool,
    bool,
]:
    """Run intent → NLP → path pipeline with per-component timing.

    Returns:
        (predicted_intent, intent_time, nlp_time, path_time,
         departure, arrival, path, distance, error,
         departure_correct, arrival_correct)
    """
    intent_fn = INTENT_STRATEGIES[intent_name]
    nlp_fn = NLP_STRATEGIES[nlp_name]
    path_finder = PATH_FINDER_STRATEGIES[path_name]

    # 1. Intent classification
    t0 = time.perf_counter()
    intent = intent_fn(case.text)
    intent_time = time.perf_counter() - t0

    if intent.name != "TRIP":
        return (
            intent.name,
            intent_time,
            0.0,
            0.0,
            None,
            None,
            None,
            None,
            "intent_filtered",
            False,
            False,
        )

    # 2. NLP extraction
    t0 = time.perf_counter()
    result = nlp_fn(case.text)
    nlp_time = time.perf_counter() - t0

    if result.error or result.departure is None or result.arrival is None:
        error = result.error or "Incomplete extraction"
        dep_correct = (
            result.departure is not None and result.departure == case.departure_gt
        )
        arr_correct = result.arrival is not None and result.arrival == case.arrival_gt
        return (
            intent.name,
            intent_time,
            nlp_time,
            0.0,
            result.departure,
            result.arrival,
            None,
            None,
            error,
            dep_correct,
            arr_correct,
        )

    dep_correct = result.departure == case.departure_gt
    arr_correct = result.arrival == case.arrival_gt

    # 3. Path finding
    t0 = time.perf_counter()
    path, distance = path_finder(graph, result.departure, result.arrival)
    path_time = time.perf_counter() - t0

    return (
        intent.name,
        intent_time,
        nlp_time,
        path_time,
        result.departure,
        result.arrival,
        path,
        distance,
        None,
        dep_correct,
        arr_correct,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_strategies() -> Tuple[List[TestResult], List[StrategyMetrics]]:
    """Evaluate all intent × NLP × path strategy combinations."""
    sentences = load_sentences()
    graph = load_graph(str(STATIONS_CSV), str(EDGES_CSV))

    test_results: List[TestResult] = []
    strategy_data: Dict[Tuple[str, str, str], Dict] = {}

    for intent_name in INTENT_STRATEGIES:
        for nlp_name in NLP_STRATEGIES:
            for path_name in PATH_FINDER_STRATEGIES:
                key = (intent_name, nlp_name, path_name)
                strategy_data[key] = {
                    "total_times": [],
                    "intent_times": [],
                    "nlp_times": [],
                    "path_times": [],
                    "passed": 0,
                    "passed_by_cat": defaultdict(int),
                    "total_by_cat": defaultdict(int),
                    "results": [],
                }

                label = f"{intent_name}+{nlp_name}+{path_name}"
                for case in tqdm(sentences, desc=f"Pipeline [{label}]"):
                    (
                        predicted_intent,
                        intent_time,
                        nlp_time,
                        path_time,
                        departure,
                        arrival,
                        path,
                        distance,
                        error,
                        dep_correct,
                        arr_correct,
                    ) = run_pipeline(case, intent_name, nlp_name, path_name, graph)

                    total_time = intent_time + nlp_time + path_time

                    if case.intent_gt == "TRIP":
                        passed = (
                            error is None and bool(path) and dep_correct and arr_correct
                        )
                    else:
                        passed = error is not None or not path

                    test_result = TestResult(
                        sentence_id=case.case_id,
                        sentence=case.text,
                        intent_strategy=intent_name,
                        nlp_strategy=nlp_name,
                        path_strategy=path_name,
                        expected_intent=case.intent_gt,
                        predicted_intent=predicted_intent,
                        departure_gt=case.departure_gt,
                        arrival_gt=case.arrival_gt,
                        intent_execution_time=intent_time,
                        nlp_execution_time=nlp_time,
                        path_execution_time=path_time,
                        total_execution_time=total_time,
                        departure=departure,
                        arrival=arrival,
                        path=path,
                        distance=distance,
                        error=error,
                        departure_correct=dep_correct,
                        arrival_correct=arr_correct,
                        passed=passed,
                    )
                    test_results.append(test_result)

                    d = strategy_data[key]
                    d["total_times"].append(total_time)
                    d["intent_times"].append(intent_time)
                    d["nlp_times"].append(nlp_time)
                    d["path_times"].append(path_time)
                    d["total_by_cat"][case.intent_gt] += 1
                    d["results"].append(test_result)
                    if passed:
                        d["passed"] += 1
                        d["passed_by_cat"][case.intent_gt] += 1

    # Aggregate metrics
    strategy_metrics: List[StrategyMetrics] = []
    for (intent_name, nlp_name, path_name), d in strategy_data.items():
        total = len(d["results"])
        results = d["results"]
        times = d["total_times"]
        intent_times = d["intent_times"]
        nlp_times = d["nlp_times"]
        path_times = d["path_times"]

        # Per-category accuracy
        per_cat: Dict[str, float] = {
            cat: _safe_div(d["passed_by_cat"][cat], d["total_by_cat"][cat])
            for cat in sorted(d["total_by_cat"].keys())
        }

        # Intent accuracy
        intent_correct = sum(
            1 for r in results if r.predicted_intent == r.expected_intent
        )

        # NLP extraction accuracy (TRIP cases)
        trip_results = [r for r in results if r.expected_intent == "TRIP"]
        dep_correct_ct = sum(r.departure_correct for r in trip_results)
        arr_correct_ct = sum(r.arrival_correct for r in trip_results)
        both_correct_ct = sum(
            r.departure_correct and r.arrival_correct for r in trip_results
        )

        # Path finding metrics (cases where extraction succeeded, path was attempted)
        path_attempted = [r for r in results if r.error is None]
        paths_found = [r for r in path_attempted if r.path]
        path_lengths = [len(r.path) for r in paths_found]
        distances = [r.distance for r in paths_found if r.distance is not None]
        # Only include non-zero path times for timing (path was actually attempted)
        path_times_nonzero = [t for t in path_times if t > 0]

        # Binary P/R/F1
        tp = sum(1 for r in results if r.expected_intent == "TRIP" and r.passed)
        fp = sum(
            1
            for r in results
            if r.expected_intent != "TRIP" and r.error is None and bool(r.path)
        )
        fn = sum(1 for r in results if r.expected_intent == "TRIP" and not r.passed)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)

        strategy_metrics.append(
            StrategyMetrics(
                intent_strategy=intent_name,
                nlp_strategy=nlp_name,
                path_strategy=path_name,
                total_tests=total,
                passed_tests=d["passed"],
                failed_tests=total - d["passed"],
                accuracy=_safe_div(d["passed"], total),
                avg_execution_time=statistics.mean(times) if times else 0,
                min_execution_time=min(times) if times else 0,
                max_execution_time=max(times) if times else 0,
                median_execution_time=statistics.median(times) if times else 0,
                p95_execution_time=_compute_p95(times),
                std_execution_time=(statistics.stdev(times) if len(times) > 1 else 0),
                intent_accuracy=_safe_div(intent_correct, total),
                intent_avg_time=statistics.mean(intent_times) if intent_times else 0,
                intent_p95_time=_compute_p95(intent_times),
                trip_total=len(trip_results),
                departure_correct=dep_correct_ct,
                arrival_correct=arr_correct_ct,
                both_correct=both_correct_ct,
                departure_accuracy=_safe_div(dep_correct_ct, len(trip_results)),
                arrival_accuracy=_safe_div(arr_correct_ct, len(trip_results)),
                both_accuracy=_safe_div(both_correct_ct, len(trip_results)),
                nlp_avg_time=statistics.mean(nlp_times) if nlp_times else 0,
                nlp_median_time=(statistics.median(nlp_times) if nlp_times else 0),
                nlp_p95_time=_compute_p95(nlp_times),
                path_total=len(path_attempted),
                paths_found=len(paths_found),
                paths_found_rate=_safe_div(len(paths_found), len(path_attempted)),
                avg_path_length=(
                    statistics.mean(path_lengths) if path_lengths else 0.0
                ),
                avg_distance=(statistics.mean(distances) if distances else 0.0),
                path_avg_time=(
                    statistics.mean(path_times_nonzero) if path_times_nonzero else 0
                ),
                path_median_time=(
                    statistics.median(path_times_nonzero) if path_times_nonzero else 0
                ),
                path_p95_time=_compute_p95(path_times_nonzero),
                precision=prec,
                recall=rec,
                f1=f1,
                per_category_accuracy=per_cat,
            )
        )

    return test_results, strategy_metrics


# ---------------------------------------------------------------------------
# CSV failure export
# ---------------------------------------------------------------------------
def save_failures_csv(results: List[TestResult], results_dir: Path) -> None:
    """Write one CSV per strategy combination containing only the failed cases."""
    combos = sorted(
        {(r.intent_strategy, r.nlp_strategy, r.path_strategy) for r in results}
    )
    for intent_name, nlp_name, path_name in combos:
        failures = [
            r
            for r in results
            if r.intent_strategy == intent_name
            and r.nlp_strategy == nlp_name
            and r.path_strategy == path_name
            and not r.passed
        ]
        if not failures:
            continue
        label = f"{intent_name}+{nlp_name}+{path_name}"
        csv_path = results_dir / f"failures_{label}.csv"
        fieldnames = [
            "sentence_id",
            "sentence",
            "expected_intent",
            "predicted_intent",
            "departure_gt",
            "arrival_gt",
            "departure",
            "arrival",
            "departure_correct",
            "arrival_correct",
            "error",
            "path",
            "intent_execution_time",
            "nlp_execution_time",
            "path_execution_time",
            "total_execution_time",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in failures:
                row = asdict(r)
                row["path"] = " -> ".join(r.path) if r.path else ""
                writer.writerow(row)
        print(f"  Failures CSV: {csv_path} ({len(failures)} cases)")


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
_CHART_PALETTE = [
    "#1f6feb",
    "#2da44e",
    "#e36209",
    "#8957e5",
    "#cf222e",
    "#0969da",
    "#bf8700",
    "#1a7f37",
]


def _bar_chart_drawing(
    series_data: List[List[float]],
    series_names: List[str],
    x_labels: List[str],
    title: str,
    y_max: float = 100.0,
    y_label: str = "%",
    width: float = 460,
    height: float = 200,
):
    """Return a ReportLab Drawing with a grouped vertical bar chart."""
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.legends import Legend
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib import colors as rl_colors

    legend_height = 14 * len(series_names)
    total_height = height + legend_height + 30

    d = Drawing(width, total_height)

    # Title
    d.add(
        String(
            width / 2,
            total_height - 14,
            title,
            textAnchor="middle",
            fontSize=11,
            fontName="Helvetica-Bold",
        )
    )

    chart = VerticalBarChart()
    chart.x = 55
    chart.y = legend_height + 10
    chart.width = width - 70
    chart.height = height - 20
    chart.data = series_data
    chart.categoryAxis.categoryNames = x_labels
    if len(x_labels) > 3:
        chart.categoryAxis.labels.angle = 25
        chart.categoryAxis.labels.boxAnchor = "ne"
        chart.categoryAxis.labels.fontSize = 7
    else:
        chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = y_max
    chart.valueAxis.valueStep = y_max / 5
    chart.valueAxis.labels.fontSize = 8
    chart.groupSpacing = 8
    chart.barSpacing = 1

    for i, hex_color in enumerate(_CHART_PALETTE[: len(series_names)]):
        chart.bars[i].fillColor = rl_colors.HexColor(hex_color)

    d.add(chart)

    # Legend
    legend = Legend()
    legend.x = 55
    legend.y = legend_height - 5
    legend.dx = 10
    legend.dy = 8
    legend.deltax = 120
    legend.deltay = 13
    legend.dxTextSpace = 5
    legend.fontSize = 8
    legend.colorNamePairs = [
        (rl_colors.HexColor(_CHART_PALETTE[i]), series_names[i])
        for i in range(len(series_names))
    ]
    legend.columnMaximum = (len(series_names) + 1) // 2
    d.add(legend)

    return d


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------
def generate_pdf_report(
    results: List[TestResult], metrics: List[StrategyMetrics], pdf_path: Path
) -> None:
    """Generate a PDF report with per-component metrics and comparison charts."""
    try:
        from reportlab.graphics.renderPDF import draw as rl_draw
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
        from reportlab.platypus.flowables import Flowable

        class _ChartFlowable(Flowable):
            def __init__(self, drawing):
                super().__init__()
                self._drawing = drawing
                self.width = drawing.width
                self.height = drawing.height

            def draw(self):
                rl_draw(self._drawing, self.canv, 0, 0)

    except ImportError:
        raise ImportError(
            "reportlab is required to generate PDF reports. "
            "Install it with: pip install reportlab"
        )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        title="Strategy Evaluation Results",
        subject="Travel Order Resolver - Strategy Performance Report",
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
    sub_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading3"],
        fontSize=11,
        textColor=colors.HexColor("#555555"),
    )
    kv_ts = TableStyle(
        [
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]
    )
    hdr_ts = TableStyle(
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
        t.setStyle(kv_ts)
        return t

    # ===== TITLE & SUMMARY =====
    story.append(Paragraph("Strategy Evaluation Results", title_style))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Summary", heading_style))
    n_sentences = len(results) // len(metrics) if metrics else 0
    story.append(
        kv_table(
            [
                ["Strategy Combinations", str(len(metrics))],
                ["Test Cases", str(n_sentences)],
                ["Dataset", _DATASET_NAME],
                ["Timestamp", time.strftime("%Y-%m-%d %H:%M:%S")],
            ],
            col1=3,
            col2=2,
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    # ===== PER-COMBINATION DETAIL =====
    story.append(Paragraph("Strategy Performance", heading_style))

    for metric in metrics:
        label = (
            f"{metric.intent_strategy.upper()} \u2192 "
            f"{metric.nlp_strategy.upper()} \u2192 "
            f"{metric.path_strategy.upper()}"
        )
        story.append(Paragraph(label, styles["Heading3"]))

        # Overall classification
        story.append(Paragraph("Overall Pipeline", sub_style))
        story.append(
            kv_table(
                [
                    [
                        "Accuracy",
                        f"{metric.accuracy * 100:.2f}% ({metric.passed_tests}/{metric.total_tests})",
                    ],
                    ["Precision (TRIP)", f"{metric.precision * 100:.2f}%"],
                    ["Recall (TRIP)", f"{metric.recall * 100:.2f}%"],
                    ["F1 Score (TRIP)", f"{metric.f1 * 100:.2f}%"],
                ]
            )
        )
        story.append(Spacer(1, 0.12 * inch))

        # Per-category accuracy
        if metric.per_category_accuracy:
            story.append(Paragraph("Per-Category Accuracy", sub_style))
            cat_data = [["Category", "Accuracy"]]
            for cat, acc in sorted(metric.per_category_accuracy.items()):
                cat_data.append([cat, f"{acc * 100:.2f}%"])
            t = Table(cat_data, colWidths=[2.5 * inch, 2.5 * inch])
            t.setStyle(hdr_ts)
            story.append(t)
            story.append(Spacer(1, 0.12 * inch))

        # Intent classification
        story.append(Paragraph("Intent Classification", sub_style))
        story.append(
            kv_table(
                [
                    ["Intent Accuracy", f"{metric.intent_accuracy * 100:.2f}%"],
                    ["Avg Time", f"{metric.intent_avg_time * 1000:.2f} ms"],
                    ["P95 Time", f"{metric.intent_p95_time * 1000:.2f} ms"],
                ]
            )
        )
        story.append(Spacer(1, 0.12 * inch))

        # NLP extraction
        story.append(
            Paragraph(
                f"Station Extraction (TRIP only, n={metric.trip_total})", sub_style
            )
        )
        story.append(
            kv_table(
                [
                    [
                        "Departure correct",
                        f"{metric.departure_accuracy * 100:.2f}% ({metric.departure_correct}/{metric.trip_total})",
                    ],
                    [
                        "Arrival correct",
                        f"{metric.arrival_accuracy * 100:.2f}% ({metric.arrival_correct}/{metric.trip_total})",
                    ],
                    [
                        "Both correct",
                        f"{metric.both_accuracy * 100:.2f}% ({metric.both_correct}/{metric.trip_total})",
                    ],
                    ["Avg Time", f"{metric.nlp_avg_time * 1000:.2f} ms"],
                    ["P95 Time", f"{metric.nlp_p95_time * 1000:.2f} ms"],
                ]
            )
        )
        story.append(Spacer(1, 0.12 * inch))

        # Path finding
        story.append(
            Paragraph(f"Path Finding (attempted: {metric.path_total})", sub_style)
        )
        story.append(
            kv_table(
                [
                    [
                        "Paths found",
                        f"{metric.paths_found_rate * 100:.2f}% ({metric.paths_found}/{metric.path_total})",
                    ],
                    ["Avg path length", f"{metric.avg_path_length:.1f} stations"],
                    ["Avg distance", f"{metric.avg_distance:.1f} km"],
                    ["Avg Time", f"{metric.path_avg_time * 1000:.2f} ms"],
                    ["P95 Time", f"{metric.path_p95_time * 1000:.2f} ms"],
                ]
            )
        )
        story.append(Spacer(1, 0.12 * inch))

        # Pipeline timing
        story.append(Paragraph("Total Pipeline Timing", sub_style))
        story.append(
            kv_table(
                [
                    ["Avg", f"{metric.avg_execution_time * 1000:.2f} ms"],
                    ["Median", f"{metric.median_execution_time * 1000:.2f} ms"],
                    ["P95", f"{metric.p95_execution_time * 1000:.2f} ms"],
                    ["Std Dev", f"{metric.std_execution_time * 1000:.2f} ms"],
                ]
            )
        )
        story.append(Spacer(1, 0.3 * inch))

    # ===== COMPARISON CHARTS =====
    if len(metrics) > 1:
        story.append(PageBreak())
        story.append(Paragraph("Strategy Comparison", heading_style))

        # Short labels: "rb+rb+dijk" → "rb+rb"
        def _short(m: StrategyMetrics) -> str:
            abbr = {
                "rule_based": "rb",
                "spacy": "spacy",
                "hf_ner": "hf",
                "finetuned_ner": "ft_ner",
                "finetuned_intent": "ft_int",
                "dijkstra": "dijk",
            }
            i = abbr.get(m.intent_strategy, m.intent_strategy[:4])
            n = abbr.get(m.nlp_strategy, m.nlp_strategy[:4])
            return f"{i}+{n}"

        x_labels = [_short(m) for m in metrics]

        # Chart 1: Quality metrics (%)
        quality_series = [
            [m.accuracy * 100 for m in metrics],
            [m.f1 * 100 for m in metrics],
            [m.intent_accuracy * 100 for m in metrics],
            [m.departure_accuracy * 100 for m in metrics],
            [m.arrival_accuracy * 100 for m in metrics],
            [m.paths_found_rate * 100 for m in metrics],
        ]
        quality_names = [
            "Accuracy",
            "F1",
            "Intent acc.",
            "Dep. acc.",
            "Arr. acc.",
            "Paths found",
        ]
        chart1 = _bar_chart_drawing(
            quality_series,
            quality_names,
            x_labels,
            "Quality Metrics (%)",
            y_max=100,
            y_label="%",
        )
        story.append(_ChartFlowable(chart1))
        story.append(Spacer(1, 0.3 * inch))

        # Chart 2: Component timing P95 (ms)
        max_ms = max(m.p95_execution_time * 1000 for m in metrics) * 1.15 or 100
        timing_series = [
            [m.intent_p95_time * 1000 for m in metrics],
            [m.nlp_p95_time * 1000 for m in metrics],
            [m.path_p95_time * 1000 for m in metrics],
            [m.p95_execution_time * 1000 for m in metrics],
        ]
        timing_names = ["Intent P95", "NLP P95", "Path P95", "Total P95"]
        chart2 = _bar_chart_drawing(
            timing_series,
            timing_names,
            x_labels,
            "Timing P95 (ms)",
            y_max=round(max_ms, -1) or 100,
            y_label="ms",
        )
        story.append(_ChartFlowable(chart2))

    doc.build(story)


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_evaluate_all_strategies():
    """Evaluate all strategy combinations and generate PDF + CSV failure reports."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_results, metrics = evaluate_strategies()

    pdf_path = RESULTS_DIR / "strategy_evaluation.pdf"
    generate_pdf_report(test_results, metrics, pdf_path)
    print(f"\n  PDF report saved to: {pdf_path}")

    save_failures_csv(test_results, RESULTS_DIR)

    for m in metrics:
        print(
            f"  [{m.intent_strategy}+{m.nlp_strategy}+{m.path_strategy}]: "
            f"acc={m.accuracy * 100:.1f}% F1={m.f1 * 100:.1f}% "
            f"intent={m.intent_accuracy * 100:.1f}% "
            f"dep={m.departure_accuracy * 100:.1f}% arr={m.arrival_accuracy * 100:.1f}% "
            f"paths={m.paths_found_rate * 100:.1f}% "
            f"avg_dist={m.avg_distance:.0f}km p95={m.p95_execution_time * 1000:.0f}ms"
        )

    assert test_results, "No test results generated"
    assert metrics, "No strategy metrics generated"
    assert sum(1 for r in test_results if r.passed) > 0, "No tests passed"
