"""Strategy evaluation test that measures performance across different NLP and pathfinding approaches.

This test evaluates all combinations of available strategies and generates
a comprehensive results report in PDF format for deliveries.
"""

import csv
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from src.graph.dijkstra import dijkstra
from src.graph.load_graph import Graph, load_graph
from src.nlp.extract_stations import StationExtractionResult, extract_stations
from src.nlp.hf_ner import extract_stations_hf

CSV_PATH = Path(__file__).resolve().parent / "data" / "generated_sentences.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATIONS_CSV = DATA_DIR / "stations.csv"
EDGES_CSV = DATA_DIR / "edges.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "test_results"


# Strategy registries
StationExtractor = Callable[[str], StationExtractionResult]
PathFinder = Callable[[Graph, str, str], Tuple[list[str], float]]

NLP_STRATEGIES: Dict[str, StationExtractor] = {
    "rule_based": extract_stations,
    "hf_ner": extract_stations_hf,
}

PATH_FINDER_STRATEGIES: Dict[str, PathFinder] = {
    "dijkstra": dijkstra,
}


@dataclass
class TestResult:
    """Individual test result."""

    sentence_id: str
    sentence: str
    nlp_strategy: str
    path_strategy: str
    expected_output: str
    nlp_execution_time: float
    path_execution_time: float
    total_execution_time: float
    departure: str | None
    arrival: str | None
    path: list[str] | None
    distance: float | None
    error: str | None
    passed: bool


@dataclass
class StrategyMetrics:
    """Aggregated metrics for a strategy combination."""

    nlp_strategy: str
    path_strategy: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    accuracy: float
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    # Enhanced timing
    median_execution_time: float = 0.0
    p95_execution_time: float = 0.0
    std_execution_time: float = 0.0
    # Component timing
    nlp_avg_time: float = 0.0
    nlp_median_time: float = 0.0
    nlp_p95_time: float = 0.0
    path_avg_time: float = 0.0
    path_median_time: float = 0.0
    path_p95_time: float = 0.0
    # Binary classification (CORRECT as positive)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # Per-category accuracy
    per_category_accuracy: Dict[str, float] = field(default_factory=dict)


def load_sentences() -> List[Tuple[str, str, str]]:
    """Load test sentences from CSV."""
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            (row["sentenceID"], row["sentence"], row["expected_output"])
            for row in reader
        ]


def validate_result(message: str, expected_output: str) -> bool:
    """Check if the result matches expected output."""
    if expected_output == "CORRECT":
        return (
            "," in message
            and "Extraction error" not in message
            and "No path found" not in message
        )
    elif expected_output == "NOT_TRIP":
        return "No path found" in message or "Extraction error" in message
    elif expected_output == "NOT_FRENCH":
        return "Extraction error" in message
    return False


def run_pipeline(
    sentence: str,
    nlp_name: str = "rule_based",
    path_name: str = "dijkstra",
) -> Tuple[
    str,
    float,
    float,
    str | None,
    str | None,
    list[str] | None,
    float | None,
    str | None,
]:
    """Run the core pipeline and return results with timing information.

    Returns:
        Tuple of (message, nlp_time, path_time, departure, arrival, path, distance, error)
    """
    nlp = NLP_STRATEGIES.get(nlp_name)
    if nlp is None:
        return (
            f"Unknown NLP strategy: {nlp_name!r}",
            0,
            0,
            None,
            None,
            None,
            None,
            f"Unknown NLP strategy: {nlp_name!r}",
        )

    path_finder = PATH_FINDER_STRATEGIES.get(path_name)
    if path_finder is None:
        return (
            f"Unknown path-finding strategy: {path_name!r}",
            0,
            0,
            None,
            None,
            None,
            None,
            f"Unknown path-finding strategy: {path_name!r}",
        )

    # NLP execution with timing
    start_nlp = time.perf_counter()
    result = nlp(sentence)
    nlp_time = time.perf_counter() - start_nlp

    if result.error:
        return (
            f"Extraction error: {result.error}",
            nlp_time,
            0,
            None,
            None,
            None,
            None,
            result.error,
        )

    if result.departure is None or result.arrival is None:
        error = "Departure or arrival not set"
        return f"Extraction error: {error}", nlp_time, 0, None, None, None, None, error

    departure = result.departure
    arrival = result.arrival

    graph = load_graph(str(STATIONS_CSV), str(EDGES_CSV))

    # Path finding execution with timing
    start_path = time.perf_counter()
    path, distance = path_finder(graph, departure, arrival)
    path_time = time.perf_counter() - start_path

    if not path:
        message = f"No path found between {departure} and {arrival}."
    else:
        path_str = " -> ".join(path)
        message = f"{departure},{arrival}"

    return message, nlp_time, path_time, departure, arrival, path, distance, None


def _safe_div(n: float, d: float) -> float:
    """Safe division returning 0 when denominator is 0."""
    return n / d if d > 0 else 0.0


def _compute_p95(values: List[float]) -> float:
    """Compute the 95th percentile of a list of values."""
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * 0.95), len(s) - 1)]


def evaluate_strategies() -> Tuple[List[TestResult], List[StrategyMetrics]]:
    """Evaluate all strategy combinations against all test sentences."""
    test_results: List[TestResult] = []
    # Track detailed data per strategy combination
    strategy_data: Dict[Tuple[str, str], Dict] = {}

    sentences = load_sentences()

    # Test each strategy combination
    for nlp_name in NLP_STRATEGIES.keys():
        for path_name in PATH_FINDER_STRATEGIES.keys():
            key = (nlp_name, path_name)
            strategy_data[key] = {
                "total_times": [],
                "nlp_times": [],
                "path_times": [],
                "passed": 0,
                "passed_by_cat": defaultdict(int),
                "total_by_cat": defaultdict(int),
                "results": [],
            }

            for sentence_id, sentence, expected_output in sentences:
                (
                    message,
                    nlp_time,
                    path_time,
                    departure,
                    arrival,
                    path,
                    distance,
                    error,
                ) = run_pipeline(sentence, nlp_name, path_name)

                total_time = nlp_time + path_time
                passed = validate_result(message, expected_output)

                test_result = TestResult(
                    sentence_id=sentence_id,
                    sentence=sentence,
                    nlp_strategy=nlp_name,
                    path_strategy=path_name,
                    expected_output=expected_output,
                    nlp_execution_time=nlp_time,
                    path_execution_time=path_time,
                    total_execution_time=total_time,
                    departure=departure,
                    arrival=arrival,
                    path=path,
                    distance=distance,
                    error=error,
                    passed=passed,
                )
                test_results.append(test_result)

                d = strategy_data[key]
                d["total_times"].append(total_time)
                d["nlp_times"].append(nlp_time)
                d["path_times"].append(path_time)
                d["total_by_cat"][expected_output] += 1
                d["results"].append(test_result)
                if passed:
                    d["passed"] += 1
                    d["passed_by_cat"][expected_output] += 1

    # Compute aggregated metrics
    strategy_metrics: List[StrategyMetrics] = []
    for (nlp_name, path_name), d in strategy_data.items():
        total = len(sentences)
        _passed = d["passed"]
        times = d["total_times"]
        nlp_times = d["nlp_times"]
        path_times = d["path_times"]

        # Per-category accuracy
        per_cat: Dict[str, float] = {}
        for cat in sorted(d["total_by_cat"].keys()):
            per_cat[cat] = _safe_div(d["passed_by_cat"][cat], d["total_by_cat"][cat])

        # Binary P/R/F1: CORRECT as positive class
        # TP = expected CORRECT and passed (pipeline succeeded)
        # FP = expected NOT_CORRECT and NOT passed (pipeline succeeded when it shouldn't)
        # FN = expected CORRECT and NOT passed (pipeline failed when it should succeed)
        tp = sum(1 for r in d["results"] if r.expected_output == "CORRECT" and r.passed)
        fp = sum(
            1 for r in d["results"] if r.expected_output != "CORRECT" and not r.passed
        )
        fn = sum(
            1 for r in d["results"] if r.expected_output == "CORRECT" and not r.passed
        )
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)

        metrics = StrategyMetrics(
            nlp_strategy=nlp_name,
            path_strategy=path_name,
            total_tests=total,
            passed_tests=_passed,
            failed_tests=total - _passed,
            accuracy=_safe_div(_passed, total),
            avg_execution_time=statistics.mean(times) if times else 0,
            min_execution_time=min(times) if times else 0,
            max_execution_time=max(times) if times else 0,
            median_execution_time=statistics.median(times) if times else 0,
            p95_execution_time=_compute_p95(times),
            std_execution_time=(statistics.stdev(times) if len(times) > 1 else 0),
            nlp_avg_time=statistics.mean(nlp_times) if nlp_times else 0,
            nlp_median_time=(statistics.median(nlp_times) if nlp_times else 0),
            nlp_p95_time=_compute_p95(nlp_times),
            path_avg_time=statistics.mean(path_times) if path_times else 0,
            path_median_time=(statistics.median(path_times) if path_times else 0),
            path_p95_time=_compute_p95(path_times),
            precision=prec,
            recall=rec,
            f1=f1,
            per_category_accuracy=per_cat,
        )
        strategy_metrics.append(metrics)

    return test_results, strategy_metrics


def generate_pdf_report(
    results: List[TestResult], metrics: List[StrategyMetrics], pdf_path: Path
) -> None:
    """Generate a PDF report using reportlab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4, letter
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
        title="Strategy Evaluation Results",
        subject="Travel Order Resolver - Strategy Performance Report",
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

    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ]
    )

    # Title
    story.append(Paragraph("Strategy Evaluation Results", title_style))
    story.append(Spacer(1, 0.3 * inch))

    # Summary
    story.append(Paragraph("Summary", heading_style))
    summary_data = [
        ["Total Strategy Combinations", str(len(metrics))],
        ["Total Test Cases", str(len(results) // len(metrics) if metrics else 0)],
        ["Timestamp", time.strftime("%Y-%m-%d %H:%M:%S")],
    ]
    summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
    summary_table.setStyle(table_style)
    story.append(summary_table)
    story.append(Spacer(1, 0.3 * inch))

    # Strategy Performance
    story.append(Paragraph("Strategy Performance", heading_style))
    for metric in metrics:
        strategy_name = f"{metric.nlp_strategy.upper()} NLP + {metric.path_strategy.upper()} Path Finder"
        story.append(Paragraph(strategy_name, styles["Heading3"]))

        sub_style = ParagraphStyle(
            "SubHeading3",
            parent=styles["Heading3"],
            fontSize=11,
            textColor=colors.HexColor("#555555"),
        )

        perf_data = [
            [
                "Accuracy",
                f"{metric.accuracy * 100:.2f}% ({metric.passed_tests}/{metric.total_tests})",
            ],
            ["Precision (CORRECT)", f"{metric.precision * 100:.2f}%"],
            ["Recall (CORRECT)", f"{metric.recall * 100:.2f}%"],
            ["F1 Score (CORRECT)", f"{metric.f1 * 100:.2f}%"],
        ]

        story.append(Paragraph("Classification Metrics", sub_style))

        perf_table = Table(perf_data, colWidths=[2.5 * inch, 2.5 * inch])
        perf_table.setStyle(table_style)
        story.append(perf_table)
        story.append(Spacer(1, 0.15 * inch))

        # Per-category accuracy
        if metric.per_category_accuracy:
            story.append(Paragraph("Per-Category Accuracy", sub_style))
            cat_header_style = TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#1f6feb"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ]
            )
            cat_data = [["Category", "Accuracy"]]
            for cat, acc in sorted(metric.per_category_accuracy.items()):
                cat_data.append([cat, f"{acc * 100:.2f}%"])
            cat_table = Table(cat_data, colWidths=[2.5 * inch, 2.5 * inch])
            cat_table.setStyle(cat_header_style)
            story.append(cat_table)
            story.append(Spacer(1, 0.15 * inch))

        # Timing
        story.append(Paragraph("Pipeline Timing", sub_style))
        timing_data = [
            [
                "Avg Execution Time",
                f"{metric.avg_execution_time * 1000:.2f} ms",
            ],
            [
                "Median Execution Time",
                f"{metric.median_execution_time * 1000:.2f} ms",
            ],
            [
                "P95 Execution Time",
                f"{metric.p95_execution_time * 1000:.2f} ms",
            ],
            [
                "Min Execution Time",
                f"{metric.min_execution_time * 1000:.2f} ms",
            ],
            [
                "Max Execution Time",
                f"{metric.max_execution_time * 1000:.2f} ms",
            ],
            [
                "Std Dev",
                f"{metric.std_execution_time * 1000:.2f} ms",
            ],
        ]
        timing_table = Table(timing_data, colWidths=[2.5 * inch, 2.5 * inch])
        timing_table.setStyle(table_style)
        story.append(timing_table)
        story.append(Spacer(1, 0.2 * inch))

        # Filter results for this strategy combination
        strategy_results = [
            r
            for r in results
            if r.nlp_strategy == metric.nlp_strategy
            and r.path_strategy == metric.path_strategy
        ]

        # NLP Results for this strategy
        story.append(
            Paragraph(
                "NLP Results",
                ParagraphStyle(
                    "SubHeading3",
                    parent=styles["Heading3"],
                    fontSize=11,
                    textColor=colors.HexColor("#555555"),
                ),
            )
        )

        nlp_strategy_results = [r for r in strategy_results if not r.error]
        nlp_strategy_errors = [r for r in strategy_results if r.error]

        nlp_strategy_data = [
            ["Inputs Processed", str(len(strategy_results))],
            ["Successfully Extracted", str(len(nlp_strategy_results))],
            ["Extraction Errors", str(len(nlp_strategy_errors))],
            [
                "Avg Extraction Time",
                f"{metric.nlp_avg_time * 1000:.2f} ms",
            ],
            [
                "Median Extraction Time",
                f"{metric.nlp_median_time * 1000:.2f} ms",
            ],
            [
                "P95 Extraction Time",
                f"{metric.nlp_p95_time * 1000:.2f} ms",
            ],
        ]
        nlp_strategy_table = Table(
            nlp_strategy_data, colWidths=[2.5 * inch, 2.5 * inch]
        )
        nlp_strategy_table.setStyle(table_style)
        story.append(nlp_strategy_table)
        story.append(Spacer(1, 0.15 * inch))

        # Path Finder Results for this strategy
        story.append(
            Paragraph(
                "Path Finder Results",
                ParagraphStyle(
                    "SubHeading3",
                    parent=styles["Heading3"],
                    fontSize=11,
                    textColor=colors.HexColor("#555555"),
                ),
            )
        )

        path_strategy_results = [r for r in strategy_results if not r.error]

        if path_strategy_results:
            avg_path_len = sum(
                len(r.path) if r.path else 0 for r in path_strategy_results
            ) / len(path_strategy_results)
            path_strategy_data = [
                ["Paths Computed", str(len(path_strategy_results))],
                [
                    "Avg Computation Time",
                    f"{metric.path_avg_time * 1000:.2f} ms",
                ],
                [
                    "Median Computation Time",
                    f"{metric.path_median_time * 1000:.2f} ms",
                ],
                [
                    "P95 Computation Time",
                    f"{metric.path_p95_time * 1000:.2f} ms",
                ],
                [
                    "Avg Path Length",
                    f"{avg_path_len:.2f} stations",
                ],
            ]
            path_strategy_table = Table(
                path_strategy_data, colWidths=[2.5 * inch, 2.5 * inch]
            )
            path_strategy_table.setStyle(table_style)
            story.append(path_strategy_table)

        story.append(Spacer(1, 0.25 * inch))

    story.append(PageBreak())

    # Per-Strategy Results
    story.append(Paragraph("Results Per Strategy", heading_style))

    # NLP Results - GLOBAL
    story.append(Paragraph("NLP Results (Global)", heading_style))
    story.append(
        Paragraph(
            "The NLP component extracts departure and arrival stations from user input.",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 0.1 * inch))

    nlp_results_global = [r for r in results if not r.error]
    nlp_errors_global = [r for r in results if r.error]

    if results:
        nlp_times_global = [r.nlp_execution_time for r in results]
        nlp_times_sorted = sorted(nlp_times_global)
        nlp_p95_idx = min(int(len(nlp_times_sorted) * 0.95), len(nlp_times_sorted) - 1)
        nlp_data_global = [
            ["Total Inputs Processed", str(len(results))],
            ["Successfully Extracted", str(len(nlp_results_global))],
            ["Extraction Errors", str(len(nlp_errors_global))],
            [
                "Avg Extraction Time",
                f"{statistics.mean(nlp_times_global) * 1000:.2f} ms",
            ],
            [
                "Median Extraction Time",
                f"{statistics.median(nlp_times_global) * 1000:.2f} ms",
            ],
            [
                "P95 Extraction Time",
                f"{nlp_times_sorted[nlp_p95_idx] * 1000:.2f} ms",
            ],
        ]
        nlp_table_global = Table(nlp_data_global, colWidths=[3 * inch, 2 * inch])
        nlp_table_global.setStyle(table_style)
        story.append(nlp_table_global)
        story.append(Spacer(1, 0.2 * inch))

    # Path Finder Results - GLOBAL
    story.append(Paragraph("Path Finder Results (Global)", heading_style))
    story.append(
        Paragraph(
            "The path finder computes the shortest route between two stations.",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 0.1 * inch))

    path_results_global = [r for r in results if not r.error]
    if path_results_global:
        path_data_global = [
            ["Total Paths Computed", str(len(path_results_global))],
            [
                "Avg Path Computation Time",
                f"{sum(r.path_execution_time for r in path_results_global) / len(path_results_global) * 1000:.2f} ms",
            ],
            [
                "Avg Path Length",
                f"{sum(len(p) if p else 0 for p in [r.path for r in path_results_global]) / len(path_results_global):.2f} stations",
            ],
        ]
        path_table_global = Table(path_data_global, colWidths=[3 * inch, 2 * inch])
        path_table_global.setStyle(table_style)
        story.append(path_table_global)
        story.append(Spacer(1, 0.2 * inch))

    # Pipeline Results
    story.append(Paragraph("Pipeline Results", heading_style))
    story.append(
        Paragraph(
            "End-to-end pipeline performance combining NLP and path finding.",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 0.1 * inch))

    all_results = results
    all_times = [r.total_execution_time for r in all_results]
    all_times_sorted = sorted(all_times)
    all_p95_idx = min(int(len(all_times_sorted) * 0.95), len(all_times_sorted) - 1)

    pipeline_data = [
        ["Total Pipeline Executions", str(len(all_results))],
        ["Avg Total Time", f"{statistics.mean(all_times) * 1000:.2f} ms"],
        [
            "Median Total Time",
            f"{statistics.median(all_times) * 1000:.2f} ms",
        ],
        [
            "P95 Total Time",
            f"{all_times_sorted[all_p95_idx] * 1000:.2f} ms",
        ],
        [
            "Min Total Time",
            f"{min(all_times) * 1000:.2f} ms",
        ],
        [
            "Max Total Time",
            f"{max(all_times) * 1000:.2f} ms",
        ],
        [
            "Std Dev",
            (
                f"{statistics.stdev(all_times) * 1000:.2f} ms"
                if len(all_times) > 1
                else "N/A"
            ),
        ],
    ]
    pipeline_table = Table(pipeline_data, colWidths=[3 * inch, 2 * inch])
    pipeline_table.setStyle(table_style)
    story.append(pipeline_table)

    # Build PDF
    doc.build(story)


def test_evaluate_all_strategies():
    """Main test function that evaluates strategies and generates PDF report."""
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    test_results, metrics = evaluate_strategies()

    # Generate PDF report
    pdf_path = RESULTS_DIR / "strategy_evaluation.pdf"
    generate_pdf_report(test_results, metrics, pdf_path)

    print(f"✓ PDF report saved to: {pdf_path}")

    # Summary assertions
    assert test_results, "No test results generated"
    assert metrics, "No strategy metrics generated"

    # Verify that at least some tests passed
    passed_count = sum(1 for r in test_results if r.passed)
    assert passed_count > 0, f"No tests passed."
