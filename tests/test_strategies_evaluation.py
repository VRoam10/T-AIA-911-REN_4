"""Strategy evaluation test that measures performance across different NLP and pathfinding approaches.

This test evaluates all combinations of available strategies and generates
a comprehensive results report in PDF format for deliveries.
"""

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from src.graph.dijkstra import dijkstra
from src.graph.load_graph import Graph, load_graph
from src.nlp.extract_stations import StationExtractionResult, extract_stations

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
    "rule_test": extract_stations,  # For testing purposes to show how to add more
}

PATH_FINDER_STRATEGIES: Dict[str, PathFinder] = {
    "dijkstra": dijkstra,
    "dijkstra_test": dijkstra,  # For testing purposes to show how to add more
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


def evaluate_strategies() -> Tuple[List[TestResult], List[StrategyMetrics]]:
    """Evaluate all strategy combinations against all test sentences."""
    test_results: List[TestResult] = []
    metrics_dict: Dict[Tuple[str, str], List[float]] = {}
    metrics_dict_passed: Dict[Tuple[str, str], int] = {}

    sentences = load_sentences()

    # Test each strategy combination
    for nlp_name in NLP_STRATEGIES.keys():
        for path_name in PATH_FINDER_STRATEGIES.keys():
            key = (nlp_name, path_name)
            metrics_dict[key] = []
            metrics_dict_passed[key] = 0

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

                metrics_dict[key].append(total_time)
                if passed:
                    metrics_dict_passed[key] += 1

    # Compute aggregated metrics
    strategy_metrics: List[StrategyMetrics] = []
    for (nlp_name, path_name), times in metrics_dict.items():
        _passed = metrics_dict_passed[(nlp_name, path_name)]
        total = len(sentences)

        metrics = StrategyMetrics(
            nlp_strategy=nlp_name,
            path_strategy=path_name,
            total_tests=total,
            passed_tests=_passed,
            failed_tests=total - _passed,
            accuracy=_passed / total if total > 0 else 0,
            avg_execution_time=sum(times) / len(times) if times else 0,
            min_execution_time=min(times) if times else 0,
            max_execution_time=max(times) if times else 0,
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

    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
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
    summary_table.setStyle(
        TableStyle(
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
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.3 * inch))

    # Strategy Performance
    story.append(Paragraph("Strategy Performance", heading_style))
    for metric in metrics:
        story.append(
            Paragraph(
                f"{metric.nlp_strategy.upper()} NLP + {metric.path_strategy.upper()} Path Finder",
                styles["Heading3"],
            )
        )

        perf_data = [
            [
                "Accuracy",
                f"{metric.accuracy * 100:.2f}% ({metric.passed_tests}/{metric.total_tests})",
            ],
            ["Avg Execution Time", f"{metric.avg_execution_time * 1000:.2f} ms"],
            ["Min Execution Time", f"{metric.min_execution_time * 1000:.2f} ms"],
            ["Max Execution Time", f"{metric.max_execution_time * 1000:.2f} ms"],
        ]
        perf_table = Table(perf_data, colWidths=[2.5 * inch, 2.5 * inch])
        perf_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f4f8")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ]
            )
        )
        story.append(perf_table)
        story.append(Spacer(1, 0.2 * inch))

    story.append(PageBreak())

    # NLP Results
    story.append(Paragraph("NLP Results", heading_style))
    story.append(
        Paragraph(
            "The NLP component extracts departure and arrival stations from user input.",
            styles["Normal"],
        )
    )

    nlp_results = [r for r in results if r.nlp_strategy == "rule_based"]
    nlp_errors = [r for r in nlp_results if r.error]
    nlp_correct = [r for r in nlp_results if not r.error]

    if nlp_results:
        nlp_data = [
            ["Total Inputs Processed", str(len(nlp_results))],
            ["Successfully Extracted", str(len(nlp_correct))],
            ["Extraction Errors", str(len(nlp_errors))],
            [
                "Avg Extraction Time",
                f"{sum(r.nlp_execution_time for r in nlp_results) / len(nlp_results) * 1000:.2f} ms",
            ],
        ]
        nlp_table = Table(nlp_data, colWidths=[3 * inch, 2 * inch])
        nlp_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ]
            )
        )
        story.append(nlp_table)
        story.append(Spacer(1, 0.2 * inch))

    # Path Finder Results
    story.append(Paragraph("Path Finder Results", heading_style))
    story.append(
        Paragraph(
            "The path finder computes the shortest route between two stations.",
            styles["Normal"],
        )
    )

    path_results = [r for r in results if r.path_strategy == "dijkstra" and not r.error]
    if path_results:
        path_data = [
            ["Total Paths Computed", str(len(path_results))],
            [
                "Avg Path Computation Time",
                f"{sum(r.path_execution_time for r in path_results) / len(path_results) * 1000:.2f} ms",
            ],
            [
                "Avg Path Length",
                f"{sum(len(p) if p else 0 for p in [r.path for r in path_results]) / len(path_results):.2f} stations",
            ],
        ]
        path_table = Table(path_data, colWidths=[3 * inch, 2 * inch])
        path_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ]
            )
        )
        story.append(path_table)
        story.append(Spacer(1, 0.2 * inch))

    story.append(PageBreak())

    # Pipeline Results
    story.append(Paragraph("Pipeline Results", heading_style))
    story.append(
        Paragraph(
            "End-to-end pipeline performance combining NLP and path finding.",
            styles["Normal"],
        )
    )

    all_results = results
    total_time = sum(r.total_execution_time for r in all_results)
    avg_time = total_time / len(all_results) if all_results else 0

    pipeline_data = [
        ["Total Pipeline Executions", str(len(all_results))],
        ["Avg Total Time", f"{avg_time * 1000:.2f} ms"],
        [
            "Min Total Time",
            f"{min(r.total_execution_time for r in all_results) * 1000:.2f} ms",
        ],
        [
            "Max Total Time",
            f"{max(r.total_execution_time for r in all_results) * 1000:.2f} ms",
        ],
    ]
    pipeline_table = Table(pipeline_data, colWidths=[3 * inch, 2 * inch])
    pipeline_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ]
        )
    )
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
