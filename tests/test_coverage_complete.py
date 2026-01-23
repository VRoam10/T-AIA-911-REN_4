"""Extra tests to exercise remaining branches for full coverage."""

from __future__ import annotations

import builtins
import importlib.util
import io

import pytest

from src.graph.dijkstra import dijkstra
from src.nlp import extract_stations as extract_module
from src.nlp.intent import _basic_french_detection, _is_french, _is_travel_request
import src.pipeline as pipeline
from src.pipeline import (
    NLP_STRATEGIES,
    PATH_FINDER_STRATEGIES,
    run_pipeline,
    solve_travel_order,
)


def test_dijkstra_invalid_nodes():
    graph = {"A": [("B", 1.0)]}
    path, distance = dijkstra(graph, "A", "C")
    assert path == []
    assert distance == float("inf")


def test_dijkstra_skips_visited_node():
    graph = {
        "A": [("B", 5.0), ("C", 1.0)],
        "B": [],
        "C": [("B", 1.0)],
        "D": [],
    }
    path, distance = dijkstra(graph, "A", "D")
    assert path == []
    assert distance == float("inf")


def test_load_stations_skips_invalid_rows(monkeypatch):
    extract_module._load_stations.cache_clear()

    csv_data = "\n".join(
        [
            "station_id,station_name",
            ",Paris",
            "PAR,",
            "LYO,Lyon",
        ]
    )

    def fake_open(self, *args, **kwargs):
        return io.StringIO(csv_data)

    monkeypatch.setattr(extract_module.Path, "open", fake_open)
    stations = extract_module._load_stations()
    assert stations == {"lyon": "LYO"}
    extract_module._load_stations.cache_clear()


def test_load_stations_oserror_returns_empty(monkeypatch):
    extract_module._load_stations.cache_clear()

    def fake_open(self, *args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(extract_module.Path, "open", fake_open)
    stations = extract_module._load_stations()
    assert stations == {}
    extract_module._load_stations.cache_clear()


def test_intent_importerror_falls_back(monkeypatch):
    import src.nlp.intent as intent_module

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langdetect":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location(
        "intent_no_langdetect", intent_module.__file__
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.detect is None
    assert module._is_french("Bonjour, comment allez-vous ?")

    monkeypatch.setattr(builtins, "__import__", original_import)


def test_is_french_short_langdetect_non_fr(monkeypatch):
    import src.nlp.intent as intent_module

    monkeypatch.setattr(intent_module, "detect", lambda _: "en")
    assert not _is_french("Hello there")


def test_is_french_langdetect_exception_fallback(monkeypatch):
    import src.nlp.intent as intent_module

    def boom(_text: str) -> str:
        raise RuntimeError("detect failed")

    monkeypatch.setattr(intent_module, "detect", boom)
    assert _is_french("Bonjour comment allez vous")


def test_is_french_short_langdetect_fr(monkeypatch):
    import src.nlp.intent as intent_module

    monkeypatch.setattr(intent_module, "detect", lambda _: "fr")
    assert _is_french("Bonjour!")


def test_basic_detection_no_words():
    assert not _basic_french_detection("!!!")


def test_basic_detection_low_french_density():
    assert not _basic_french_detection("trajet hello world test")


def test_basic_detection_short_travel_term():
    assert _basic_french_detection("trajet")


def test_travel_request_dash_pattern(monkeypatch):
    import src.nlp.intent as intent_module

    target = (
        r"^\s*(?:trajet|itin[ée]raire|chemin|route|direction)\s+"
        r"[\w\s-]+\s*[-–>]\s*[\w\s-]+\s*$"
    )
    original_search = intent_module.re.search

    def fake_search(pattern, text, flags=0):
        if pattern == target:
            return original_search(pattern, text, flags)
        return None

    monkeypatch.setattr(intent_module.re, "search", fake_search)
    assert _is_travel_request("direction Paris - Lyon")


def test_travel_request_keyword_count(monkeypatch):
    import src.nlp.intent as intent_module

    original_search = intent_module.re.search

    def fake_search(pattern, text, flags=0):
        if "(?:" in pattern:
            return None
        return original_search(pattern, text, flags)

    monkeypatch.setattr(intent_module.re, "search", fake_search)
    assert _is_travel_request("itinéraire Paris trajet")


def test_travel_request_de_a_pattern(monkeypatch):
    import src.nlp.intent as intent_module

    target = (
        r"\b(?:de|depuis|partir de)\s+[\w\s-]+\s+"
        r"(?:à|vers|jusqu\'?à|pour)\s+[\w\s-]+"
    )
    original_search = intent_module.re.search

    def fake_search(pattern, text, flags=0):
        if pattern == target:
            return original_search(pattern, text, flags)
        return None

    monkeypatch.setattr(intent_module.re, "search", fake_search)
    assert _is_travel_request("de Paris à Lyon")


def test_solve_travel_order_unknown_strategies(monkeypatch):
    assert "Unknown NLP strategy" in solve_travel_order("test", nlp_name="missing")
    assert "Unknown path-finding strategy" in solve_travel_order(
        "test",
        path_name="missing",
    )


def test_solve_travel_order_error_and_no_path(monkeypatch):
    from src.nlp.extract_stations import StationExtractionResult

    def fake_nlp(_sentence: str) -> StationExtractionResult:
        return StationExtractionResult(None, None, "Bad parse")

    def fake_path(_graph, _start, _end):
        return [], 0.0

    monkeypatch.setitem(NLP_STRATEGIES, "fake", fake_nlp)
    monkeypatch.setitem(PATH_FINDER_STRATEGIES, "fake", fake_path)
    assert "Extraction error" in solve_travel_order("test", "fake", "fake")

    def ok_nlp(_sentence: str) -> StationExtractionResult:
        return StationExtractionResult("AAA", "BBB", None)

    monkeypatch.setitem(NLP_STRATEGIES, "ok", ok_nlp)
    assert "No path found between AAA and BBB." in solve_travel_order(
        "test",
        "ok",
        "fake",
    )


def test_solve_travel_order_missing_departure(monkeypatch):
    from src.nlp.extract_stations import StationExtractionResult

    def bad_nlp(_sentence: str) -> StationExtractionResult:
        return StationExtractionResult(None, "BBB", None)

    monkeypatch.setitem(NLP_STRATEGIES, "bad", bad_nlp)
    with pytest.raises(ValueError):
        solve_travel_order("test", "bad", "dijkstra")


def test_solve_travel_order_success(monkeypatch):
    from src.nlp.extract_stations import StationExtractionResult

    def ok_nlp(_sentence: str) -> StationExtractionResult:
        return StationExtractionResult("AAA", "BBB", None)

    def ok_path(_graph, _start, _end):
        return ["AAA", "BBB"], 42.0

    monkeypatch.setitem(NLP_STRATEGIES, "ok2", ok_nlp)
    monkeypatch.setitem(PATH_FINDER_STRATEGIES, "ok2", ok_path)
    monkeypatch.setattr(
        pipeline,
        "load_graph",
        lambda *_args, **_kwargs: {"AAA": [("BBB", 42.0)], "BBB": []},
    )
    result = solve_travel_order("test", "ok2", "ok2")
    assert "Shortest path: AAA -> BBB" in result
    assert "Total distance: 42.0 km" in result


def test_run_pipeline_prints(capsys, monkeypatch):
    monkeypatch.setattr(pipeline, "solve_travel_order", lambda _s: "OK")
    run_pipeline()
    captured = capsys.readouterr()
    assert "Sentence:" in captured.out
    assert "OK" in captured.out


def test_pipeline_main_executes(capsys):
    import runpy

    runpy.run_module("src.pipeline", run_name="__main__")
    captured = capsys.readouterr()
    assert "Sentence:" in captured.out
