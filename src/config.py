"""Centralized configuration using Pydantic Settings.

This module provides a single source of truth for all configuration,
replacing hardcoded values scattered across the codebase:
- pipeline.py:26-29 (paths)
- hf_ner/ner.py:13-14 (model IDs)
- asr.py device/compute_type defaults
- strategies.py geocoding settings

Configuration can be overridden via environment variables:
- TOR_NLP_DEFAULT_STRATEGY=hf_ner
- TOR_ASR_DEVICE=cpu
- TOR_GRAPH_DATA_DIR=/path/to/data
- etc.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NLPConfig(BaseSettings):
    """NLP-related configuration.

    Environment variables prefixed with TOR_NLP_.
    """

    model_config = SettingsConfigDict(env_prefix="TOR_NLP_")

    default_strategy: Literal["rule_based", "hf_ner", "spacy"] = "rule_based"
    hf_ner_model: str = "Jean-Baptiste/camembert-ner"
    hf_ner_dates_model: str = "Jean-Baptiste/camembert-ner-with-dates"
    spacy_model: str = "fr_core_news_md"


class ASRConfig(BaseSettings):
    """ASR-related configuration.

    Environment variables prefixed with TOR_ASR_.
    """

    model_config = SettingsConfigDict(env_prefix="TOR_ASR_")

    default_model: str = "large-v3"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    compute_type: str = "float16"
    fallback_device: str = "cpu"
    fallback_compute_type: str = "int8"
    beam_size: int = 5


class GraphConfig(BaseSettings):
    """Graph data configuration.

    Environment variables prefixed with TOR_GRAPH_.
    """

    model_config = SettingsConfigDict(env_prefix="TOR_GRAPH_")

    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data"
    )
    stations_file: str = "stations.csv"
    edges_file: str = "edges.csv"

    @property
    def stations_path(self) -> Path:
        """Full path to stations CSV file."""
        return self.data_dir / self.stations_file

    @property
    def edges_path(self) -> Path:
        """Full path to edges CSV file."""
        return self.data_dir / self.edges_file


class GeocodingConfig(BaseSettings):
    """Geocoding configuration.

    Environment variables prefixed with TOR_GEO_.
    """

    model_config = SettingsConfigDict(env_prefix="TOR_GEO_")

    user_agent: str = "travel-order-resolver"
    timeout_seconds: int = 10
    rate_limit_delay: float = 1.0
    max_retries: int = 2
    error_wait_seconds: float = 2.0


class ObservabilityConfig(BaseSettings):
    """Logging and observability configuration.

    Environment variables prefixed with TOR_LOG_.
    """

    model_config = SettingsConfigDict(env_prefix="TOR_LOG_")

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    structured: bool = False  # Set True for JSON logging


class AppConfig(BaseSettings):
    """Main application configuration aggregating all sub-configs.

    This is the main entry point for configuration. Sub-configurations
    can be accessed via attributes:

        config = get_config()
        print(config.nlp.default_strategy)
        print(config.graph.stations_path)

    Environment variables prefixed with TOR_.
    """

    model_config = SettingsConfigDict(env_prefix="TOR_")

    nlp: NLPConfig = Field(default_factory=NLPConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    geocoding: GeocodingConfig = Field(default_factory=GeocodingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    output_dir: Path = Field(default_factory=Path.cwd)

    @property
    def project_root(self) -> Path:
        """Return the project root directory."""
        return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Get the singleton application configuration.

    Configuration is loaded once and cached. To reload configuration
    (e.g., in tests), use reset_config() first.

    Returns:
        The application configuration instance.
    """
    return AppConfig()


def reset_config() -> None:
    """Reset the configuration cache.

    Call this in tests to ensure a fresh configuration is loaded.
    """
    get_config.cache_clear()
