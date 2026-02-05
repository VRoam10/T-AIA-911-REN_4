"""Adapters layer - Concrete implementations of ports.

This module contains implementations of the port interfaces,
connecting the application to external systems like:
- NLP models (SpaCy, HuggingFace, rule-based)
- Geocoding services (Nominatim)
- Graph storage (CSV files)
- ASR models (Whisper)
- Rendering engines (Folium)
- Caching systems (in-memory, null)
"""
