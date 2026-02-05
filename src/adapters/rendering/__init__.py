"""Rendering adapters - Implementations of MapRendererPort.

Available implementations:
- FoliumMapRenderer: Folium-based interactive map rendering
"""

from .folium_adapter import FoliumMapRenderer

__all__ = ["FoliumMapRenderer"]
