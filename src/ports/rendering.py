"""Rendering port - Abstraction for map and visualization generation.

This protocol defines the contract for map rendering, allowing
different implementations (Folium, Plotly, etc.) to be used.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from ..domain.models import Station


class MapRendererPort(Protocol):
    """Port for map rendering.

    Replaces: viz/map.py:46-86 (plot_path)
    Implementation: adapters/rendering/folium_adapter.py

    Map renderers visualize routes on interactive maps.
    """

    def render(
        self,
        stations: Sequence[Station],
        output_path: Path,
    ) -> Path:
        """Render a route on a map and save to file.

        Args:
            stations: Sequence of stations forming the route.
            output_path: Where to save the rendered map.

        Returns:
            Path to the generated map file.
        """
        ...
