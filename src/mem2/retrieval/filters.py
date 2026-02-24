"""Format-independent concept filtering.

Works on concept names (strings) and frequency metadata. Knows nothing about
ConceptMemory, OE entries, or any specific memory format. Any retriever can
compose this.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConceptFilter:
    """Filters a list of concept names by frequency and count cap.

    Parameters
    ----------
    max_frequency : float
        Drop concepts whose selection frequency exceeds this threshold.
        0 = disabled (no frequency filtering).
    max_concepts : int
        Keep at most this many concepts. 0 = no limit.
    frequency_file : str
        Path to JSON mapping concept names to selection fractions.
    """

    def __init__(
        self,
        max_frequency: float = 0.0,
        max_concepts: int = 0,
        frequency_file: str = "",
    ):
        self.max_frequency = float(max_frequency)
        self.max_concepts = int(max_concepts)
        self._frequencies: dict[str, float] = {}
        if frequency_file:
            path = Path(frequency_file)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                self._frequencies = json.loads(path.read_text())
                logger.info(
                    f"Loaded concept frequencies for {len(self._frequencies)} concepts"
                )

    @property
    def frequencies(self) -> dict[str, float]:
        """Loaded frequency data (concept_name → fraction)."""
        return self._frequencies

    def filter(self, names: list[str]) -> list[str]:
        """Filter concept names. Returns filtered list (preserves order)."""
        result = names
        if self.max_frequency > 0.0 and self._frequencies:
            result = [
                n for n in result
                if self._frequencies.get(n, 0.0) <= self.max_frequency
            ]
        if self.max_concepts > 0:
            result = result[:self.max_concepts]
        return result
