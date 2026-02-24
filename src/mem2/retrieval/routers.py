"""Format-independent retrieval routing gate.

Decides whether to include hints for a given problem. Works on concept names
and hint text — knows nothing about any specific memory format. Any retriever
can compose this.
"""
from __future__ import annotations


class RetrievalRouter:
    """Routing gate: decides whether to include hints for a problem.

    Strategies:
    - ``"none"``: always include hints.
    - ``"selection_confidence"``: skip if all selected concepts are
      high-frequency (generic).
    - ``"hint_length"``: skip if rendered hint text exceeds a character limit.

    Parameters
    ----------
    strategy : str
        Routing strategy name.
    frequency_threshold : float
        For ``selection_confidence``: concepts above this frequency are
        considered generic.
    max_hint_chars : int
        For ``hint_length``: skip hints exceeding this char count. 0 = disabled.
    frequencies : dict
        Concept name → selection fraction. Shared from ConceptFilter.
    """

    def __init__(
        self,
        strategy: str = "none",
        frequency_threshold: float = 0.5,
        max_hint_chars: int = 0,
        frequencies: dict[str, float] | None = None,
    ):
        self.strategy = strategy
        self.frequency_threshold = float(frequency_threshold)
        self.max_hint_chars = int(max_hint_chars)
        self._frequencies = frequencies or {}

    def should_include(
        self, names: list[str] | None, hint_text: str | None
    ) -> bool:
        """Return True if hints should be included for this problem."""
        if self.strategy == "none":
            return True

        if self.strategy == "selection_confidence":
            if not names or not self._frequencies:
                return True
            all_generic = all(
                self._frequencies.get(n, 0.0) > self.frequency_threshold
                for n in names
            )
            return not all_generic

        if self.strategy == "hint_length":
            if not hint_text:
                return True
            if self.max_hint_chars > 0 and len(hint_text) > self.max_hint_chars:
                return False
            return True

        return True
