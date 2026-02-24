"""Format-independent retrieval routing gate.

Decides whether to include hints for a given problem. Works on concept names
and hint text — knows nothing about any specific memory format. Any retriever
can compose this.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoutingDecision:
    """Result of a routing gate evaluation.

    ``include`` is the gate verdict; ``reasons`` records which conditions
    triggered a skip (empty when ``include=True``).  Supports ``bool()``
    so existing ``if not router.should_include(...)`` patterns keep working.
    """

    include: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.include


class RetrievalRouter:
    """Routing gate: decides whether to include hints for a problem.

    Strategies:
    - ``"none"``: always include hints.
    - ``"selection_confidence"``: skip if all selected concepts are
      high-frequency (generic).
    - ``"hint_length"``: skip if rendered hint text exceeds a character limit.

    Composite thresholds (AND logic — any fail → skip):
    - ``max_concept_count``: skip if number of selected concepts exceeds limit.
    - ``max_pre_filter_count``: skip if pre-filter concept count exceeds limit.
    - ``max_hint_chars``: skip if rendered hint text exceeds character limit.

    When only ``strategy`` is set (no new thresholds), behaves exactly as
    before.  When any threshold is set, those conditions are always evaluated
    alongside ``strategy``.

    Parameters
    ----------
    strategy : str
        Routing strategy name.
    frequency_threshold : float
        For ``selection_confidence``: concepts above this frequency are
        considered generic.
    max_hint_chars : int
        Skip hints exceeding this char count. 0 = disabled.
    max_concept_count : int
        Skip if ``len(names)`` exceeds this. 0 = disabled.
    max_pre_filter_count : int
        Skip if ``pre_filter_count`` exceeds this. 0 = disabled.
    frequencies : dict
        Concept name → selection fraction. Shared from ConceptFilter.
    """

    def __init__(
        self,
        strategy: str = "none",
        frequency_threshold: float = 0.5,
        max_hint_chars: int = 0,
        max_concept_count: int = 0,
        max_pre_filter_count: int = 0,
        frequencies: dict[str, float] | None = None,
    ):
        self.strategy = strategy
        self.frequency_threshold = float(frequency_threshold)
        self.max_hint_chars = int(max_hint_chars)
        self.max_concept_count = int(max_concept_count)
        self.max_pre_filter_count = int(max_pre_filter_count)
        self._frequencies = frequencies or {}

    # ------------------------------------------------------------------ #
    #  Internal: check whether any composite threshold is configured       #
    # ------------------------------------------------------------------ #
    def _has_composite_thresholds(self) -> bool:
        return (
            self.max_concept_count > 0
            or self.max_pre_filter_count > 0
            or self.max_hint_chars > 0
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def should_include(
        self,
        names: list[str] | None,
        hint_text: str | None,
        *,
        pre_filter_count: int = 0,
    ) -> RoutingDecision:
        """Return a ``RoutingDecision`` indicating whether hints should be included.

        All enabled conditions are evaluated with AND logic — any failure
        causes a skip.  The ``reasons`` list records which conditions fired.
        """
        # Fast-path: no composite thresholds AND strategy is "none"
        if self.strategy == "none" and not self._has_composite_thresholds():
            return RoutingDecision(include=True)

        reasons: list[str] = []

        # ── Composite threshold: concept count ─────────────────────────
        if self.max_concept_count > 0 and names:
            n = len(names)
            if n > self.max_concept_count:
                reasons.append(f"concept_count:{n}>{self.max_concept_count}")

        # ── Composite threshold: pre-filter count ──────────────────────
        if self.max_pre_filter_count > 0 and pre_filter_count > self.max_pre_filter_count:
            reasons.append(
                f"pre_filter_count:{pre_filter_count}>{self.max_pre_filter_count}"
            )

        # ── Composite threshold: hint length ───────────────────────────
        if self.max_hint_chars > 0 and hint_text and len(hint_text) > self.max_hint_chars:
            reasons.append(
                f"hint_chars:{len(hint_text)}>{self.max_hint_chars}"
            )

        # ── Strategy-based gate ────────────────────────────────────────
        if self.strategy == "selection_confidence":
            if names and self._frequencies:
                all_generic = all(
                    self._frequencies.get(n, 0.0) > self.frequency_threshold
                    for n in names
                )
                if all_generic:
                    reasons.append("all_generic")

        elif self.strategy == "hint_length":
            # Legacy: strategy="hint_length" uses max_hint_chars as its gate.
            # Already handled above in composite threshold section.
            # Only add reason if not already caught by composite check.
            if (
                self.max_hint_chars > 0
                and hint_text
                and len(hint_text) > self.max_hint_chars
                and not any(r.startswith("hint_chars:") for r in reasons)
            ):
                reasons.append(
                    f"hint_chars:{len(hint_text)}>{self.max_hint_chars}"
                )

        if reasons:
            return RoutingDecision(include=False, reasons=reasons)
        return RoutingDecision(include=True)
