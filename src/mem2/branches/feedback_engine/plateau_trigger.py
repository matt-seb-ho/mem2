"""Plateau trigger: detects when a rolling window of outcomes stops improving.

Used by the reorg builder (axis A4) to decide when to kick off a global
rebuild instead of continuing accretive writes.

The trigger is stateless-per-call — callers pass the history they own and
get back a decision. This keeps the trigger testable and independent of the
orchestrator's retry loop.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlateauDecision:
    should_trigger: bool
    reason: str
    window_start: int
    window_end: int
    window_mean: float
    prior_mean: float
    delta: float


def detect_plateau(
    outcomes: list[float],
    *,
    window: int = 10,
    min_delta: float = 0.01,
) -> PlateauDecision:
    """Trigger when the latest ``window`` outcomes improve <= ``min_delta`` over
    the previous ``window`` outcomes.

    Parameters
    ----------
    outcomes : list[float]
        Per-step outcome scores in chronological order (e.g. pass-rate, 0/1).
    window : int
        Rolling window length. Needs ``2 * window`` observations to fire.
    min_delta : float
        Minimum delta (window mean - prior mean) to count as "still improving".

    Returns
    -------
    PlateauDecision
    """
    n = len(outcomes)
    if n < 2 * window:
        return PlateauDecision(
            should_trigger=False,
            reason=f"insufficient history: {n} < {2 * window}",
            window_start=-1,
            window_end=-1,
            window_mean=0.0,
            prior_mean=0.0,
            delta=0.0,
        )
    prior = outcomes[n - 2 * window : n - window]
    recent = outcomes[n - window : n]
    prior_mean = sum(prior) / window
    recent_mean = sum(recent) / window
    delta = recent_mean - prior_mean
    should = delta <= min_delta
    return PlateauDecision(
        should_trigger=should,
        reason=(
            f"delta {delta:.4f} <= {min_delta:.4f} over window={window}"
            if should
            else f"still improving: delta {delta:.4f} > {min_delta:.4f}"
        ),
        window_start=n - window,
        window_end=n,
        window_mean=recent_mean,
        prior_mean=prior_mean,
        delta=delta,
    )


def detect_plateau_every_k(step: int, *, k: int = 20) -> PlateauDecision:
    """Trigger every ``k`` steps regardless of outcomes. Counterfactual to
    ``detect_plateau`` — used for axis A4 as the "every-k" variant."""
    should = step > 0 and step % k == 0
    return PlateauDecision(
        should_trigger=should,
        reason=f"every-{k} step={step}",
        window_start=max(0, step - k),
        window_end=step,
        window_mean=0.0,
        prior_mean=0.0,
        delta=0.0,
    )
