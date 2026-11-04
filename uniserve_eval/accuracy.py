"""Computes the two-proportion comparison the quality protocol accepts.

The design's quality confirmation runs one benchmark point twice, on the same
checkpoint, sample set and sampling settings, and accepts the candidate when
its accuracy difference from the reference lies within the 95% confidence
interval of the difference of two binomial proportions at that sample size.
Equivalently, the two accuracies must not differ by more than the interval's
half-width, so the interval of the difference contains zero.

The interval is the usual large-sample (Wald) interval for the difference of
two independent proportions::

    d = p_candidate - p_reference
    SE = sqrt(p_reference (1 - p_reference) / n_reference
              + p_candidate (1 - p_candidate) / n_candidate)
    interval = d +/- z * SE

with ``z`` the standard normal quantile of the requested confidence. The
quality runs use the complete GSM8K test split (1319 rows) at accuracies near
0.9, where the normal approximation is accurate; the module reports the
standard error and the half-width so a reader can judge the margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist


@dataclass(frozen=True)
class Proportion:
    """A binomial outcome count with the size of its sample.

    Attributes:
        correct: Samples answered correctly.
        total: Samples scored, including incorrect and failed ones.
    """

    correct: int
    total: int

    def __post_init__(self) -> None:
        """Reject counts outside the sample."""
        if self.total < 1:
            raise ValueError("a proportion needs at least one sample")
        if not 0 <= self.correct <= self.total:
            raise ValueError("correct must lie within [0, total]")

    @property
    def value(self) -> float:
        """Return the observed proportion."""
        return self.correct / self.total


@dataclass(frozen=True)
class DifferenceInterval:
    """The confidence interval of a difference of two proportions.

    Attributes:
        difference: Candidate accuracy minus reference accuracy.
        standard_error: Standard error of the difference.
        half_width: ``z * standard_error``, half the interval's length.
        low: Lower end of the interval.
        high: Upper end of the interval.
        accepted: Whether the difference lies within the interval's
            half-width of zero, i.e. whether the interval contains zero.
    """

    difference: float
    standard_error: float
    half_width: float
    low: float
    high: float
    accepted: bool


def difference_interval(
    reference: Proportion,
    candidate: Proportion,
    *,
    confidence: float = 0.95,
) -> DifferenceInterval:
    """Return the confidence interval of a difference of two accuracies.

    ``reference`` is the run accepted as the baseline and ``candidate`` the
    run under confirmation, so a positive difference means the candidate
    scored higher. The two runs are treated as independent samples, which
    holds because both score the same fixed sample set with the same
    checkpoint and settings.

    Args:
        reference: The baseline outcome counts.
        candidate: The candidate outcome counts.
        confidence: Two-sided confidence level in (0, 1).

    Returns:
        The difference, its standard error, the interval, and whether the
        difference is within the interval's half-width of zero.

    Raises:
        ValueError: If ``confidence`` is outside (0, 1).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    difference = candidate.value - reference.value
    variance = (
        reference.value * (1.0 - reference.value) / reference.total
        + candidate.value * (1.0 - candidate.value) / candidate.total
    )
    standard_error = sqrt(variance)
    # The two-sided quantile: the normal inverse at 0.5 plus half the
    # interval's mass on each side.
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    half_width = z * standard_error
    # A degenerate sample (every outcome equal in both runs) has no spread;
    # only an identical accuracy then lies within the interval.
    accepted = abs(difference) <= half_width
    return DifferenceInterval(
        difference=difference,
        standard_error=standard_error,
        half_width=half_width,
        low=difference - half_width,
        high=difference + half_width,
        accepted=accepted,
    )
