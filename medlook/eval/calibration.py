"""Calibration metrics: Expected Calibration Error (ECE), overconfident-error rate, and
the risk-coverage curve + Area Under the Risk-Coverage curve (AURC) for selective
prediction.

All functions operate on plain (confidence, correctness) pairs, so both the local mock
eval path and the real Colab eval path (after `schema.parse` extracts a confidence
score) share this exact code -- there's only one implementation of the math.

Lower AURC and lower overconfident-error are better. Lower ECE is better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


def expected_calibration_error(confidences: Sequence[float], corrects: Sequence[bool], n_bins: int = 10) -> float:
    """Standard binned ECE: sum over bins of |bin_accuracy - bin_avg_confidence|,
    weighted by the fraction of samples in each bin."""
    if len(confidences) != len(corrects):
        raise ValueError("confidences and corrects must be the same length")
    n = len(confidences)
    if n == 0:
        return 0.0

    bins = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, corrects):
        conf = min(max(conf, 0.0), 1.0)
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))

    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        bin_conf = sum(c for c, _ in bucket) / len(bucket)
        bin_acc = sum(1 for _, correct in bucket if correct) / len(bucket)
        ece += (len(bucket) / n) * abs(bin_acc - bin_conf)
    return ece


def overconfident_error_rate(
    confidences: Sequence[float], corrects: Sequence[bool], threshold: float = 0.8
) -> float:
    """Fraction of INCORRECT predictions among all predictions with confidence >=
    threshold. Returns 0.0 (not undefined) if no prediction meets the threshold --
    callers should also check `n_high_confidence` if that distinction matters."""
    high_conf_pairs = [(c, correct) for c, correct in zip(confidences, corrects) if c >= threshold]
    if not high_conf_pairs:
        return 0.0
    wrong = sum(1 for _, correct in high_conf_pairs if not correct)
    return wrong / len(high_conf_pairs)


def risk_coverage_curve(
    confidences: Sequence[float], corrects: Sequence[bool]
) -> List[Tuple[float, float]]:
    """Sorts predictions by DESCENDING confidence and returns (coverage, risk) points,
    where coverage = fraction of all predictions accepted (most confident first) and
    risk = error rate among accepted predictions. This is the standard selective
    prediction curve used to evaluate systems that can abstain (FLAG_UNCERTAIN /
    ESCALATE) rather than just accuracy alone."""
    n = len(confidences)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: confidences[i], reverse=True)
    points = []
    cumulative_errors = 0
    for i, idx in enumerate(order, start=1):
        if not corrects[idx]:
            cumulative_errors += 1
        coverage = i / n
        risk = cumulative_errors / i
        points.append((coverage, risk))
    return points


def area_under_risk_coverage(confidences: Sequence[float], corrects: Sequence[bool]) -> float:
    """Trapezoidal AURC over the risk-coverage curve. Lower is better (a lower-AURC
    system has lower error rates across all coverage levels, i.e. its confidence score
    is a more useful signal for deciding when to abstain)."""
    curve = risk_coverage_curve(confidences, corrects)
    if not curve:
        return 0.0
    area = 0.0
    prev_coverage, prev_risk = 0.0, curve[0][1]
    for coverage, risk in curve:
        area += (coverage - prev_coverage) * (risk + prev_risk) / 2.0
        prev_coverage, prev_risk = coverage, risk
    return area


@dataclass
class CalibrationReport:
    n: int
    ece: float
    overconfident_error_rate: float
    aurc: float
    risk_coverage_curve: List[Tuple[float, float]]

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "ece": self.ece,
            "overconfident_error_rate": self.overconfident_error_rate,
            "aurc": self.aurc,
            # Downsampled for report readability; full-resolution curve is recomputable
            # from raw (confidence, correct) pairs if ever needed for a plot.
            "risk_coverage_curve": _downsample_curve(self.risk_coverage_curve),
        }


def _downsample_curve(curve: List[Tuple[float, float]], max_points: int = 20) -> List[Tuple[float, float]]:
    if len(curve) <= max_points:
        return curve
    step = len(curve) / max_points
    indices = sorted({min(int(i * step), len(curve) - 1) for i in range(max_points)} | {len(curve) - 1})
    return [curve[i] for i in indices]


def calibration_report(
    confidences: Sequence[float],
    corrects: Sequence[bool],
    n_bins: int = 10,
    overconf_threshold: float = 0.8,
) -> CalibrationReport:
    return CalibrationReport(
        n=len(confidences),
        ece=expected_calibration_error(confidences, corrects, n_bins=n_bins),
        overconfident_error_rate=overconfident_error_rate(confidences, corrects, threshold=overconf_threshold),
        aurc=area_under_risk_coverage(confidences, corrects),
        risk_coverage_curve=risk_coverage_curve(confidences, corrects),
    )
