"""Bjøntegaard delta-rate utilities for monotonic RD curves."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator


def pareto_front(rate, quality) -> tuple[np.ndarray, np.ndarray]:
    """Keep strictly improving quality points as rate increases."""
    rate = np.asarray(rate, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    if rate.ndim != 1 or quality.ndim != 1 or len(rate) != len(quality):
        raise ValueError("rate and quality must be one-dimensional arrays of equal length")
    if np.any(rate <= 0) or not np.isfinite(rate).all() or not np.isfinite(quality).all():
        raise ValueError("rate must be positive and all RD values must be finite")

    order = np.argsort(rate)
    rate = rate[order]
    quality = quality[order]
    selected = []
    best_quality = -np.inf
    for index, value in enumerate(quality):
        if value > best_quality:
            selected.append(index)
            best_quality = value
    if len(selected) < 4:
        raise ValueError(
            "BD-rate requires four Pareto-optimal points; "
            "check non-monotonic mAP values or choose different rate points"
        )
    return rate[selected], quality[selected]


def _as_curve(rate, quality) -> tuple[np.ndarray, np.ndarray]:
    rate = np.asarray(rate, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    if rate.ndim != 1 or quality.ndim != 1 or len(rate) != len(quality):
        raise ValueError("rate and quality must be one-dimensional arrays of equal length")
    if len(rate) < 4:
        raise ValueError("BD-rate requires at least four RD points")
    if np.any(rate <= 0) or not np.isfinite(rate).all() or not np.isfinite(quality).all():
        raise ValueError("rate must be positive and all RD values must be finite")

    order = np.argsort(quality)
    quality, log_rate = quality[order], np.log10(rate[order])
    if np.any(np.diff(quality) <= 0):
        raise ValueError("quality values must be strictly monotonic for PCHIP BD-rate")
    return log_rate, quality


def compute_bd_rate(anchor_rate, anchor_quality, candidate_rate, candidate_quality) -> float:
    """Return average candidate bitrate change (%) at equal quality.

    A negative result means the candidate needs fewer bits than the anchor.
    """
    anchor_log_rate, anchor_quality = _as_curve(anchor_rate, anchor_quality)
    candidate_log_rate, candidate_quality = _as_curve(candidate_rate, candidate_quality)
    lower = max(anchor_quality.min(), candidate_quality.min())
    upper = min(anchor_quality.max(), candidate_quality.max())
    if lower >= upper:
        raise ValueError("The two RD curves do not overlap in quality")

    anchor = PchipInterpolator(anchor_quality, anchor_log_rate)
    candidate = PchipInterpolator(candidate_quality, candidate_log_rate)
    average_difference = (quad(candidate, lower, upper)[0] - quad(anchor, lower, upper)[0]) / (upper - lower)
    return float((10**average_difference - 1) * 100)


def compute_bd_metric(anchor_rate, anchor_quality, candidate_rate, candidate_quality) -> float:
    """Return average candidate quality gain at equal logarithmic bitrate."""
    anchor_log_rate, anchor_quality = _as_curve(anchor_rate, anchor_quality)
    candidate_log_rate, candidate_quality = _as_curve(candidate_rate, candidate_quality)
    lower = max(anchor_log_rate.min(), candidate_log_rate.min())
    upper = min(anchor_log_rate.max(), candidate_log_rate.max())
    if lower >= upper:
        raise ValueError("The two RD curves do not overlap in bitrate")

    if np.any(np.diff(anchor_log_rate) <= 0) or np.any(np.diff(candidate_log_rate) <= 0):
        raise ValueError("rate must increase with quality for BD-metric")
    anchor = PchipInterpolator(anchor_log_rate, anchor_quality)
    candidate = PchipInterpolator(candidate_log_rate, candidate_quality)
    return float((quad(candidate, lower, upper)[0] - quad(anchor, lower, upper)[0]) / (upper - lower))
