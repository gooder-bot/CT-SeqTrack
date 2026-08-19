"""Pure expert selection for optional RA-PMM top-2 support."""

import numpy as np


def select_secondary_expert(
    prediction,
    fixed_margins,
    max_margins,
    coverage_scale,
    fallback_sigma_parallel_perpendicular,
):
    probabilities = np.asarray(
        prediction.get("mode_probabilities", ()), dtype=np.float64
    ).reshape(-1)
    centers = np.asarray(prediction.get("mode_centers_xy", ()), dtype=np.float64)
    if probabilities.size != 3 or centers.shape != (3, 2):
        return None
    if not np.isfinite(probabilities).all() or not np.isfinite(centers).all():
        return None
    order = np.argsort(-probabilities)
    first, second = int(order[0]), int(order[1])
    separation = float(np.linalg.norm(centers[first] - centers[second]))
    if probabilities[second] < 0.15 or separation < 0.5:
        return None
    quantiles = np.asarray(
        prediction.get("support_quantiles_pp", ()), dtype=np.float64
    )
    if quantiles.shape == (3, 2) and np.isfinite(quantiles).all():
        margin = np.clip(quantiles[2], fixed_margins, max_margins)
    else:
        margin = coverage_scale * np.asarray(
            fallback_sigma_parallel_perpendicular, dtype=np.float64
        )
    alpha = float(
        np.clip(
            probabilities[second]
            / max(probabilities[first] + probabilities[second], 1e-8),
            0.25,
            0.40,
        )
    )
    return {
        "center_xy": centers[second],
        "probability": float(probabilities[second]),
        "separation": separation,
        "quota_fraction": alpha,
        "margin": margin,
    }


__all__ = ["select_secondary_expert"]
