"""Shared implementations of paper-derived perturbation-model metrics.

Meta-metrics are intentionally excluded. This module scores model predictions;
it does not qualify or calibrate the scoring protocols.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Mapping

import numpy as np
from scipy.spatial.distance import cdist


Array = np.ndarray


METRIC_DIRECTIONS: dict[str, str] = {
    "mse": "minimize",
    "weighted_mse": "minimize",
    "pearson_delta_control": "maximize",
    "pearson_delta_control_degs": "maximize",
    "pearson_delta_perturbed_mean": "maximize",
    "pearson_delta_perturbed_mean_degs": "maximize",
    "r2_delta_control": "maximize",
    "r2_delta_control_degs": "maximize",
    "r2_delta_perturbed_mean": "maximize",
    "r2_delta_perturbed_mean_degs": "maximize",
    "weighted_r2_delta_control": "maximize",
    "weighted_r2_delta_perturbed_mean": "maximize",
    "normalized_inverse_rank": "maximize",
    "centroid_accuracy": "maximize",
}


def _as_2d(values: Array, name: str) -> Array:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return array


def _check_same_shape(**arrays: Array) -> None:
    shapes = {name: np.asarray(value).shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Arrays must have identical shapes, got {shapes}")


def _rowwise_mse(truth: Array, prediction: Array, weights: Array | None = None) -> Array:
    _check_same_shape(truth=truth, prediction=prediction)
    squared_error = np.square(truth - prediction)
    if weights is None:
        return np.nanmean(squared_error, axis=1)
    _check_same_shape(truth=truth, weights=weights)
    valid_weights = np.where(np.isfinite(weights) & (weights >= 0), weights, 0)
    denominator = valid_weights.sum(axis=1)
    numerator = np.nansum(valid_weights * squared_error, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(truth.shape[0], np.nan, dtype=float),
        where=denominator > 0,
    )


def _rowwise_pearson(truth: Array, prediction: Array, mask: Array | None = None) -> Array:
    _check_same_shape(truth=truth, prediction=prediction)
    result = np.full(truth.shape[0], np.nan, dtype=float)
    for row in range(truth.shape[0]):
        keep = np.isfinite(truth[row]) & np.isfinite(prediction[row])
        if mask is not None:
            keep &= mask[row]
        minimum_features = 3 if mask is not None else 2
        if keep.sum() < minimum_features:
            continue
        observed = truth[row, keep]
        predicted = prediction[row, keep]
        observed = observed - observed.mean()
        predicted = predicted - predicted.mean()
        denominator = np.sqrt(
            np.sum(np.square(observed)) * np.sum(np.square(predicted))
        )
        if denominator > 0:
            result[row] = np.sum(observed * predicted) / denominator
    return result


def _rowwise_r2(
    truth: Array,
    prediction: Array,
    mask: Array | None = None,
    weights: Array | None = None,
) -> Array:
    _check_same_shape(truth=truth, prediction=prediction)
    result = np.full(truth.shape[0], np.nan, dtype=float)
    for row in range(truth.shape[0]):
        keep = np.isfinite(truth[row]) & np.isfinite(prediction[row])
        if mask is not None:
            keep &= mask[row]
        minimum_features = 3 if mask is not None else 2
        if keep.sum() < minimum_features:
            continue
        observed = truth[row, keep]
        predicted = prediction[row, keep]
        if weights is None:
            row_weights = np.ones_like(observed)
        else:
            row_weights = np.asarray(weights[row, keep], dtype=float)
            row_weights = np.where(
                np.isfinite(row_weights) & (row_weights >= 0), row_weights, 0
            )
        weight_sum = row_weights.sum()
        if weight_sum <= 0:
            continue
        weighted_mean = np.sum(row_weights * observed) / weight_sum
        denominator = np.sum(row_weights * np.square(observed - weighted_mean))
        if denominator > 0:
            result[row] = 1 - np.sum(
                row_weights * np.square(observed - predicted)
            ) / denominator
    return result


def _candidate_groups(
    n_conditions: int,
    candidate_groups: Sequence[Hashable] | None,
) -> list[list[int]]:
    if candidate_groups is None:
        return [list(range(n_conditions))]
    if len(candidate_groups) != n_conditions:
        raise ValueError("candidate_groups must have one value per condition")
    grouped: defaultdict[Hashable, list[int]] = defaultdict(list)
    for index, group in enumerate(candidate_groups):
        grouped[group].append(index)
    return list(grouped.values())


def normalized_inverse_rank(
    truth: Array,
    prediction: Array,
    candidate_groups: Sequence[Hashable] | None = None,
) -> Array:
    """Miller et al. Normalized Inverse Rank using Euclidean distance.

    The released preprint code computes the fraction of unmatched observed
    centroids that are farther from a prediction than its matched centroid.
    Candidate sets are evaluated within covariate groups. Ties are losses,
    matching the strict comparison in the authors' implementation.
    """

    truth = _as_2d(truth, "truth")
    prediction = _as_2d(prediction, "prediction")
    _check_same_shape(truth=truth, prediction=prediction)
    scores = np.full(truth.shape[0], np.nan, dtype=float)
    for indices in _candidate_groups(truth.shape[0], candidate_groups):
        if len(indices) < 2:
            continue
        group_truth = truth[indices]
        group_prediction = prediction[indices]
        distances = cdist(group_prediction, group_truth, metric="euclidean")
        matched = np.diag(distances)
        wins = distances > matched[:, None]
        scores[indices] = wins.sum(axis=1) / (len(indices) - 1)
    return scores


def centroid_accuracy(
    truth: Array,
    prediction: Array,
    candidate_groups: Sequence[Hashable] | None = None,
) -> Array:
    """Systema centroid accuracy using Euclidean distance."""

    return normalized_inverse_rank(truth, prediction, candidate_groups)


def score_prediction_metrics(
    truth: Array,
    prediction: Array,
    control_reference: Array,
    perturbed_mean_reference: Array,
    deg_mask: Array,
    deg_weights: Array,
    candidate_groups: Sequence[Hashable] | None = None,
) -> dict[str, Array]:
    """Score the 13 Miller protocols plus Systema centroid accuracy.

    The 13 Miller protocols are MSE, DEG-weighted MSE, ten combinations of
    Pearson or R-squared with control or perturbed-mean references and
    all-gene/DEG/continuous-DE weighting, and Normalized Inverse Rank.

    The caller constructs the references and DE evidence. In the Miller et al.
    preprint code, continuous weights are the squared, min-max-normalized
    absolute ground-truth DE test statistics, normalized to sum to one.
    """

    arrays: Mapping[str, Array] = {
        "truth": _as_2d(truth, "truth"),
        "prediction": _as_2d(prediction, "prediction"),
        "control_reference": _as_2d(control_reference, "control_reference"),
        "perturbed_mean_reference": _as_2d(
            perturbed_mean_reference, "perturbed_mean_reference"
        ),
        "deg_mask": np.asarray(deg_mask, dtype=bool),
        "deg_weights": np.asarray(deg_weights, dtype=float),
    }
    _check_same_shape(**arrays)

    truth_control = arrays["truth"] - arrays["control_reference"]
    prediction_control = arrays["prediction"] - arrays["control_reference"]
    truth_perturbed = arrays["truth"] - arrays["perturbed_mean_reference"]
    prediction_perturbed = arrays["prediction"] - arrays["perturbed_mean_reference"]

    return {
        "mse": _rowwise_mse(arrays["truth"], arrays["prediction"]),
        "weighted_mse": _rowwise_mse(
            arrays["truth"], arrays["prediction"], arrays["deg_weights"]
        ),
        "pearson_delta_control": _rowwise_pearson(truth_control, prediction_control),
        "pearson_delta_control_degs": _rowwise_pearson(
            truth_control, prediction_control, arrays["deg_mask"]
        ),
        "pearson_delta_perturbed_mean": _rowwise_pearson(
            truth_perturbed, prediction_perturbed
        ),
        "pearson_delta_perturbed_mean_degs": _rowwise_pearson(
            truth_perturbed, prediction_perturbed, arrays["deg_mask"]
        ),
        "r2_delta_control": _rowwise_r2(truth_control, prediction_control),
        "r2_delta_control_degs": _rowwise_r2(
            truth_control, prediction_control, arrays["deg_mask"]
        ),
        "r2_delta_perturbed_mean": _rowwise_r2(truth_perturbed, prediction_perturbed),
        "r2_delta_perturbed_mean_degs": _rowwise_r2(
            truth_perturbed, prediction_perturbed, arrays["deg_mask"]
        ),
        "weighted_r2_delta_control": _rowwise_r2(
            truth_control, prediction_control, weights=arrays["deg_weights"]
        ),
        "weighted_r2_delta_perturbed_mean": _rowwise_r2(
            truth_perturbed,
            prediction_perturbed,
            weights=arrays["deg_weights"],
        ),
        "normalized_inverse_rank": normalized_inverse_rank(
            arrays["truth"], arrays["prediction"], candidate_groups
        ),
        "centroid_accuracy": centroid_accuracy(
            arrays["truth"], arrays["prediction"], candidate_groups
        ),
    }


def aggregate_metric_scores(scores: Mapping[str, Array]) -> dict[str, float]:
    """Macro-average finite condition-level scores without silent imputation."""

    aggregated = {}
    for metric_id, values in scores.items():
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        aggregated[metric_id] = float(finite.mean()) if finite.size else float("nan")
    return aggregated
