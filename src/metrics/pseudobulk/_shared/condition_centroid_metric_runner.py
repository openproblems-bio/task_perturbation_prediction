"""I/O adapter for one condition-centroid metric component."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np

from condition_centroid_metrics import METRIC_DIRECTIONS, score_prediction_metrics


def run_condition_centroid_metric(
    metric_id: str,
    prepared_path: str | Path,
    output_path: str | Path,
) -> None:
    """Score one metric from an aligned condition-centroid NPZ bundle."""

    if metric_id not in METRIC_DIRECTIONS:
        raise ValueError(f"Unknown condition-centroid metric: {metric_id}")

    prepared_path = Path(prepared_path)
    output_path = Path(output_path)
    with np.load(prepared_path, allow_pickle=False) as prepared:
        scores = score_prediction_metrics(
            truth=prepared["truth"],
            prediction=prepared["prediction"],
            control_reference=prepared["control_reference"],
            perturbed_mean_reference=prepared["perturbed_mean_reference"],
            deg_mask=prepared["deg_mask"].astype(bool),
            deg_weights=prepared["deg_weights"],
            candidate_groups=prepared["candidate_groups"].tolist(),
        )[metric_id]
        metadata = json.loads(str(prepared["metadata_json"].item()))
        condition_keys = [str(value) for value in prepared["condition_keys"].tolist()]

    if len(condition_keys) != len(scores):
        raise ValueError(
            "condition_keys must contain one identifier per condition score, got "
            f"{len(condition_keys)} identifiers and {len(scores)} scores"
        )

    finite = scores[np.isfinite(scores)]
    aggregate = float(finite.mean()) if finite.size else float("nan")
    per_condition = {
        condition: None if not np.isfinite(value) else float(value)
        for condition, value in zip(condition_keys, scores)
    }

    output = ad.AnnData(X=np.empty((0, 0), dtype=np.float32))
    output.uns["dataset_id"] = metadata["dataset_id"]
    output.uns["method_id"] = metadata["method_id"]
    output.uns["metric_ids"] = np.asarray([metric_id], dtype=str)
    output.uns["metric_values"] = np.asarray([aggregate], dtype=float)
    output.uns["per_condition_json"] = json.dumps(per_condition, sort_keys=True)
    output.uns["adapter_metadata_json"] = json.dumps(metadata, sort_keys=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(output_path, compression="gzip")

    print(f"{metric_id}: {aggregate:.8g}")
    print(f"Wrote score file to {output_path}")
