"""Shared Viash integration test for every condition-centroid component."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np


## VIASH START
meta = {"executable": "", "name": "mse"}
## VIASH END


def main() -> None:
    metric_id = meta["name"]
    truth = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 4.0, 6.0],
            [6.0, 4.0, 2.0, 0.0],
        ]
    )
    zeros = np.zeros_like(truth)
    with tempfile.TemporaryDirectory(prefix=f"{metric_id}-") as temp_dir:
        temp_path = Path(temp_dir)
        prepared_path = temp_path / "prepared.npz"
        output_path = temp_path / "score.h5ad"
        np.savez_compressed(
            prepared_path,
            truth=truth,
            prediction=truth,
            control_reference=zeros,
            perturbed_mean_reference=zeros,
            deg_mask=np.ones_like(truth, dtype=bool),
            deg_weights=np.full_like(truth, 0.25),
            candidate_groups=np.asarray(["a", "a", "b", "b"]),
            condition_keys=np.asarray(["a1", "a2", "b1", "b2"]),
            metadata_json=np.asarray(
                json.dumps(
                    {"dataset_id": "synthetic", "method_id": "perfect_prediction"}
                )
            ),
        )
        subprocess.run(
            [
                meta["executable"],
                "--prepared",
                str(prepared_path),
                "--output",
                str(output_path),
            ],
            check=True,
        )

        score = ad.read_h5ad(output_path)
        assert score.shape == (0, 0)
        assert score.uns["dataset_id"] == "synthetic"
        assert score.uns["method_id"] == "perfect_prediction"
        assert score.uns["metric_ids"].tolist() == [metric_id]
        expected = 0.0 if metric_id in {"mse", "weighted_mse"} else 1.0
        assert np.isclose(score.uns["metric_values"][0], expected)


if __name__ == "__main__":
    main()
