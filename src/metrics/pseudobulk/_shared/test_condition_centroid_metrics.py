"""Unit tests for the shared paper-derived metric formulas."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np


sys.path.insert(0, str(Path(__file__).parent))

from condition_centroid_metric_runner import run_condition_centroid_metric
from condition_centroid_metrics import centroid_accuracy, score_prediction_metrics


class PredictionMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
                [0.0, 2.0, 4.0, 6.0],
                [6.0, 4.0, 2.0, 0.0],
            ]
        )
        self.reference = np.zeros_like(self.truth)
        self.mask = np.ones_like(self.truth, dtype=bool)
        self.weights = np.full_like(self.truth, 0.25)
        self.groups = ["a", "a", "b", "b"]

    def test_perfect_prediction(self) -> None:
        scores = score_prediction_metrics(
            self.truth,
            self.truth,
            self.reference,
            self.reference,
            self.mask,
            self.weights,
            self.groups,
        )
        self.assertTrue(np.allclose(scores["mse"], 0))
        self.assertTrue(np.allclose(scores["weighted_mse"], 0))
        for metric_id, values in scores.items():
            if metric_id not in {"mse", "weighted_mse"}:
                self.assertTrue(np.allclose(values, 1), metric_id)

    def test_retrieval_uses_candidate_groups(self) -> None:
        prediction = self.truth.copy()
        prediction[[0, 1]] = prediction[[1, 0]]
        scores = centroid_accuracy(self.truth, prediction, self.groups)
        np.testing.assert_array_equal(scores, np.array([0.0, 0.0, 1.0, 1.0]))

    def test_retrieval_ties_are_failures(self) -> None:
        prediction = np.zeros((2, 2))
        truth = np.array([[1.0, 0.0], [-1.0, 0.0]])
        np.testing.assert_array_equal(
            centroid_accuracy(truth, prediction), np.array([0.0, 0.0])
        )

    def test_zero_weight_genes_do_not_contribute(self) -> None:
        truth = np.array([[0.0, 100.0]])
        prediction = np.array([[0.0, 0.0]])
        scores = score_prediction_metrics(
            truth,
            prediction,
            np.zeros_like(truth),
            np.zeros_like(truth),
            np.ones_like(truth, dtype=bool),
            np.array([[1.0, 0.0]]),
        )
        self.assertEqual(scores["weighted_mse"][0], 0.0)

    def test_de_mask_excludes_unselected_gene_errors(self) -> None:
        prediction = self.truth.copy()
        prediction[:, -1] += 100
        mask = self.mask.copy()
        mask[:, -1] = False
        scores = score_prediction_metrics(
            self.truth,
            prediction,
            self.reference,
            self.reference,
            mask,
            self.weights,
            self.groups,
        )
        self.assertTrue(np.all(scores["pearson_delta_control"] < 1))
        self.assertTrue(np.allclose(scores["pearson_delta_control_degs"], 1))
        self.assertTrue(np.all(scores["r2_delta_control"] < 1))
        self.assertTrue(np.allclose(scores["r2_delta_control_degs"], 1))

    def test_runner_writes_one_score_h5ad(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            prepared_path = temp_path / "prepared.npz"
            output_path = temp_path / "score.h5ad"
            np.savez_compressed(
                prepared_path,
                truth=self.truth,
                prediction=self.truth,
                control_reference=self.reference,
                perturbed_mean_reference=self.reference,
                deg_mask=self.mask,
                deg_weights=self.weights,
                candidate_groups=np.asarray(self.groups),
                condition_keys=np.asarray(["a1", "a2", "b1", "b2"]),
                metadata_json=np.asarray(
                    json.dumps(
                        {"dataset_id": "synthetic", "method_id": "perfect"}
                    )
                ),
            )
            run_condition_centroid_metric("mse", prepared_path, output_path)
            output = ad.read_h5ad(output_path)
            self.assertEqual(output.uns["dataset_id"], "synthetic")
            self.assertEqual(output.uns["method_id"], "perfect")
            self.assertEqual(output.uns["metric_ids"].tolist(), ["mse"])
            np.testing.assert_array_equal(output.uns["metric_values"], [0.0])


if __name__ == "__main__":
    unittest.main()
