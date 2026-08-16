# Paper-derived condition-centroid metrics

This branch adds 14 direct metrics for evaluating perturbation-model
predictions. It does not add meta-metrics or change the existing Open Problems
metrics. Each new component lives under `src/metrics/pseudobulk/` and
uses the score-file interface declared in
`src/api/comp_condition_centroid_metric.yaml`.

The implementations follow the associated paper codebases:

- 13 CellSimBench protocols from Miller et al.; and
- Systema centroid accuracy, retained under its paper name even though its
  released formula is the same as normalized inverse rank when evaluated over
  the same candidate pool.

## Input contract

The metrics operate on aligned condition-by-gene matrices stored in the
evaluator-prepared NPZ bundle declared by
`src/api/file_condition_centroid_bundle.yaml`:

- `truth`: observed condition centroids;
- `prediction`: predicted condition centroids;
- `control_reference`: matched control centroid for each condition;
- `perturbed_mean_reference`: equal-condition-weighted mean of training
  perturbation centroids in the relevant covariate group;
- `deg_mask`: evaluator-computed significant-DE-gene mask;
- `deg_weights`: evaluator-computed continuous DE weights;
- `candidate_groups`: groups that define valid retrieval candidates;
- `condition_keys`: stable condition identifiers; and
- `metadata_json`: dataset and method provenance.

The prediction, truth, reference, mask, and weight arrays must have identical
shape. The metric runner macro-averages finite per-condition scores and writes
the aggregate to `uns["metric_values"]`. It also records per-condition scores
and adapter provenance.

## Metrics

Let `Y` be the observed centroid, `P` the predicted centroid, `C` the matched
control, and `M` the training-perturbation mean. The DE mask and weights are
computed from held-out observations, not supplied by a model.

| Metric | What it tests | Calculation | Direction |
|---|---|---|---:|
| `mse` | Absolute centroid accuracy | Mean squared error between `Y` and `P` over all genes | Minimize |
| `weighted_mse` | Accuracy on genes with strong DE evidence | Squared error weighted by continuous DE weights | Minimize |
| `pearson_delta_control` | Pattern of the total response relative to control | Pearson correlation between `Y-C` and `P-C` over all genes | Maximize |
| `pearson_delta_control_degs` | Control-relative response pattern on DE genes | Same Pearson correlation restricted by `deg_mask` | Maximize |
| `pearson_delta_perturbed_mean` | Perturbation-specific pattern beyond the shared perturbed state | Pearson correlation between `Y-M` and `P-M` | Maximize |
| `pearson_delta_perturbed_mean_degs` | Perturbation-specific pattern on DE genes | Same Pearson correlation restricted by `deg_mask` | Maximize |
| `r2_delta_control` | Calibrated control-relative effect | R-squared between `Y-C` and `P-C` over all genes | Maximize |
| `r2_delta_control_degs` | Calibrated control-relative effect on DE genes | Same R-squared restricted by `deg_mask` | Maximize |
| `r2_delta_perturbed_mean` | Calibrated perturbation-specific effect | R-squared between `Y-M` and `P-M` | Maximize |
| `r2_delta_perturbed_mean_degs` | Calibrated perturbation-specific effect on DE genes | Same R-squared restricted by `deg_mask` | Maximize |
| `weighted_r2_delta_control` | Calibrated control-relative effect emphasizing DE evidence | DE-weighted R-squared between `Y-C` and `P-C` | Maximize |
| `weighted_r2_delta_perturbed_mean` | Calibrated perturbation-specific effect emphasizing DE evidence | DE-weighted R-squared between `Y-M` and `P-M` | Maximize |
| `normalized_inverse_rank` | Whether a prediction retrieves its matched observed perturbation | Fraction of unmatched truths farther away than the match | Maximize |
| `centroid_accuracy` | Systema's centroid-retrieval accuracy | Same released retrieval formula under the same candidate pool | Maximize |

### DE weighting

The continuous weights match the Miller codebase: take the absolute
ground-truth DE test statistic, min-max normalize it within condition, square
it, and normalize the resulting weights to sum to one. These weights are not
`1 - adjusted p-value`.

### Correlation and R-squared

Pearson correlation measures gene-wise pattern agreement and is insensitive to
some scale and offset errors. R-squared also penalizes those calibration
errors and can be negative. DEG-restricted scores are undefined when fewer
than three eligible finite genes remain; the aggregate uses the remaining
finite condition scores.

Although four IDs contain `_degs`, their primary prediction input is still a
condition centroid. The DEG mask only selects the gene space.

### Retrieval

For each prediction, retrieval metrics compute Euclidean distances to observed
centroids within its candidate group. The score is the fraction of unmatched
observations whose distance is strictly greater than the matched distance.
Ties therefore count as failures. Candidate grouping is part of the metric
protocol and must be reported with a result.

## SciPlex3 smoke adapter

`scripts/create_resources/prepare_sciplex3_metric_bundle.py` provides an
executable smoke test using the supplied SciPlex3 H5AD file. It uses `rep1`
condition means as predictions and `rep2` condition means as held-out truth,
then prepares the references and DE evidence required by the metrics.

The adapter reads selected rows directly from the on-disk CSR count matrix,
log-normalizes each cell, and writes an ignored NPZ artifact:

```bash
python3 scripts/create_resources/prepare_sciplex3_metric_bundle.py

viash run src/metrics/pseudobulk/mse/config.vsh.yaml -- \
  --prepared resources/datasets/srivatsan_2020_sciplex3/train_mean_smoke_input.npz \
  --output resources/datasets/srivatsan_2020_sciplex3/train_mean_smoke_mse.h5ad
```

This is a code smoke test, not a model benchmark result. The dataset and
generated artifacts remain ignored under `resources/`.

## Tests

The shared tests cover perfect predictions, DEG masking, DE weighting,
candidate grouping, strict retrieval ties, and the H5AD score output:

```bash
python3 -m unittest discover \
  -s src/metrics/pseudobulk/_shared \
  -p 'test_*.py' -v
```

Each Viash component also registers the standard configuration check and the
shared synthetic integration test under `test_resources`.

## Sources

- [Miller et al. preprint](https://www.biorxiv.org/content/10.1101/2025.10.20.683304v1)
- [Miller et al. evaluation code](https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines/tree/edf89745269d)
- [Systema paper](https://www.nature.com/articles/s41587-025-02777-8)
- [Systema evaluation code](https://github.com/mlbio-epfl/systema/tree/aaf5b5353993b48b78543f2f93b3e18ca65df515/evaluation)
- [scPertEval preprint](https://www.biorxiv.org/content/10.64898/2026.07.23.740433v1)
- [Perturbation Metrics preprint](https://www.biorxiv.org/content/10.1101/2023.12.26.572833v1)
