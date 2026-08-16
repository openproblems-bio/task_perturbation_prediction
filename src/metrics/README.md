# Metric families

Metric components are grouped by the least-derived prediction representation
they require:

- `single_cell/` consumes predicted cells by genes with condition labels;
- `pseudobulk/` consumes one gene-expression centroid per condition; and
- `deg/` consumes one differential-expression result per condition.

An evaluator may derive pseudobulk centroids and DE results from a valid
single-cell prediction. It may also derive DE results from pseudobulk output
when the output preserves the replicate and control structure required by the
declared DE procedure. The reverse transformations are not valid.

Each metric component remains responsible for declaring its concrete input
contract through its Viash API merge. `metric_suite.yaml` records the current
components, scoring directions, sources, and protocol-specific references.

This directory contains direct prediction metrics only. Metric-evaluation
protocols belong outside these representation families.
