"""Prepare a SciPlex3 replicate-mean prediction metric smoke test.

The source H5AD stores a 647,840 x 18,413 CSR count matrix. This adapter reads
only the selected CSR rows instead of materializing the full matrix in memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats


CONDITION_COLUMNS = ("cell_type", "sm_name", "dose_uM", "timepoint_hr")
TRAIN_REPLICATE = "rep1"
TRUTH_REPLICATE = "rep2"


def _decode_strings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind not in {"O", "S", "U"}:
        return values
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else value for value in values],
        dtype=object,
    )


def _read_h5ad_array(node: h5py.Dataset | h5py.Group):
    """Read the H5AD array encodings needed by the SciPlex metadata."""

    encoding = node.attrs.get("encoding-type", "array")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if isinstance(node, h5py.Dataset):
        return _decode_strings(node[:])
    if encoding == "categorical":
        categories = _read_h5ad_array(node["categories"])
        return pd.Categorical.from_codes(
            node["codes"][:],
            categories=categories,
            ordered=bool(node.attrs.get("ordered", False)),
        )
    if encoding == "nullable-string-array":
        values = pd.array(_decode_strings(node["values"][:]), dtype="string")
        values[node["mask"][:].astype(bool)] = pd.NA
        return values
    raise ValueError(f"Unsupported H5AD array encoding: {encoding}")


def _read_frame(
    path: Path,
    key: str,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Read selected dataframe columns across old and new AnnData encodings."""

    with h5py.File(path, "r") as handle:
        group = handle[key]
        index_key = group.attrs["_index"]
        if isinstance(index_key, bytes):
            index_key = index_key.decode("utf-8")
        if columns is None:
            columns = tuple(_decode_strings(group.attrs["column-order"]).tolist())
        frame = pd.DataFrame(
            {column: _read_h5ad_array(group[column]) for column in columns},
            index=pd.Index(_read_h5ad_array(group[index_key]), name=index_key),
        )
    return frame


def _read_csr_rows(path: Path, row_ids: np.ndarray) -> sp.csr_matrix:
    """Read selected rows from ``layers/counts`` without loading all 777M entries."""

    row_ids = np.asarray(row_ids, dtype=np.int64)
    if row_ids.ndim != 1 or np.any(np.diff(row_ids) <= 0):
        raise ValueError("row_ids must be unique and strictly increasing")

    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    output_indptr = np.zeros(len(row_ids) + 1, dtype=np.int64)
    with h5py.File(path, "r") as handle:
        counts = handle["layers/counts"]
        source_indptr = counts["indptr"][:]
        n_genes = int(counts.attrs["shape"][1])
        for output_row, source_row in enumerate(row_ids):
            start = int(source_indptr[source_row])
            stop = int(source_indptr[source_row + 1])
            data_parts.append(counts["data"][start:stop])
            index_parts.append(counts["indices"][start:stop])
            output_indptr[output_row + 1] = output_indptr[output_row] + stop - start

    data = np.concatenate(data_parts) if data_parts else np.array([], dtype=np.int32)
    indices = (
        np.concatenate(index_parts) if index_parts else np.array([], dtype=np.int32)
    )
    return sp.csr_matrix(
        (data, indices, output_indptr),
        shape=(len(row_ids), n_genes),
    )


def _log_normalize(counts: sp.csr_matrix, target_sum: float = 10_000) -> sp.csr_matrix:
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("Selected cells include an empty count profile")
    normalized = sp.diags(target_sum / totals) @ counts.astype(np.float64)
    normalized.data = np.log1p(normalized.data)
    return normalized.tocsr()


def _mean_and_variance(values: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, int]:
    n = values.shape[0]
    mean = np.asarray(values.mean(axis=0)).ravel()
    mean_square = np.asarray(values.multiply(values).mean(axis=0)).ravel()
    variance = np.maximum(mean_square - mean * mean, 0)
    if n > 1:
        variance *= n / (n - 1)
    return mean, variance, n


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full(pvalues.shape, np.nan)
    finite = np.where(np.isfinite(pvalues))[0]
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(pvalues[finite])]
    ranked = pvalues[order] * finite.size / np.arange(1, finite.size + 1)
    adjusted[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return adjusted


def _overestimated_t_test(
    target: sp.csr_matrix,
    reference: sp.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Match Scanpy's ``t-test_overestim_var`` used by the Miller codebase."""

    target_mean, target_var, target_n = _mean_and_variance(target)
    reference_mean, reference_var, _ = _mean_and_variance(reference)
    reference_n = target_n
    standard_error_squared = target_var / target_n + reference_var / reference_n
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = (target_mean - reference_mean) / np.sqrt(standard_error_squared)
        degrees_freedom = standard_error_squared**2 / (
            (target_var / target_n) ** 2 / max(target_n - 1, 1)
            + (reference_var / reference_n) ** 2 / max(reference_n - 1, 1)
        )
    statistic = np.nan_to_num(statistic, nan=0, posinf=0, neginf=0)
    degrees_freedom = np.where(
        np.isfinite(degrees_freedom) & (degrees_freedom > 0),
        degrees_freedom,
        1,
    )
    pvalues = np.nan_to_num(
        2 * stats.t.sf(np.abs(statistic), degrees_freedom),
        nan=1,
    )
    return statistic, _bh_adjust(pvalues)


def _mejia_weights(statistic: np.ndarray) -> np.ndarray:
    """Squared min-max absolute DE statistic weights from the Miller codebase."""

    magnitude = np.abs(np.asarray(statistic, dtype=np.float64))
    finite = np.isfinite(magnitude)
    if not finite.any():
        return np.zeros_like(magnitude)
    low, high = magnitude[finite].min(), magnitude[finite].max()
    scaled = (
        (magnitude - low) / (high - low)
        if high > low
        else np.zeros_like(magnitude)
    )
    weights = np.square(np.nan_to_num(scaled, nan=0))
    total = weights.sum()
    return weights / total if total > 0 else np.full(weights.size, 1 / weights.size)


def _condition_tuple(row: pd.Series) -> tuple:
    return tuple(row[column] for column in CONDITION_COLUMNS)


def _eligible_conditions(obs: pd.DataFrame, min_cells: int) -> pd.DataFrame:
    treated = obs.loc[~obs["control"].astype(bool)]
    counts = (
        treated.groupby(["source_replicate", *CONDITION_COLUMNS], observed=True)
        .size()
        .unstack("source_replicate", fill_value=0)
    )
    for replicate in (TRAIN_REPLICATE, TRUTH_REPLICATE):
        if replicate not in counts:
            counts[replicate] = 0
    eligible = counts.loc[
        (counts[TRAIN_REPLICATE] >= min_cells)
        & (counts[TRUTH_REPLICATE] >= min_cells)
    ].reset_index()
    eligible["support"] = eligible[[TRAIN_REPLICATE, TRUTH_REPLICATE]].min(axis=1)
    return eligible


def _balanced_conditions(eligible: pd.DataFrame, maximum: int) -> list[tuple]:
    by_cell_type = {
        cell_type: group.sort_values(
            ["support", "sm_name", "dose_uM", "timepoint_hr"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
        for cell_type, group in eligible.groupby("cell_type", observed=True)
    }
    selected: list[tuple] = []
    rank = 0
    while len(selected) < maximum:
        added = False
        for cell_type in sorted(by_cell_type):
            group = by_cell_type[cell_type]
            if rank < len(group):
                selected.append(_condition_tuple(group.iloc[rank]))
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        rank += 1
    return selected


def _sample_indices(
    indices: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) > maximum:
        indices = rng.choice(indices, maximum, replace=False)
    return np.sort(indices)


def prepare(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    obs = _read_frame(
        input_path,
        "obs",
        columns=(*CONDITION_COLUMNS, "source_replicate", "control"),
    )
    var = _read_frame(input_path, "var", columns=())
    obs = obs.reset_index(drop=True)

    eligible = _eligible_conditions(obs, args.min_cells)
    conditions = _balanced_conditions(eligible, args.max_conditions)
    if len(conditions) < 2:
        raise ValueError("Fewer than two conditions have enough cells in both replicates")

    rng = np.random.default_rng(args.seed)
    train_rows: dict[tuple, np.ndarray] = {}
    truth_rows: dict[tuple, np.ndarray] = {}
    for condition in conditions:
        condition_mask = np.ones(len(obs), dtype=bool)
        for column, value in zip(CONDITION_COLUMNS, condition):
            condition_mask &= obs[column].to_numpy() == value
        train_rows[condition] = _sample_indices(
            np.where(condition_mask & (obs["source_replicate"] == TRAIN_REPLICATE))[0],
            args.max_cells,
            rng,
        )
        truth_rows[condition] = _sample_indices(
            np.where(condition_mask & (obs["source_replicate"] == TRUTH_REPLICATE))[0],
            args.max_cells,
            rng,
        )

    selected_cell_types = sorted({condition[0] for condition in conditions})
    control_rows: dict[tuple[str, str], np.ndarray] = {}
    for replicate in (TRAIN_REPLICATE, TRUTH_REPLICATE):
        for cell_type in selected_cell_types:
            mask = (
                obs["control"].astype(bool).to_numpy()
                & (obs["source_replicate"] == replicate).to_numpy()
                & (obs["cell_type"] == cell_type).to_numpy()
            )
            control_rows[(replicate, cell_type)] = _sample_indices(
                np.where(mask)[0], args.max_cells, rng
            )

    all_rows = np.unique(
        np.concatenate(
            [*train_rows.values(), *truth_rows.values(), *control_rows.values()]
        )
    )
    print(f"Reading {len(all_rows):,} selected cells from {input_path}")
    expression = _log_normalize(_read_csr_rows(input_path, all_rows))
    row_lookup = {source: index for index, source in enumerate(all_rows)}

    def matrix(rows: np.ndarray) -> sp.csr_matrix:
        return expression[[row_lookup[int(row)] for row in rows]]

    train_matrices = {key: matrix(rows) for key, rows in train_rows.items()}
    truth_matrices = {key: matrix(rows) for key, rows in truth_rows.items()}
    control_matrices = {key: matrix(rows) for key, rows in control_rows.items()}

    prediction = np.vstack(
        [np.asarray(train_matrices[key].mean(axis=0)).ravel() for key in conditions]
    )
    truth = np.vstack(
        [np.asarray(truth_matrices[key].mean(axis=0)).ravel() for key in conditions]
    )

    train_condition_means: dict[str, np.ndarray] = {}
    for cell_type in selected_cell_types:
        cell_type_rows = [
            prediction[index]
            for index, condition in enumerate(conditions)
            if condition[0] == cell_type
        ]
        train_condition_means[cell_type] = np.mean(cell_type_rows, axis=0)

    control_reference = np.vstack(
        [
            np.asarray(
                control_matrices[(TRAIN_REPLICATE, condition[0])].mean(axis=0)
            ).ravel()
            for condition in conditions
        ]
    )
    perturbed_mean_reference = np.vstack(
        [train_condition_means[condition[0]] for condition in conditions]
    )

    deg_mask_rows: list[np.ndarray] = []
    deg_weight_rows: list[np.ndarray] = []
    for condition in conditions:
        cell_type = condition[0]
        rest = sp.vstack(
            [
                truth_matrices[other]
                for other in conditions
                if other != condition and other[0] == cell_type
            ]
            + [control_matrices[(TRUTH_REPLICATE, cell_type)]],
            format="csr",
        )
        statistic, adjusted_pvalue = _overestimated_t_test(
            truth_matrices[condition], rest
        )
        deg_mask_rows.append(adjusted_pvalue < 0.05)
        deg_weight_rows.append(_mejia_weights(statistic))

    condition_json = [
        json.dumps(
            {
                column: value.item() if isinstance(value, np.generic) else value
                for column, value in zip(CONDITION_COLUMNS, condition)
            },
            sort_keys=True,
        )
        for condition in conditions
    ]
    metadata = {
        "dataset_id": "srivatsan_2020_sciplex3",
        "method_id": "train_replicate_means",
        "input_path": str(input_path),
        "normalization": "log1p(counts / library_size * 10000)",
        "train_replicate": TRAIN_REPLICATE,
        "truth_replicate": TRUTH_REPLICATE,
        "max_cells_per_condition": args.max_cells,
        "seed": args.seed,
        "prediction_object": "condition centroid",
        "de_method": "t-test_overestim_var against rest of held-out cell-type cells",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        truth=truth,
        prediction=prediction,
        control_reference=control_reference,
        perturbed_mean_reference=perturbed_mean_reference,
        deg_mask=np.vstack(deg_mask_rows),
        deg_weights=np.vstack(deg_weight_rows),
        candidate_groups=np.asarray([condition[0] for condition in conditions], dtype=str),
        condition_keys=np.asarray(condition_json, dtype=str),
        gene_names=np.asarray(var.index.astype(str), dtype=str),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(
        f"Wrote {len(conditions)} conditions x {truth.shape[1]:,} genes to {output_path}"
    )


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = repo_root / "resources/datasets/srivatsan_2020_sciplex3"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=dataset_dir / "srivatsan_2020_sciplex3_compressed.h5ad",
        type=Path,
    )
    parser.add_argument(
        "--output", default=dataset_dir / "train_mean_smoke_input.npz", type=Path
    )
    parser.add_argument("--max-conditions", type=int, default=18)
    parser.add_argument("--min-cells", type=int, default=32)
    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(_parse_args())
