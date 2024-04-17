# DGE Perturbation Prediction


DGE Perturbation Prediction

Path:
[`src/task`](https://github.com/openproblems-bio/task-dge-perturbation-prediction/tree/main/src/task)

## Motivation

TODO: fill in

## Description

TODO: fill in

## Authors & contributors

| name              | roles  |
|:------------------|:-------|
| Artur Szałata     | author |
| Robrecht Cannoodt | author |

## API

``` mermaid
flowchart LR
  file_sc_counts("Single Cell Counts")
  comp_process_dataset[/"Data processor"/]
  file_de_train("DE train")
  file_de_test("DE test")
  file_id_map("ID Map")
  comp_control_method[/"Control Method"/]
  comp_method[/"Method"/]
  comp_metric[/"Metric"/]
  file_prediction("Prediction")
  file_score("Score")
  file_lincs_id_compound_mapping("Mapping compound names to lincs ids and smiles")
  file_sc_counts---comp_process_dataset
  comp_process_dataset-->file_de_train
  comp_process_dataset-->file_de_test
  comp_process_dataset-->file_id_map
  file_de_train---comp_control_method
  file_de_train---comp_method
  file_de_test---comp_control_method
  file_de_test---comp_metric
  file_id_map---comp_control_method
  file_id_map---comp_method
  comp_control_method-->file_prediction
  comp_method-->file_prediction
  comp_metric-->file_score
  file_prediction---comp_metric
  file_lincs_id_compound_mapping---comp_process_dataset
```

## File format: Single Cell Counts

Anndata with the counts of the whole dataset.

Example file: `resources/neurips-2023-raw/sc_counts.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'dose_uM', 'timepoint_hr', 'raw_cell_id', 'hashtag_id', 'well', 'container_format', 'row', 'col', 'plate_name', 'cell_id', 'cell_type', 'split', 'donor_id', 'sm_name'
     obsm: 'HTO_clr', 'X_pca', 'X_umap', 'protein_counts'
     layers: 'counts'

</div>

Slot description:

<div class="small">

| Slot                      | Type      | Description                                    |
|:--------------------------|:----------|:-----------------------------------------------|
| `obs["dose_uM"]`          | `integer` | Dose in micromolar.                            |
| `obs["timepoint_hr"]`     | `float`   | Time point measured in hours.                  |
| `obs["raw_cell_id"]`      | `string`  | Original cell identifier.                      |
| `obs["hashtag_id"]`       | `string`  | Identifier for hashtag oligo.                  |
| `obs["well"]`             | `string`  | Well location in the plate.                    |
| `obs["container_format"]` | `string`  | Format of the container (e.g., 96-well plate). |
| `obs["row"]`              | `string`  | Row in the plate.                              |
| `obs["col"]`              | `integer` | Column in the plate.                           |
| `obs["plate_name"]`       | `string`  | Name of the plate.                             |
| `obs["cell_id"]`          | `string`  | Unique cell identifier.                        |
| `obs["cell_type"]`        | `string`  | Type of cell (e.g., B cells, T cells CD4+).    |
| `obs["split"]`            | `string`  | Dataset split type (e.g., control, treated).   |
| `obs["donor_id"]`         | `string`  | Identifier for the donor.                      |
| `obs["sm_name"]`          | `string`  | Name of the small molecule used for treatment. |
| `obsm["HTO_clr"]`         | `matrix`  | Corrected counts for hashing tags.             |
| `obsm["X_pca"]`           | `matrix`  | Principal component analysis results.          |
| `obsm["X_umap"]`          | `matrix`  | UMAP dimensionality reduction results.         |
| `obsm["protein_counts"]`  | `matrix`  | Count data for proteins.                       |
| `layers["counts"]`        | `matrix`  | Raw count data for each gene across cells.     |

</div>

## Component type: Data processor

Path:
[`src/dge_perturbation_prediction`](https://github.com/openproblems-bio/openproblems-v2/tree/main/src/dge_perturbation_prediction)

A DGE regression dataset processor

Arguments:

<div class="small">

| Name                          | Type   | Description                                                                                  |
|:------------------------------|:-------|:---------------------------------------------------------------------------------------------|
| `--sc_counts`                 | `file` | Anndata with the counts of the whole dataset.                                                |
| `--lincs_id_compound_mapping` | `file` | Parquet file mapping compound names to lincs ids and smiles.                                 |
| `--de_train`                  | `file` | (*Output*) Differential expression results for training.                                     |
| `--de_test`                   | `file` | (*Output*) Differential expression results for testing.                                      |
| `--id_map`                    | `file` | (*Output*) File indicates the order of de_test, the cell types and the small molecule names. |

</div>

## File format: DE train

Differential expression results for training.

Example file: `resources/neurips-2023-data/de_train.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'split', 'control'
     layers: 'P.Value', 'adj.P.Value', 'is_de', 'is_de_adj', 'logFC', 'sign_log10_pval'

</div>

Slot description:

<div class="small">

| Slot                        | Type      | Description                                                                                                                                                                                                                                                                                                      |
|:----------------------------|:----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `obs["cell_type"]`          | `string`  | The annotated cell type of each cell based on RNA expression.                                                                                                                                                                                                                                                    |
| `obs["sm_name"]`            | `string`  | The primary name for the (parent) compound (in a standardized representation) as chosen by LINCS. This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                        |
| `obs["sm_lincs_id"]`        | `string`  | The global LINCS ID (parent) compound (in a standardized representation). This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                                                |
| `obs["SMILES"]`             | `string`  | Simplified molecular-input line-entry system (SMILES) representations of the compounds used in the experiment. This is a 1D representation of molecular structure. These SMILES are provided by Cellarity based on the specific compounds ordered for this experiment.                                           |
| `obs["split"]`              | `string`  | Split. Must be one of ‘control’, ‘train’, ‘public_test’, or ‘private_test’.                                                                                                                                                                                                                                      |
| `obs["control"]`            | `boolean` | Boolean indicating whether this instance was used as a control.                                                                                                                                                                                                                                                  |
| `layers["P.Value"]`         | `double`  | P-value of the differential expression test.                                                                                                                                                                                                                                                                     |
| `layers["adj.P.Value"]`     | `double`  | Adjusted P-value of the differential expression test.                                                                                                                                                                                                                                                            |
| `layers["is_de"]`           | `boolean` | Whether the gene is differentially expressed.                                                                                                                                                                                                                                                                    |
| `layers["is_de_adj"]`       | `boolean` | Whether the gene is differentially expressed after adjustment.                                                                                                                                                                                                                                                   |
| `layers["logFC"]`           | `double`  | Log fold change of the differential expression test.                                                                                                                                                                                                                                                             |
| `layers["sign_log10_pval"]` | `double`  | Differential expression value (-log10(p-value) \* sign(LFC)) for each gene. Here, LFC is the estimated log-fold change in expression between the treatment and control condition after shrinkage as calculated by Limma. Positive LFC means the gene goes up in the treatment condition relative to the control. |

</div>

## File format: DE test

Differential expression results for testing.

Example file: `resources/neurips-2023-data/de_test.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'id', 'cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'split', 'control'
     layers: 'P.Value', 'adj.P.Value', 'is_de', 'is_de_adj', 'logFC', 'sign_log10_pval'

</div>

Slot description:

<div class="small">

| Slot                        | Type      | Description                                                                                                                                                                                                                                                                                                      |
|:----------------------------|:----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `obs["id"]`                 | `integer` | Index of the test observation.                                                                                                                                                                                                                                                                                   |
| `obs["cell_type"]`          | `string`  | The annotated cell type of each cell based on RNA expression.                                                                                                                                                                                                                                                    |
| `obs["sm_name"]`            | `string`  | The primary name for the (parent) compound (in a standardized representation) as chosen by LINCS. This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                        |
| `obs["sm_lincs_id"]`        | `string`  | The global LINCS ID (parent) compound (in a standardized representation). This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                                                |
| `obs["SMILES"]`             | `string`  | Simplified molecular-input line-entry system (SMILES) representations of the compounds used in the experiment. This is a 1D representation of molecular structure. These SMILES are provided by Cellarity based on the specific compounds ordered for this experiment.                                           |
| `obs["split"]`              | `string`  | Split. Must be one of ‘control’, ‘train’, ‘public_test’, or ‘private_test’.                                                                                                                                                                                                                                      |
| `obs["control"]`            | `boolean` | Boolean indicating whether this instance was used as a control.                                                                                                                                                                                                                                                  |
| `layers["P.Value"]`         | `double`  | P-value of the differential expression test.                                                                                                                                                                                                                                                                     |
| `layers["adj.P.Value"]`     | `double`  | Adjusted P-value of the differential expression test.                                                                                                                                                                                                                                                            |
| `layers["is_de"]`           | `boolean` | Whether the gene is differentially expressed.                                                                                                                                                                                                                                                                    |
| `layers["is_de_adj"]`       | `boolean` | Whether the gene is differentially expressed after adjustment.                                                                                                                                                                                                                                                   |
| `layers["logFC"]`           | `double`  | Log fold change of the differential expression test.                                                                                                                                                                                                                                                             |
| `layers["sign_log10_pval"]` | `double`  | Differential expression value (-log10(p-value) \* sign(LFC)) for each gene. Here, LFC is the estimated log-fold change in expression between the treatment and control condition after shrinkage as calculated by Limma. Positive LFC means the gene goes up in the treatment condition relative to the control. |

</div>

## File format: ID Map

File indicates the order of de_test, the cell types and the small
molecule names.

Example file: `resources/neurips-2023-data/id_map.csv`

Format:

<div class="small">

    AnnData object
     obs: 'id', 'cell_type', 'sm_name'

</div>

Slot description:

<div class="small">

| Slot               | Type      | Description                    |
|:-------------------|:----------|:-------------------------------|
| `obs["id"]`        | `integer` | Index of the test observation. |
| `obs["cell_type"]` | `string`  | Cell type name.                |
| `obs["sm_name"]`   | `string`  | Small molecule name.           |

</div>

## Component type: Control Method

Path:
[`src/dge_perturbation_prediction/control_methods`](https://github.com/openproblems-bio/openproblems-v2/tree/main/src/dge_perturbation_prediction/control_methods)

A control method.

Arguments:

<div class="small">

| Name         | Type   | Description                                                                       |
|:-------------|:-------|:----------------------------------------------------------------------------------|
| `--de_train` | `file` | Differential expression results for training.                                     |
| `--de_test`  | `file` | Differential expression results for testing.                                      |
| `--id_map`   | `file` | File indicates the order of de_test, the cell types and the small molecule names. |
| `--output`   | `file` | (*Output*) Differential Gene Expression prediction.                               |

</div>

## Component type: Method

Path:
[`src/dge_perturbation_prediction/methods`](https://github.com/openproblems-bio/openproblems-v2/tree/main/src/dge_perturbation_prediction/methods)

A regression method.

Arguments:

<div class="small">

| Name         | Type   | Description                                                                       |
|:-------------|:-------|:----------------------------------------------------------------------------------|
| `--de_train` | `file` | Differential expression results for training.                                     |
| `--id_map`   | `file` | File indicates the order of de_test, the cell types and the small molecule names. |
| `--output`   | `file` | (*Output*) Differential Gene Expression prediction.                               |

</div>

## Component type: Metric

Path:
[`src/dge_perturbation_prediction/metrics`](https://github.com/openproblems-bio/openproblems-v2/tree/main/src/dge_perturbation_prediction/metrics)

A metric to compare two predictions.

Arguments:

<div class="small">

| Name           | Type   | Description                                       |
|:---------------|:-------|:--------------------------------------------------|
| `--de_test`    | `file` | Differential expression results for testing.      |
| `--prediction` | `file` | Differential Gene Expression prediction.          |
| `--output`     | `file` | (*Output*) File indicating the score of a metric. |

</div>

## File format: Prediction

Differential Gene Expression prediction

Example file: `resources/neurips-2023-data/output_rf.parquet`

Format:

<div class="small">

    AnnData object
     obs: 'id'
     layers: 'sign_log10_pval'

</div>

Slot description:

<div class="small">

| Slot                        | Type      | Description                                                          |
|:----------------------------|:----------|:---------------------------------------------------------------------|
| `obs["id"]`                 | `integer` | Index of the test observation.                                       |
| `layers["sign_log10_pval"]` | `double`  | Predicted sign of the logFC times the log10 of the adjusted p-value. |

</div>

## File format: Score

File indicating the score of a metric.

Example file: `resources/neurips-2023-data/score_rf.json`

Format:

<div class="small">

    AnnData object
     uns: 'dataset_id', 'method_id', 'metric_ids', 'metric_values'

</div>

Slot description:

<div class="small">

| Slot                   | Type     | Description                                                                                  |
|:-----------------------|:---------|:---------------------------------------------------------------------------------------------|
| `uns["dataset_id"]`    | `string` | A unique identifier for the dataset.                                                         |
| `uns["method_id"]`     | `string` | A unique identifier for the method.                                                          |
| `uns["metric_ids"]`    | `string` | One or more unique metric identifiers.                                                       |
| `uns["metric_values"]` | `double` | The metric values obtained for the given prediction. Must be of same length as ‘metric_ids’. |

</div>

## File format: Mapping compound names to lincs ids and smiles

Parquet file mapping compound names to lincs ids and smiles.

Example file:
`resources/neurips-2023-raw/lincs_id_compound_mapping.parquet`

Format:

<div class="small">

    AnnData object
     obs: 'compound_id', 'sm_lincs_id', 'sm_name', 'smiles'

</div>

Slot description:

<div class="small">

| Slot                 | Type     | Description                                           |
|:---------------------|:---------|:------------------------------------------------------|
| `obs["compound_id"]` | `string` | Unique identifier for the compound.                   |
| `obs["sm_lincs_id"]` | `string` | LINCS identifier for the compound.                    |
| `obs["sm_name"]`     | `string` | Name of the compound.                                 |
| `obs["smiles"]`      | `string` | SMILES notation representing the molecular structure. |

</div>

