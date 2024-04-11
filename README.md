# DGE Perturbation Prediction


DGE Perturbation Prediction

Path:
[`_viash_par/task_dir_1/dge_perturbation_prediction`](https://github.com/openproblems-bio/task-dge-perturbation-prediction/tree/main/_viash_par/task_dir_1/dge_perturbation_prediction)

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
  file_de_per_plate_by_celltype("DE per plate by cell type")
  comp_process_dataset[/"Data processor"/]
  file_de_train("DE train")
  file_de_test("DE test")
  comp_method[/"Method"/]
  comp_metric[/"Metric"/]
  file_prediction("Prediction")
  file_score("Score")
  file_de_per_plate("DE per plate")
  file_de_per_plate_by_celltype---comp_process_dataset
  comp_process_dataset-->file_de_train
  comp_process_dataset-->file_de_test
  file_de_train---comp_method
  file_de_test---comp_metric
  comp_method-->file_prediction
  comp_metric-->file_score
  file_prediction---comp_metric
  file_de_per_plate---comp_process_dataset
```

## File format: DE per plate by cell type

Differential expression results per plate and cell type.

Example file:
`resources/neurips-2023-raw/de_per_plate_by_cell_type.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'dose_uM', 'plate_name', 'cell_type', 'n_de_genes', 'n_de_genes_adj', 'split', 'donor_id', 'sm_name'
     layers: '-log10(p-value)sign(lfc)', 'P.Value', 'adj.P.Val', 'is_de', 'is_de_adj', 'lfc', 'logFC', 'masked_lfc', 'masked_sign(lfc)'

</div>

Slot description:

<div class="small">

| Slot                                 | Type      | Description                                                                 |
|:-------------------------------------|:----------|:----------------------------------------------------------------------------|
| `obs["dose_uM"]`                     | `integer` | Dose in uM.                                                                 |
| `obs["plate_name"]`                  | `string`  | Name of the plate.                                                          |
| `obs["cell_type"]`                   | `string`  | Cell type.                                                                  |
| `obs["n_de_genes"]`                  | `integer` | Number of differentially expressed genes.                                   |
| `obs["n_de_genes_adj"]`              | `integer` | Number of differentially expressed genes after multiple testing correction. |
| `obs["split"]`                       | `string`  | Split of the data.                                                          |
| `obs["donor_id"]`                    | `string`  | Donor ID.                                                                   |
| `obs["sm_name"]`                     | `string`  | Name of the small molecule.                                                 |
| `layers["-log10(p-value)sign(lfc)"]` | `double`  | Log10 of the p-value multiplied by the sign of the log fold change.         |
| `layers["P.Value"]`                  | `double`  | P-value.                                                                    |
| `layers["adj.P.Val"]`                | `double`  | Adjusted p-value.                                                           |
| `layers["is_de"]`                    | `boolean` | Is differentially expressed.                                                |
| `layers["is_de_adj"]`                | `boolean` | Is differentially expressed after multiple testing correction.              |
| `layers["lfc"]`                      | `double`  | Log fold change.                                                            |
| `layers["logFC"]`                    | `double`  | Log fold change.                                                            |
| `layers["masked_lfc"]`               | `double`  | Masked log fold change.                                                     |
| `layers["masked_sign(lfc)"]`         | `double`  | Masked sign of the log fold change.                                         |

</div>

## Component type: Data processor

Path:
[`src/dge_perturbation_prediction`](https://github.com/openproblems-bio/openproblems-v2/tree/main/src/dge_perturbation_prediction)

A DGE regression dataset processor

Arguments:

<div class="small">

| Name                         | Type   | Description                                              |
|:-----------------------------|:-------|:---------------------------------------------------------|
| `--de_per_plate_by_celltype` | `file` | Differential expression results per plate and cell type. |
| `--de_per_plate_by_celltype` | `file` | Differential expression results per plate and cell type. |
| `--de_train`                 | `file` | (*Output*) Differential expression results for training. |
| `--de_test`                  | `file` | (*Output*) Differential expression results for testing.  |

</div>

## File format: DE train

Differential expression results for training.

Example file: `resources/datasets/de_train.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control'
     layers: '-log10(p-value)sign(lfc)'

</div>

Slot description:

<div class="small">

| Slot                                 | Type      | Description                                                                                                                                                                                                                                                                                                     |
|:-------------------------------------|:----------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `obs["cell_type"]`                   | `string`  | The annotated cell type of each cell based on RNA expression.                                                                                                                                                                                                                                                   |
| `obs["sm_name"]`                     | `string`  | The primary name for the (parent) compound (in a standardized representation) as chosen by LINCS. This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                       |
| `obs["sm_lincs_id"]`                 | `string`  | The global LINCS ID (parent) compound (in a standardized representation). This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                                               |
| `obs["SMILES"]`                      | `string`  | Simplified molecular-input line-entry system (SMILES) representations of the compounds used in the experiment. This is a 1D representation of molecular structure. These SMILES are provided by Cellarity based on the specific compounds ordered for this experiment.                                          |
| `obs["control"]`                     | `boolean` | Boolean indicating whether this instance was used as a control.                                                                                                                                                                                                                                                 |
| `layers["-log10(p-value)sign(lfc)"]` | `double`  | ifferential expression value (-log10(p-value) \* sign(LFC)) for each gene. Here, LFC is the estimated log-fold change in expression between the treatment and control condition after shrinkage as calculated by Limma. Positive LFC means the gene goes up in the treatment condition relative to the control. |

</div>

## File format: DE test

Differential expression results for testing.

Example file: `resources/datasets/de_test.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control'
     layers: '-log10(p-value)sign(lfc)'

</div>

Slot description:

<div class="small">

| Slot                                 | Type      | Description                                                                                                                                                                                                                                                                                                     |
|:-------------------------------------|:----------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `obs["cell_type"]`                   | `string`  | The annotated cell type of each cell based on RNA expression.                                                                                                                                                                                                                                                   |
| `obs["sm_name"]`                     | `string`  | The primary name for the (parent) compound (in a standardized representation) as chosen by LINCS. This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                       |
| `obs["sm_lincs_id"]`                 | `string`  | The global LINCS ID (parent) compound (in a standardized representation). This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                                               |
| `obs["SMILES"]`                      | `string`  | Simplified molecular-input line-entry system (SMILES) representations of the compounds used in the experiment. This is a 1D representation of molecular structure. These SMILES are provided by Cellarity based on the specific compounds ordered for this experiment.                                          |
| `obs["control"]`                     | `boolean` | Boolean indicating whether this instance was used as a control.                                                                                                                                                                                                                                                 |
| `layers["-log10(p-value)sign(lfc)"]` | `double`  | ifferential expression value (-log10(p-value) \* sign(LFC)) for each gene. Here, LFC is the estimated log-fold change in expression between the treatment and control condition after shrinkage as calculated by Limma. Positive LFC means the gene goes up in the treatment condition relative to the control. |

</div>

## Component type: Method

Path:
[`src/dge_perturbation_prediction/methods`](https://github.com/openproblems-bio/openproblems-v2/tree/main/src/dge_perturbation_prediction/methods)

A regression method.

Arguments:

<div class="small">

| Name       | Type   | Description                                         |
|:-----------|:-------|:----------------------------------------------------|
| `--input`  | `file` | Differential expression results for training.       |
| `--output` | `file` | (*Output*) Differential Gene Expression prediction. |

</div>

## Component type: Metric

Path:
[`src/dge_perturbation_prediction/metrics`](https://github.com/openproblems-bio/openproblems-v2/tree/main/src/dge_perturbation_prediction/metrics)

A metric to compare two predictions.

Arguments:

<div class="small">

| Name           | Type   | Description                                  |
|:---------------|:-------|:---------------------------------------------|
| `--de_test`    | `file` | Differential expression results for testing. |
| `--prediction` | `file` | Differential Gene Expression prediction.     |
| `--output`     | `file` | (*Output*) Metric score file.                |

</div>

## File format: Prediction

Differential Gene Expression prediction

Example file: `resources/processed/prediction.h5ad`

Format:

<div class="small">

    AnnData object
     layers: 'prediction'

</div>

Slot description:

<div class="small">

| Slot                   | Type     | Description                                    |
|:-----------------------|:---------|:-----------------------------------------------|
| `layers["prediction"]` | `double` | Predicted differential gene expression values. |

</div>

## File format: Score

Metric score file

Example file: `...h5ad`

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

## File format: DE per plate

Differential expression results per plate and cell type.

Example file: `resources/neurips-2023-raw/de_per_plate.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'dose_uM', 'plate_name', 'cell_type', 'n_de_genes', 'n_de_genes_adj', 'sm_name'
     layers: '-log10(p-value)sign(lfc)', 'P.Value', 'adj.P.Val', 'is_de', 'is_de_adj', 'lfc', 'logFC', 'masked_lfc', 'masked_sign(lfc)'

</div>

Slot description:

<div class="small">

| Slot                                 | Type      | Description                                                                 |
|:-------------------------------------|:----------|:----------------------------------------------------------------------------|
| `obs["dose_uM"]`                     | `integer` | Dose in uM.                                                                 |
| `obs["plate_name"]`                  | `string`  | Name of the plate.                                                          |
| `obs["cell_type"]`                   | `string`  | Cell type.                                                                  |
| `obs["n_de_genes"]`                  | `integer` | Number of differentially expressed genes.                                   |
| `obs["n_de_genes_adj"]`              | `integer` | Number of differentially expressed genes after multiple testing correction. |
| `obs["sm_name"]`                     | `string`  | Name of the small molecule.                                                 |
| `layers["-log10(p-value)sign(lfc)"]` | `double`  | Log10 of the p-value multiplied by the sign of the log fold change.         |
| `layers["P.Value"]`                  | `double`  | P-value.                                                                    |
| `layers["adj.P.Val"]`                | `double`  | Adjusted p-value.                                                           |
| `layers["is_de"]`                    | `boolean` | Is differentially expressed.                                                |
| `layers["is_de_adj"]`                | `boolean` | Is differentially expressed after multiple testing correction.              |
| `layers["lfc"]`                      | `double`  | Log fold change.                                                            |
| `layers["logFC"]`                    | `double`  | Log fold change.                                                            |
| `layers["masked_lfc"]`               | `double`  | Masked log fold change.                                                     |
| `layers["masked_sign(lfc)"]`         | `double`  | Masked sign of the log fold change.                                         |

</div>

