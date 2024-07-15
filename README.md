# Perturbation Prediction


<!--
This file is automatically generated from the tasks's api/*.yaml files.
Do not edit this file directly.
-->

Predicting how small molecules change gene expression in different cell
types.

Path to source:
[`src`](https://github.com/openproblems-bio/task_perturbation_prediction/tree/main/src)

## README

## Installation

You need to have Docker, Java, and Viash installed. Follow [these
instructions](https://openproblems.bio/documentation/fundamentals/requirements)
to install the required dependencies.

## Add a method

To add a method to the repository, follow the instructions in the
`scripts/add_a_method.sh` script.

## Frequently used commands

To get started, you can run the following commands:

``` bash
git clone git@github.com:openproblems-bio/task_perturbation_prediction.git

cd task_perturbation_prediction

# download resources
scripts/download_resources.sh
```

To run the benchmark, you first need to build the components.
Afterwards, you can run the benchmark:

``` bash
viash ns build --parallel --setup cachedbuild

scripts/run_benchmark.sh
```

After adding a component, it is recommended to run the tests to ensure
that the component is working correctly:

``` bash
viash ns test --parallel
```

Optionally, you can provide the `--query` argument to test only a subset
of components:

``` bash
viash ns test --parallel --query "component_name"
```

## Motivation

Human biology can be complex, in part due to the function and interplay
of the body’s approximately 37 trillion cells, which are organized into
tissues, organs, and systems. However, recent advances in single-cell
technologies have provided unparalleled insight into the function of
cells and tissues at the level of DNA, RNA, and proteins. Yet leveraging
single-cell methods to develop medicines requires mapping causal links
between chemical perturbations and the downstream impact on cell state.
These experiments are costly and labor intensive, and not all cells and
tissues are amenable to high-throughput transcriptomic screening. If
data science could help accurately predict chemical perturbations in new
cell types, it could accelerate and expand the development of new
medicines.

Several methods have been developed for drug perturbation prediction,
most of which are variations on the autoencoder architecture (Dr.VAE,
scGEN, and ChemCPA). However, these methods lack proper benchmarking
datasets with diverse cell types to determine how well they generalize.
The largest available training dataset is the NIH-funded Connectivity
Map (CMap), which comprises over 1.3M small molecule perturbation
measurements. However, the CMap includes observations of only 978 genes,
less than 5% of all genes. Furthermore, the CMap data is comprised
almost entirely of measurements in cancer cell lines, which may not
accurately represent human biology.

## Description

This task aims to predict how small molecules change gene expression in
different cell types. This task was a [Kaggle
competition](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview)
as part of the [NeurIPS 2023 competition
track](https://neurips.cc/virtual/2023/competition/66586).

The task is to predict the gene expression profile of a cell after a
small molecule perturbation. For this competition, we designed and
generated a novel single-cell perturbational dataset in human peripheral
blood mononuclear cells (PBMCs). We selected 144 compounds from the
Library of Integrated Network-Based Cellular Signatures (LINCS)
Connectivity Map dataset ([PMID:
29195078](https://pubmed.ncbi.nlm.nih.gov/29195078/)) and measured
single-cell gene expression profiles after 24 hours of treatment. The
experiment was repeated in three healthy human donors, and the compounds
were selected based on diverse transcriptional signatures observed in
CD34+ hematopoietic stem cells (data not released). We performed this
experiment in human PBMCs because the cells are commercially available
with pre-obtained consent for public release and PBMCs are a primary,
disease-relevant tissue that contains multiple mature cell types
(including T-cells, B-cells, myeloid cells, and NK cells) with
established markers for annotation of cell types. To supplement this
dataset, we also measured cells from each donor at baseline with joint
scRNA and single-cell chromatin accessibility measurements using the 10x
Multiome assay. We hope that the addition of rich multi-omic data for
each donor and cell type at baseline will help establish biological
priors that explain the susceptibility of particular genes to exhibit
perturbation responses in difference biological contexts.

## Authors & contributors

| name              | roles       |
|:------------------|:------------|
| Artur Szałata     | author      |
| Robrecht Cannoodt | author      |
| Daniel Burkhardt  | author      |
| Malte D. Luecken  | author      |
| Tin M. Tunjic     | contributor |
| Mengbo Wang       | contributor |
| Andrew Benz       | author      |
| Tianyu Liu        | contributor |
| Jalil Nourisa     | contributor |
| Rico Meinl        | contributor |

## API

``` mermaid
flowchart LR
  file_sc_counts("Single Cell Counts")
  comp_process_dataset[/"Process dataset"/]
  file_de_train_h5ad("DE train")
  file_de_test_h5ad("DE test")
  file_id_map("ID Map")
  comp_control_method[/"Control Method"/]
  comp_method[/"Method"/]
  comp_metric[/"Metric"/]
  file_prediction("Prediction")
  file_model("Model")
  file_score("Score")
  file_sc_counts---comp_process_dataset
  comp_process_dataset-->file_de_train_h5ad
  comp_process_dataset-->file_de_test_h5ad
  comp_process_dataset-->file_id_map
  file_de_train_h5ad---comp_control_method
  file_de_train_h5ad---comp_method
  file_de_test_h5ad---comp_control_method
  file_de_test_h5ad---comp_metric
  file_id_map---comp_control_method
  file_id_map---comp_method
  comp_control_method-->file_prediction
  comp_method-->file_prediction
  comp_method-->file_model
  comp_metric-->file_score
  file_prediction---comp_metric
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

## Component type: Process dataset

Path:
[`src/process_dataset`](https://github.com/openproblems-bio/openproblems-v2/tree/main/src/process_dataset)

Process the raw dataset

Arguments:

<div class="small">

| Name              | Type   | Description                                                                                                         |
|:------------------|:-------|:--------------------------------------------------------------------------------------------------------------------|
| `--sc_counts`     | `file` | Anndata with the counts of the whole dataset.                                                                       |
| `--de_train_h5ad` | `file` | (*Output*) Differential expression results for training. Default: `de_train.h5ad`.                                  |
| `--de_test_h5ad`  | `file` | (*Output*) Differential expression results for testing. Default: `de_test.h5ad`.                                    |
| `--id_map`        | `file` | (*Output*) File indicates the order of de_test, the cell types and the small molecule names. Default: `id_map.csv`. |

</div>

## File format: DE train

Differential expression results for training.

Example file: `resources/datasets/neurips-2023-data/de_train.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'split', 'control'
     layers: 'logFC', 'AveExpr', 't', 'P.Value', 'adj.P.Value', 'B', 'is_de', 'is_de_adj', 'sign_log10_pval', 'clipped_sign_log10_pval'
     uns: 'dataset_id', 'dataset_name', 'dataset_url', 'dataset_reference', 'dataset_summary', 'dataset_description', 'dataset_organism', 'single_cell_obs'

</div>

Slot description:

<div class="small">

| Slot                                | Type        | Description                                                                                                                                                                                                                                                                                                       |
|:------------------------------------|:------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `obs["cell_type"]`                  | `string`    | The annotated cell type of each cell based on RNA expression.                                                                                                                                                                                                                                                     |
| `obs["sm_name"]`                    | `string`    | The primary name for the (parent) compound (in a standardized representation) as chosen by LINCS. This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                         |
| `obs["sm_lincs_id"]`                | `string`    | The global LINCS ID (parent) compound (in a standardized representation). This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                                                 |
| `obs["SMILES"]`                     | `string`    | Simplified molecular-input line-entry system (SMILES) representations of the compounds used in the experiment. This is a 1D representation of molecular structure. These SMILES are provided by Cellarity based on the specific compounds ordered for this experiment.                                            |
| `obs["split"]`                      | `string`    | Split. Must be one of ‘control’, ‘train’, ‘public_test’, or ‘private_test’.                                                                                                                                                                                                                                       |
| `obs["control"]`                    | `boolean`   | Boolean indicating whether this instance was used as a control.                                                                                                                                                                                                                                                   |
| `layers["logFC"]`                   | `double`    | Log fold change of the differential expression test.                                                                                                                                                                                                                                                              |
| `layers["AveExpr"]`                 | `double`    | (*Optional*) Average expression of the differential expression test.                                                                                                                                                                                                                                              |
| `layers["t"]`                       | `double`    | (*Optional*) T-statistic of the differential expression test.                                                                                                                                                                                                                                                     |
| `layers["P.Value"]`                 | `double`    | P-value of the differential expression test.                                                                                                                                                                                                                                                                      |
| `layers["adj.P.Value"]`             | `double`    | Adjusted P-value of the differential expression test.                                                                                                                                                                                                                                                             |
| `layers["B"]`                       | `double`    | (*Optional*) B-statistic of the differential expression test.                                                                                                                                                                                                                                                     |
| `layers["is_de"]`                   | `boolean`   | Whether the gene is differentially expressed.                                                                                                                                                                                                                                                                     |
| `layers["is_de_adj"]`               | `boolean`   | Whether the gene is differentially expressed after adjustment.                                                                                                                                                                                                                                                    |
| `layers["sign_log10_pval"]`         | `double`    | Differential expression value (`-log10(p-value) * sign(LFC)`) for each gene. Here, LFC is the estimated log-fold change in expression between the treatment and control condition after shrinkage as calculated by Limma. Positive LFC means the gene goes up in the treatment condition relative to the control. |
| `layers["clipped_sign_log10_pval"]` | `double`    | A clipped version of the sign_log10_pval layer. Values are clipped to be between -4 and 4 (i.e. `-log10(0.0001)` and `-log10(0.0001)`).                                                                                                                                                                           |
| `uns["dataset_id"]`                 | `string`    | A unique identifier for the dataset. This is different from the `obs.dataset_id` field, which is the identifier for the dataset from which the cell data is derived.                                                                                                                                              |
| `uns["dataset_name"]`               | `string`    | A human-readable name for the dataset.                                                                                                                                                                                                                                                                            |
| `uns["dataset_url"]`                | `string`    | (*Optional*) Link to the original source of the dataset.                                                                                                                                                                                                                                                          |
| `uns["dataset_reference"]`          | `string`    | (*Optional*) Bibtex reference of the paper in which the dataset was published.                                                                                                                                                                                                                                    |
| `uns["dataset_summary"]`            | `string`    | Short description of the dataset.                                                                                                                                                                                                                                                                                 |
| `uns["dataset_description"]`        | `string`    | Long description of the dataset.                                                                                                                                                                                                                                                                                  |
| `uns["dataset_organism"]`           | `string`    | (*Optional*) The organism of the sample in the dataset.                                                                                                                                                                                                                                                           |
| `uns["single_cell_obs"]`            | `dataframe` | A dataframe with the cell-level metadata for the training set.                                                                                                                                                                                                                                                    |

</div>

## File format: DE test

Differential expression results for testing.

Example file: `resources/datasets/neurips-2023-data/de_test.h5ad`

Format:

<div class="small">

    AnnData object
     obs: 'cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'split', 'control'
     layers: 'logFC', 'AveExpr', 't', 'P.Value', 'adj.P.Value', 'B', 'is_de', 'is_de_adj', 'sign_log10_pval', 'clipped_sign_log10_pval'
     uns: 'dataset_id', 'dataset_name', 'dataset_url', 'dataset_reference', 'dataset_summary', 'dataset_description', 'dataset_organism', 'single_cell_obs'

</div>

Slot description:

<div class="small">

| Slot                                | Type        | Description                                                                                                                                                                                                                                                                                                       |
|:------------------------------------|:------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `obs["cell_type"]`                  | `string`    | The annotated cell type of each cell based on RNA expression.                                                                                                                                                                                                                                                     |
| `obs["sm_name"]`                    | `string`    | The primary name for the (parent) compound (in a standardized representation) as chosen by LINCS. This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                         |
| `obs["sm_lincs_id"]`                | `string`    | The global LINCS ID (parent) compound (in a standardized representation). This is provided to map the data in this experiment to the LINCS Connectivity Map data.                                                                                                                                                 |
| `obs["SMILES"]`                     | `string`    | Simplified molecular-input line-entry system (SMILES) representations of the compounds used in the experiment. This is a 1D representation of molecular structure. These SMILES are provided by Cellarity based on the specific compounds ordered for this experiment.                                            |
| `obs["split"]`                      | `string`    | Split. Must be one of ‘control’, ‘train’, ‘public_test’, or ‘private_test’.                                                                                                                                                                                                                                       |
| `obs["control"]`                    | `boolean`   | Boolean indicating whether this instance was used as a control.                                                                                                                                                                                                                                                   |
| `layers["logFC"]`                   | `double`    | Log fold change of the differential expression test.                                                                                                                                                                                                                                                              |
| `layers["AveExpr"]`                 | `double`    | (*Optional*) Average expression of the differential expression test.                                                                                                                                                                                                                                              |
| `layers["t"]`                       | `double`    | (*Optional*) T-statistic of the differential expression test.                                                                                                                                                                                                                                                     |
| `layers["P.Value"]`                 | `double`    | P-value of the differential expression test.                                                                                                                                                                                                                                                                      |
| `layers["adj.P.Value"]`             | `double`    | Adjusted P-value of the differential expression test.                                                                                                                                                                                                                                                             |
| `layers["B"]`                       | `double`    | (*Optional*) B-statistic of the differential expression test.                                                                                                                                                                                                                                                     |
| `layers["is_de"]`                   | `boolean`   | Whether the gene is differentially expressed.                                                                                                                                                                                                                                                                     |
| `layers["is_de_adj"]`               | `boolean`   | Whether the gene is differentially expressed after adjustment.                                                                                                                                                                                                                                                    |
| `layers["sign_log10_pval"]`         | `double`    | Differential expression value (`-log10(p-value) * sign(LFC)`) for each gene. Here, LFC is the estimated log-fold change in expression between the treatment and control condition after shrinkage as calculated by Limma. Positive LFC means the gene goes up in the treatment condition relative to the control. |
| `layers["clipped_sign_log10_pval"]` | `double`    | A clipped version of the sign_log10_pval layer. Values are clipped to be between -4 and 4 (i.e. `-log10(0.0001)` and `-log10(0.0001)`).                                                                                                                                                                           |
| `uns["dataset_id"]`                 | `string`    | A unique identifier for the dataset. This is different from the `obs.dataset_id` field, which is the identifier for the dataset from which the cell data is derived.                                                                                                                                              |
| `uns["dataset_name"]`               | `string`    | A human-readable name for the dataset.                                                                                                                                                                                                                                                                            |
| `uns["dataset_url"]`                | `string`    | (*Optional*) Link to the original source of the dataset.                                                                                                                                                                                                                                                          |
| `uns["dataset_reference"]`          | `string`    | (*Optional*) Bibtex reference of the paper in which the dataset was published.                                                                                                                                                                                                                                    |
| `uns["dataset_summary"]`            | `string`    | Short description of the dataset.                                                                                                                                                                                                                                                                                 |
| `uns["dataset_description"]`        | `string`    | Long description of the dataset.                                                                                                                                                                                                                                                                                  |
| `uns["dataset_organism"]`           | `string`    | (*Optional*) The organism of the sample in the dataset.                                                                                                                                                                                                                                                           |
| `uns["single_cell_obs"]`            | `dataframe` | A dataframe with the cell-level metadata.                                                                                                                                                                                                                                                                         |

</div>

## File format: ID Map

File indicates the order of de_test, the cell types and the small
molecule names.

Example file: `resources/datasets/neurips-2023-data/id_map.csv`

Format:

<div class="small">

    Tabular data
     'id', 'cell_type', 'sm_name'

</div>

Slot description:

<div class="small">

| Column      | Type      | Description                    |
|:------------|:----------|:-------------------------------|
| `id`        | `integer` | Index of the test observation. |
| `cell_type` | `string`  | Cell type name.                |
| `sm_name`   | `string`  | Small molecule name.           |

</div>

## Component type: Control Method

Path:
[`src/control_methods`](https://github.com/openproblems-bio/openproblems-v2/tree/main/src/control_methods)

A control method.

Arguments:

<div class="small">

| Name              | Type     | Description                                                                         |
|:------------------|:---------|:------------------------------------------------------------------------------------|
| `--de_train_h5ad` | `file`   | (*Optional*) Differential expression results for training.                          |
| `--de_test_h5ad`  | `file`   | Differential expression results for testing.                                        |
| `--id_map`        | `file`   | File indicates the order of de_test, the cell types and the small molecule names.   |
| `--layer`         | `string` | (*Optional*) Which layer to use for prediction. Default: `clipped_sign_log10_pval`. |
| `--output`        | `file`   | (*Output*) Differential Gene Expression prediction.                                 |

</div>

## Component type: Method

Path:
[`src/methods`](https://github.com/openproblems-bio/openproblems-v2/tree/main/src/methods)

A perturbation prediction method

Arguments:

<div class="small">

| Name              | Type     | Description                                                                                                         |
|:------------------|:---------|:--------------------------------------------------------------------------------------------------------------------|
| `--de_train_h5ad` | `file`   | (*Optional*) Differential expression results for training.                                                          |
| `--id_map`        | `file`   | File indicates the order of de_test, the cell types and the small molecule names.                                   |
| `--layer`         | `string` | (*Optional*) Which layer to use for prediction. Default: `clipped_sign_log10_pval`.                                 |
| `--output`        | `file`   | (*Output*) Differential Gene Expression prediction.                                                                 |
| `--output_model`  | `file`   | (*Optional, Output*) Optional model output. If no value is passed, the model will be removed at the end of the run. |

</div>

## Component type: Metric

Path:
[`src/metrics`](https://github.com/openproblems-bio/openproblems-v2/tree/main/src/metrics)

A perturbation prediction metric

Arguments:

<div class="small">

| Name                 | Type     | Description                                                                                   |
|:---------------------|:---------|:----------------------------------------------------------------------------------------------|
| `--de_test_h5ad`     | `file`   | Differential expression results for testing.                                                  |
| `--de_test_layer`    | `string` | (*Optional*) In which layer to find the DE data. Default: `clipped_sign_log10_pval`.          |
| `--prediction`       | `file`   | Differential Gene Expression prediction.                                                      |
| `--prediction_layer` | `string` | (*Optional*) In which layer to find the predicted DE data. Default: `prediction`.             |
| `--output`           | `file`   | (*Output*) File indicating the score of a metric.                                             |
| `--resolve_genes`    | `string` | (*Optional*) How to resolve difference in genes between the two datasets. Default: `de_test`. |
| `--resolve_genes`    | `string` | (*Optional*) How to resolve difference in genes between the two datasets. Default: `de_test`. |

</div>

## File format: Prediction

Differential Gene Expression prediction

Example file: `resources/datasets/neurips-2023-data/prediction.h5ad`

Format:

<div class="small">

    AnnData object
     layers: 'prediction'
     uns: 'dataset_id', 'method_id'

</div>

Slot description:

<div class="small">

| Slot                   | Type     | Description                                                                                                                                                          |
|:-----------------------|:---------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `layers["prediction"]` | `double` | Predicted differential gene expression.                                                                                                                              |
| `uns["dataset_id"]`    | `string` | A unique identifier for the dataset. This is different from the `obs.dataset_id` field, which is the identifier for the dataset from which the cell data is derived. |
| `uns["method_id"]`     | `string` | A unique identifier for the method used to generate the prediction.                                                                                                  |

</div>

## File format: Model

Optional model output. If no value is passed, the model will be removed
at the end of the run.

Example file: `resources/datasets/neurips-2023-data/model/`

Format:

<div class="small">

</div>

Slot description:

<div class="small">

</div>

## File format: Score

File indicating the score of a metric.

Example file: `resources/datasets/neurips-2023-data/score.h5ad`

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

