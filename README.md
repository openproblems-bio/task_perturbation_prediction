# task-dge-perturbation-prediction

This repository contains the code for the task of predicting the perturbation effects.

## Install

You need to have Docker, Java, and Viash installed. Please follow [these instructions](https://openproblems.bio/documentation/fundamentals/requirements) to install all of the required dependencies.

## First steps

### 1. Clone this repository

```bash
git clone git@github.com:openproblems-bio/task-dge-perturbation-prediction.git
```

### 2. Sync resources

```bash
scripts/1_sync_resources.sh
```

### 3. Build the Docker images

```bash
scripts/3_build_components.sh
```

### 4. Process the raw data

```bash
scripts/4_process_dataset.sh
```

### 5. Run the baseline

```bash
scripts/5_run_rf_method.sh
```

### 6. Evaluate the baseline

```bash
scripts/6_run_rf_metric.sh
```
