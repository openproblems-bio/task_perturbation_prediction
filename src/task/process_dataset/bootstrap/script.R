requireNamespace("anndata", quietly = TRUE)
requireNamespace("arrow", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)
library(tidyr, warn.conflicts = FALSE)
library(purrr, warn.conflicts = FALSE)

## VIASH START
par <- list(
  train_parquet = "resources/neurips-2023-kaggle/de_train.parquet",
  train_h5ad = "resources/neurips-2023-kaggle/de_train.h5ad",
  test_parquet = "resources/neurips-2023-kaggle/de_test.parquet",
  test_h5ad = "resources/neurips-2023-kaggle/de_test.h5ad",
  output_train_parquet = "output/de_train_filtered_*.parquet",
  output_train_h5ad = "output/de_train_filtered_*.h5ad",
  output_test_parquet = "output/de_test_filtered_*.parquet",
  output_test_h5ad = "output/de_test_filtered_*.h5ad",
  bootstrap_num_replicates = 10,
  bootstrap_obs_fraction = .95,
  bootstrap_var_fraction = .95
)
## VIASH END

# Load data
train_parquet <- arrow::read_parquet(par$train_parquet)
train_h5ad <- anndata::read_h5ad(par$train_h5ad)
test_parquet <- arrow::read_parquet(par$test_parquet)
test_h5ad <- anndata::read_h5ad(par$test_h5ad)

for (i in seq_len(par$bootstrap_num_replicates)) {
  cat("Generating replicate", i, "\n", sep = "")

  # sample indices
  obs_ix <- sample.int(
    nrow(train_h5ad),
    round(nrow(train_parquet) * par$bootstrap_obs_fraction, 0),
    replace = FALSE
  )
  var_ix <- sample.int(
    ncol(train_h5ad),
    round(ncol(train_parquet) * par$bootstrap_var_fraction, 0),
    replace = FALSE
  )

  # subset h5ad
  output_train_h5ad <- train_h5ad[obs_ix, var_ix]
  output_test_h5ad <- test_h5ad[, var_ix]

  # subset parquet
  train_parquet_cols <- c(
    intersect(colnames(train_parquet), colnames(train_h5ad$obs)),
    colnames(output_train_h5ad)
  )
  test_parquet_cols <- c(
    intersect(colnames(test_parquet), colnames(test_h5ad$obs)),
    colnames(output_test_h5ad)
  )
  output_train_parquet <- train_parquet[obs_ix, train_parquet_cols]
  output_test_parquet <- test_parquet[, test_parquet_cols]

  # write output
  output_train_parquet_path <- gsub("\\*", i, par$output_train_parquet)
  output_train_h5ad_path <- gsub("\\*", i, par$output_train_h5ad)
  output_test_parquet_path <- gsub("\\*", i, par$output_test_parquet)
  output_test_h5ad_path <- gsub("\\*", i, par$output_test_h5ad)

  zzz <- output_train_h5ad$write_h5ad(output_train_h5ad_path, compression = "gzip")
  arrow::write_parquet(output_train_parquet, output_train_parquet_path)
  zzz <- output_test_h5ad$write_h5ad(output_test_h5ad_path, compression = "gzip")
  arrow::write_parquet(output_test_parquet, output_test_parquet_path)
}
