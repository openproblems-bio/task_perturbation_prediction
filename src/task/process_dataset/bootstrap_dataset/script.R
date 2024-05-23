requireNamespace("anndata", quietly = TRUE)
requireNamespace("arrow", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)
library(tidyr, warn.conflicts = FALSE)
library(purrr, warn.conflicts = FALSE)

## VIASH START
par <- list(
  input_parquet = "resources/neurips-2023-kaggle/de_train.parquet",
  input_h5ad = "resources/neurips-2023-kaggle/de_train.h5ad",
  output_parquet = "output/de_train_filtered_*.parquet",
  output_h5ad = "output/de_train_filtered_*.h5ad",
  bootstrap_num_replicates = NULL,
  bootstrap_sample_fraction = NULL
)
## VIASH END

# Load data
input_parquet <- arrow::read_parquet(par$input_parquet)
input_h5ad <- anndata::read_h5ad(par$input_h5ad)

# fill in defaults
bootstrap_num_replicates <- par$bootstrap_num_replicates %||% 1
bootstrap_sample_fraction <- par$bootstrap_sample_fraction %||% 1

for (i in seq_len(bootstrap_num_replicates)) {
  cat("Generating replicate", i, "\n", sep = "")
  ix <- sample.int(
    nrow(input_parquet),
    round(nrow(input_parquet) * bootstrap_sample_fraction, 0),
    replace = FALSE
  )
  output_parquet <- input_parquet[ix, ]
  output_h5ad <- input_h5ad[ix, ]

  output_parquet_path <- gsub("\\*", i, par$output_parquet)
  output_h5ad_path <- gsub("\\*", i, par$output_h5ad)

  zzz <- output$write_h5ad(output_h5ad_path, compression = "gzip")
  arrow::write_parquet(output_parquet, output_parquet_path)
}
