requireNamespace("anndata", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)
library(tidyr, warn.conflicts = FALSE)
library(purrr, warn.conflicts = FALSE)

## VIASH START
par <- list(
  input = "resources/neurips-2023-raw/sc_counts_reannotated_with_counts.h5ad",
  output = "output/sc_counts_bootstrapped_*.h5ad",
  num_replicates = 10,
  obs_fraction = .95,
  var_fraction = 1
)
## VIASH END

# Load data
input <- anndata::read_h5ad(par$input)

for (i in seq_len(par$num_replicates)) {
  cat("Generating replicate", i, "\n", sep = "")

  # sample indices
  obs_ix <- sample.int(
    nrow(input),
    round(nrow(input) * par$obs_fraction, 0),
    replace = FALSE
  )
  var_ix <- sample.int(
    ncol(input),
    round(ncol(input) * par$var_fraction, 0),
    replace = FALSE
  )

  # subset h5ad
  output <- input[obs_ix, var_ix]

  original_dataset_id <- output$uns[["dataset_id"]]
  dataset_id <- paste0(original_dataset_id, "-bootstrap", i)
  output$uns[["dataset_id"]] <- dataset_id
  output$uns[["original_dataset_id"]] <- original_dataset_id

  # write output
  output_path <- gsub("\\*", i, par$output)

  zzz <- output$write_h5ad(output_path, compression = "gzip")
}
