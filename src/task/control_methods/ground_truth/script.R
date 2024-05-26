requireNamespace("arrow", quietly = TRUE)
requireNamespace("anndata", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)

## VIASH START
par <- list(
  de_train_h5ad = "resources/neurips-2023-data/de_train.h5ad",
  de_test_h5ad = "resources/neurips-2023-data/de_test.h5ad",
  layer = "sign_log10_pval",
  id_map = "resources/neurips-2023-data/id_map.csv",
  output = "resources/neurips-2023-data/output_identity.parquet"
)
## VIASH END

# read data
de_test_h5ad <- anndata::read_h5ad(par$de_test_h5ad)
id_map <- read.csv(par$id_map)

# remove unneeded columns
output <- data.frame(
  id = as.integer(id_map$id),
  de_test_h5ad$layers[[par$layer]],
  stringsAsFactors = FALSE,
  check.names = FALSE
)

# store output
arrow::write_parquet(output, par$output)
