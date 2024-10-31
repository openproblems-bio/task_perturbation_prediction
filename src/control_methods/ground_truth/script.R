requireNamespace("anndata", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)

## VIASH START
par <- list(
  de_train = "resources/datasets/neurips-2023-data/de_train.h5ad",
  de_test = "resources/datasets/neurips-2023-data/de_test.h5ad",
  layer = "clipped_sign_log10_pval",
  id_map = "resources/datasets/neurips-2023-data/id_map.csv",
  output = "resources/datasets/neurips-2023-data/output_identity.h5ad"
)
## VIASH END

# read data
de_test <- anndata::read_h5ad(par$de_test)

# remove unneeded columns
output <- anndata::AnnData(
  layers = list(
    prediction = de_test$layers[[par$layer]]
  ),
  obs = de_test$obs[, c()],
  var = de_test$var[, c()],
  uns = list(
    dataset_id = de_test$uns$dataset_id,
    method_id = meta$name
  )
)

# write output
output$write_h5ad(par$output, compression = "gzip")