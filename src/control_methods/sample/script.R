requireNamespace("anndata", quietly = TRUE)

## VIASH START
par <- list(
  de_train = "resources/datasets/neurips-2023-data/de_train.h5ad",
  de_test = "resources/datasets/neurips-2023-data/de_test.h5ad",
  layer = "clipped_sign_log10_pval",
  id_map = "resources/datasets/neurips-2023-data/id_map.csv",
  output = "resources/datasets/neurips-2023-data/output_identity.h5ad"
)
meta <- list(
  name = "sample"
)
## VIASH END

# read data
de_train <- anndata::read_h5ad(par$de_train)
id_map <- read.csv(par$id_map)

# get gene names
gene_names <- de_train$var_names

input_layer <- de_train$layers[[par$layer]]

prediction <- sapply(gene_names, function(gene_name) {
  sample(input_layer[,gene_name], size = nrow(id_map), replace = TRUE)
})
rownames(prediction) <- id_map$id


# remove unneeded columns
output <- anndata::AnnData(
  layers = list(prediction = prediction),
  var = de_train$var[, c()],
  shape = c(nrow(id_map), length(gene_names)),
  uns = list(
    dataset_id = de_train$uns$dataset_id,
    method_id = meta$name
  )
)

# write output
output$write_h5ad(par$output, compression = "gzip")
