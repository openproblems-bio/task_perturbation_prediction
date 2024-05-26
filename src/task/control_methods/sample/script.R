requireNamespace("arrow", quietly = TRUE)
requireNamespace("anndata", quietly = TRUE)

## VIASH START
par <- list(
  de_train = "resources/neurips-2023-data/de_train.parquet",
  de_test = "resources/neurips-2023-data/de_test.parquet",
  id_map = "resources/neurips-2023-data/id_map.csv",
  output = "resources/neurips-2023-data/output_identity.parquet"
)
## VIASH END

# read data
de_train_h5ad <- anndata::read_h5ad(par$de_train_h5ad)
id_map <- arrow::read_csv_arrow(par$id_map)

# get gene names
gene_names <- de_train_h5ad$var_names

# create output data structure
output <- data.frame(id = id_map$id)

# generate random data
for (gene_name in gene_names) {
  output[[gene_name]] <- sample(de_train_h5ad$layers[[par$layer]][,gene_name], size = nrow(output), replace = TRUE)
}

# store output
arrow::write_parquet(output, par$output)
