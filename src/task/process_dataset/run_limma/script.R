requireNamespace("anndata", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)
library(tidyr, warn.conflicts = FALSE)
library(purrr, warn.conflicts = FALSE)
library(tibble, warn.conflicts = FALSE)
library(edgeR)

## VIASH START
par <- list(
  input = "resources/neurips-2023-data/small_pseudobulk.h5ad",
  de_sig_cutoff = 0.05,
  control_compound = "Dimethyl Sulfoxide",
  # for public data
  output = "resources/neurips-2023-data/de_train_qlf.h5ad",
  input_splits = c("train", "control", "public_test"),
  output_splits = c("train", "control", "public_test")
  # # for private data
  # output = "resources/neurips-2023-data/de_test.h5ad",
  # input_splits = c("train", "control", "public_test", "private_test"),
  # output_splits = c("private_test")
)
meta <- list(
  cpus = 30L
)
## VIASH END

# load data
adata <- anndata::read_h5ad(par$input)

# select [cell_type, sm_name] pairs which will be used for DE analysis
new_obs <- adata$obs %>%
  select(cell_type, sm_name, sm_lincs_id, SMILES, split, control) %>%
  distinct() %>%
  filter(sm_name != par$control_compound)

if (!is.null(par$output_splits)) {
  new_obs <- new_obs %>%
    filter(split %in% par$output_splits)
}

# check which cell_types to run limma for
cell_types <- as.character(unique(new_obs$cell_type))

# helper function for transforming values to limma compatible values
limma_trafo <- function(value) {
  gsub("[^[:alnum:]]", "_", value)
}

# run glmQLFit per cell type
ql_fits <- list()

for (cell_type in cell_types) {
  cat("Running glmQLFit for cell type: ", cell_type, "\n")

  # subset data by cell type and split
  obs_filt <- (adata$obs$cell_type == cell_type) & (adata$obs$split %in% par$input_splits)
  cell_type_adata <- adata[obs_filt, ]

  # prepare count data
  counts <- Matrix::t(cell_type_adata$X)
  counts <- as.matrix(counts)

  d <- DGEList(counts)

  # Filter lowly expressed genes
  keep <- filterByExpr(d, design = model.matrix(~ 0 + sm_name + donor_id, data = cell_type_adata$obs))
  d <- d[keep,, keep.lib.sizes=FALSE]


  d <- calcNormFactors(d)

  # create design matrix
  design <- model.matrix(~ 0 + sm_name + donor_id, data = cell_type_adata$obs %>% mutate_all(limma_trafo))

  # Estimate dispersions
  d <- estimateDisp(d, design)

  # Check if dispersions are available
  if(is.null(d$common.dispersion)) {
    stop("Dispersion values could not be estimated.")
  }

  # fit model
  fit <- glmQLFit(d, design, robust=TRUE)
  ql_fits[[cell_type]] <- fit
}

# run DE tests for each [cell_type, sm_name] pair
de_results <- bind_rows(lapply(seq_len(nrow(new_obs)), function(row_i) {
  cat("Computing DE contrasts (", row_i, "/", nrow(new_obs), ")\n")
  cell_type <- new_obs$cell_type[row_i]
  sm_name <- new_obs$sm_name[row_i]

  fit <- ql_fits[[cell_type]]

  # define contrast
  contrast_formula <- paste0("sm_name", limma_trafo(sm_name), " - sm_name", limma_trafo(par$control_compound))
  contrast <- makeContrasts(contrasts = contrast_formula, levels = colnames(coef(fit)))

  # run contrast test
  test_results <- glmQLFTest(fit, contrast = contrast)

  # Convert TopTags object to a data frame
  test_results_df <- as.data.frame(topTags(test_results, n = Inf))

  # Add cell type and sm_name columns
  test_results_df <- test_results_df %>%
    mutate(cell_type = cell_type, sm_name = sm_name, row_i = row_i)

  return(test_results_df)
}))

# Assuming the first column of de_results needs to be set as 'gene'
de_results <- rownames_to_column(de_results, var = "gene")

# Transform DE results and prepare for writing
de_results_final <- de_results %>%
  mutate(
    gene = factor(gene),  # Convert gene names to factor
    adj.P.Value = p.adjust(PValue, method = "BH"),  # Adjust p-values
    sign_log10_pval = sign(logFC) * -log10(ifelse(adj.P.Value == 0, .Machine$double.eps, adj.P.Value)),  # Compute signed log10 p-values
    is_de = PValue < par$de_sig_cutoff,  # Determine significant DE based on unadjusted p-value
    is_de_adj = adj.P.Value < par$de_sig_cutoff  # Determine significant DE based on adjusted p-value
  ) %>%
  as_tibble()

# Update row names in new_obs for easy reference
rownames(new_obs) <- paste0(new_obs$cell_type, ", ", new_obs$sm_name)

# Create variable features (var) DataFrame
new_var <- data.frame(row.names = levels(de_results_final$gene))

# Organize data into layers
layer_names <- c("is_de", "is_de_adj", "logFC", "logCPM", "F", "PValue", "adj.P.Value", "FDR", "sign_log10_pval")
layers <- map(setNames(layer_names, layer_names), ~ {
  de_results_final %>%
    select(gene, row_i, !!sym(.x)) %>%
    arrange(row_i) %>%
    pivot_wider(names_from = gene, values_from = !!sym(.x)) %>%
    select(-row_i) %>%
    as.matrix()
})


# Create an AnnData object with the organized data
output <- anndata::AnnData(
  obs = new_obs,
  var = new_var,
  layers = setNames(layers, layer_names)
)

# Write the AnnData object to file with compression
zz <- output$write_h5ad(par$output, compression = "gzip")
cat("Data written to h5ad file at:", par$output, "\n")

