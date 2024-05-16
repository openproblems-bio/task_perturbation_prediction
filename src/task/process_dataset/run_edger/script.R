requireNamespace("anndata", quietly = TRUE)
library(dplyr, warn.conflicts = FALSE)
library(tidyr, warn.conflicts = FALSE)
library(purrr, warn.conflicts = FALSE)
library(tibble, warn.conflicts = FALSE)
library(edgeR)
library(parallel)

## VIASH START
par <- list(
  input = "resources/neurips-2023-data/pseudobulk_cleaned.h5ad",
  de_sig_cutoff = 0.05,
  control_compound = "Dimethyl Sulfoxide",
  # for public data
  output = "resources/neurips-2023-data/de_train.h5ad",
  input_splits = c("train", "control", "public_test"),
  output_splits = c("train", "control", "public_test")
  # # for private data
#   output = "resources/neurips-2023-data/de_test.h5ad",
#   input_splits = c("train", "control", "public_test", "private_test"),
#   output_splits = c("private_test")
)
## VIASH END

num_cores <- detectCores() - 1

# load data
adata <- anndata::read_h5ad(par$input)

# select [cell_type, sm_name] pairs which will be used for DE analysis
new_obs <- adata$obs %>%
  select(sm_cell_type, cell_type, sm_name, sm_lincs_id, SMILES, split, control) %>%
  distinct() %>%
  filter(sm_name != par$control_compound)

if (!is.null(par$output_splits)) {
  new_obs <- new_obs %>%
    filter(split %in% par$output_splits)
}

# helper function for transforming values to limma compatible values
limma_trafo <- function(value) {
  gsub("[^[:alnum:]]", "_", value)
}

# run glmQLFit per cell type
ql_fits <- list()

cat("Running glmQLFit\n")

# subset data by split
obs_filt <- adata$obs$split %in% par$input_splits
adata <- adata[obs_filt, ]

counts <- Matrix::t(adata$X)
counts <- as.matrix(counts)
start_time <- Sys.time()
d <- DGEList(counts)
d <- calcNormFactors(d)

design <- model.matrix(~ 0 + sm_cell_type + donor_id, data = adata$obs %>% mutate_all(limma_trafo))
# Estimate dispersions
d <- estimateDisp(d, design)
end_time <- Sys.time()
total_duration <- end_time - start_time
total_seconds <- as.numeric(total_duration, units="secs")
minutes <- as.integer(total_seconds %/% 60)
seconds <- as.integer(total_seconds %% 60)
cat(sprintf("Time in steps before fitting: %d minutes and %d seconds\n", minutes, seconds))

# fit model
start_time <- Sys.time()
fit <- glmQLFit(d, design, robust=TRUE)
end_time <- Sys.time()
total_duration <- end_time - start_time
total_seconds <- as.numeric(total_duration, units="secs")
minutes <- as.integer(total_seconds %/% 60)
seconds <- as.integer(total_seconds %% 60)
cat(sprintf("Total fitting time: %d minutes and %d seconds\n", minutes, seconds))

# Initialize list to store DE results
de_results_list <- list()

start_time <- Sys.time()
# Iterate over rows of new_obs to compute contrasts
de_results_list <- mclapply(seq_len(nrow(new_obs)), function(i) {
  cat("Computing DE contrasts for row:", i, "of", nrow(new_obs), "\n")
  current_obs <- new_obs[i, ]
  sm_cell_type <- current_obs$sm_cell_type

  # Define contrast
  control_name_for_cell_type <- paste(par$control_compound, current_obs$cell_type, sep = "_")
  transformed_sm_cell_type <- limma_trafo(sm_cell_type)
  transformed_control_name <- limma_trafo(control_name_for_cell_type)
  contrast_formula <- paste0("sm_cell_type", transformed_sm_cell_type, " - sm_cell_type", transformed_control_name)
  contrast <- makeContrasts(contrasts = contrast_formula, levels = colnames(coef(fit)))
  test_results <- glmQLFTest(fit, contrast = contrast)

  # Convert TopTags object to a data frame and annotate with relevant columns
  test_results_df <- as.data.frame(topTags(test_results, n = Inf))
  test_results_df$gene <- rownames(test_results_df)
  test_results_df$cell_type <- current_obs$cell_type
  test_results_df$sm_name <- current_obs$sm_name
  test_results_df$row_i <- i

  # Store results
    return(test_results_df)
}, mc.cores = num_cores)

# Combine results from all iterations
de_results <- bind_rows(de_results_list)
de_results_final <- de_results %>%
  mutate(
   gene = factor(gene),
    adj.P.Value = p.adjust(PValue, method = "BH"),
    sign_log10_pval = sign(logFC) * -log10(ifelse(adj.P.Value == 0, .Machine$double.eps, adj.P.Value)),
    is_de = PValue < par$de_sig_cutoff,
    is_de_adj = adj.P.Value < par$de_sig_cutoff
  ) %>%
  as_tibble()

end_time <- Sys.time()
total_duration <- end_time - start_time
total_seconds <- as.numeric(total_duration, units="secs")
minutes <- as.integer(total_seconds %/% 60)
seconds <- as.integer(total_seconds %% 60)
cat(sprintf("Total contrasts time: %d minutes and %d seconds\n", minutes, seconds))

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


output <- anndata::AnnData(
  obs = new_obs,
  var = data.frame(row.names = levels(de_results_final$gene)),
  layers = setNames(layers, layer_names)
)

output$write_h5ad(par$output, compression = "gzip")
cat("DE results written to:", par$output, "\n")
