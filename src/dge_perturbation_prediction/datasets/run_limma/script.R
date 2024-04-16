requireNamespace("anndata", quietly = TRUE)
options(tidyverse.quiet = TRUE)
library(tidyverse)

## VIASH START
par <- list(
  input = "resources/neurips-2023-data/pseudobulk.h5ad",
  output = "resources/neurips-2023-data/de.h5ad"
)
meta <- list(
  cpus = 30L
)
## VIASH END

# static parameters
control_compound <- "Dimethyl Sulfoxide"
de_sig_cutoff <- 0.05

# load data
adata <- anndata::read_h5ad(par$input)
# > adata
# AnnData object with n_obs × n_vars = 3318 × 21265
#     obs: 'well', 'timepoint_hr', 'col', 'dose_uM', 'plate_name', 'cell_type', 'sm_lincs_id', 'SMILES', 'sm_name', 'library_id', 'cell_id', 'row', 'split', 'donor_id', 'control'
#     layers: 'counts'

# create new obs
new_obs <- adata$obs %>%
  select(cell_type, sm_name, sm_lincs_id, SMILES, split, control) %>%
  distinct() %>%
  filter(sm_name != control_compound)

# run limma
cell_types <- as.character(unique(adata$obs$cell_type))

# helper function for transforming values to limma compatible values
limma_trafo <- function(value) {
  gsub("[^[:alnum:]]", "_", value)
}

# run limma for each cell type
limma_fit_per_celltype <- list()

for (cell_type_i in seq_along(cell_types)) {
  cell_type <- cell_types[[cell_type_i]]
  cat(cell_type_i, "/", length(cell_types), ": Running limma for cell type: ", cell_type, "\n", sep = "")

  # subset to cell type
  cell_type_adata <- adata[adata$obs$cell_type == cell_type, ]

  # calc norm factors
  d0 <- Matrix::t(cell_type_adata$layers[["counts"]]) %>%
    edgeR::DGEList() %>%
    edgeR::calcNormFactors()

  # create design matrix
  mm <- model.matrix(
    ~ 0 + sm_name + donor_id + plate_name + row,
    cell_type_adata$obs %>% mutate_all(limma_trafo)
  )

  # voom transformation
  y <- limma::voom(d0, design = mm, plot = FALSE)

  # run limma
  fit <- limma::lmFit(y, mm)

  limma_fit_per_celltype[[cell_type_i]] <- fit
}


# run limma for each cell type and compound between each perturbation
de_df <- list_rbind(map(
  seq_len(nrow(new_obs)),
  function(row_i) {
    cat(row_i, "/", nrow(new_obs), "\n", sep = "")
    # get cell type and compound
    cell_type <- new_obs$cell_type[[row_i]]
    sm_name <- new_obs$sm_name[[row_i]]
    
    # get limma fit
    fit <- limma_fit_per_celltype[[cell_type]]

    # run contrast fit
    contrast_formula <- paste0(
      "sm_name", limma_trafo(sm_name),
      " - ",
      "sm_name", limma_trafo(control_compound)
    )
    contr <- limma::makeContrasts(
      contrasts = contrast_formula,
      levels = colnames(coef(fit))
    )
    
    limma::contrasts.fit(fit, contr) %>%
      limma::eBayes() %>%
      limma::topTable(n = Inf, sort = "none") %>%
      rownames_to_column("gene") %>%
      mutate(row_i = row_i)
  }
))

# transform data
de_df2 <- de_df %>%
  mutate(
    # convert gene names to factor
    gene = factor(gene),
    # readjust p-values for multiple testing <----- !!!!!!!!!!!!!!!!!!!!!!
    adj.P.Value = p.adjust(P.Value, method = "BH"),
    # compute sign log10 p-values
    sign_log10_pval = sign(logFC) * -log10(ifelse(P.Value == 0, .Machine$double.eps, P.Value)),
    sign_log10_pval_adj = sign(logFC) * -log10(ifelse(adj.P.Value == 0, .Machine$double.eps, adj.P.Value)),
    is_de = P.Value < de_sig_cutoff,
    is_de_adj = adj.P.Val < de_sig_cutoff
  ) %>%
  as_tibble()

rownames(new_obs) <- paste0(new_obs$cell_type, ", ", new_obs$sm_name)
new_var <- data.frame(row.names = levels(de_df2$gene))

# create layers from de_df
layer_names <- c("is_de", "is_de_adj", "logFC", "P.Value", "adj.P.Value", "sign_log10_pval", "sign_log10_pval_adj")
layers <- list()
for (layer_name in layer_names) {
  layers[[layer_name]] <- de_df2 %>%
    select(gene, row_i, !!layer_name) %>%
    arrange(row_i) %>%
    spread(gene, !!layer_name) %>%
    select(-row_i) %>%
    as.matrix()
}

# create anndata object
output <- anndata::AnnData(
  obs = new_obs,
  var = new_var,
  layers = setNames(layers, layer_names)
)

# write to file
output$write_h5ad(par$output, compression = "gzip")
