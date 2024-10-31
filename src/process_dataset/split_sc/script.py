import anndata as ad
import numpy as np
import pandas as pd

## VIASH START
par = {
  'filtered_sc_counts': 'resources/neurips-2023-data/sc_counts_cleaned.h5ad',
  'pseudobulk_filtered_with_uns': 'resources/neurips-2023-data/pseudobulk_cleaned.h5ad',
  'sc_train': 'sc_train.h5ad',
  'sc_test': 'sc_test.h5ad'
}
## VIASH END

print(">> Load sc and pseudobulk", flush=True)
filtered_sc_counts = ad.read_h5ad(par["filtered_sc_counts"])
pseudobulk_filtered_with_uns = ad.read_h5ad(par["pseudobulk_filtered_with_uns"])

print(f"single cell: {filtered_sc_counts}")
print(f"pseudobulk: {pseudobulk_filtered_with_uns}")

print(">> Process sc adata", flush=True)
filtered_sc_counts.X = filtered_sc_counts.raw.X
del filtered_sc_counts.raw
filtered_sc_counts = filtered_sc_counts[:, pseudobulk_filtered_with_uns.var_names]

# applying filtering from pseudobulk to sc
pseudobulk_filtered_with_uns.obs["plate_well_cell_type"] = \
    pseudobulk_filtered_with_uns.obs["plate_name"].astype(str) + "_" + \
    pseudobulk_filtered_with_uns.obs["well"].astype(str) + "_" + \
        pseudobulk_filtered_with_uns.obs["cell_type"].astype(str)

filtered_sc_counts = filtered_sc_counts[filtered_sc_counts.obs["plate_well_celltype_reannotated"].isin(
    set(pseudobulk_filtered_with_uns.obs["plate_well_cell_type"].unique()))]

# updating the split from kaggle to neurips
filtered_sc_counts.obs["cell_type_orig_updated"] = filtered_sc_counts.obs["cell_type_orig"].apply(lambda x: "T cells" if x.startswith("T ") else x)
filtered_sc_counts.obs["sm_cell_type_orig"] = filtered_sc_counts.obs["sm_name"].astype(str) + "_" + filtered_sc_counts.obs["cell_type_orig_updated"].astype(str)
mapping_to_split = filtered_sc_counts.obs.groupby("sm_cell_type_orig")["split"].apply(lambda x: x.unique()[0]).to_dict()
filtered_sc_counts.obs["sm_cell_type"] = filtered_sc_counts.obs["sm_name"].astype(str) + "_" + filtered_sc_counts.obs["cell_type"].astype(str)
filtered_sc_counts.obs["split"] = filtered_sc_counts.obs["sm_cell_type"].map(mapping_to_split)
filtered_sc_counts.obs['control'] = filtered_sc_counts.obs['sm_name'].eq("Dimethyl Sulfoxide").astype(int)
filtered_sc_counts.obs["orig_split"] = filtered_sc_counts.obs["split"].copy()
filtered_sc_counts.obs['split'] = np.where(
    filtered_sc_counts.obs['split'] == 'private_test', 'test', 'train')

# remove var (keeping the gene names), uns, obsm and obsp
columns_to_remove_from_obs = ["cell_id", "leiden_res1", "group", "cell_type_orig", "cell_type_orig_updated", "sm_cell_type_orig", "orig_split"]
filtered_sc_counts.obs = filtered_sc_counts.obs.drop(columns=columns_to_remove_from_obs, errors='ignore')
filtered_sc_counts.var = pd.DataFrame(index=filtered_sc_counts.var.index)
filtered_sc_counts.uns.clear()
filtered_sc_counts.obsm.clear()
filtered_sc_counts.obsp.clear()
for col in filtered_sc_counts.obs.columns:
    if col not in ["cell_count_by_well_celltype", "cell_count_by_plate_well", "obs_id"]:
        filtered_sc_counts.obs[col] = filtered_sc_counts.obs[col].astype("category")

print(">> Save sc dataset into splits", flush=True)
filtered_sc_counts[filtered_sc_counts.obs["split"] == "train"].write_h5ad(par["sc_train"], compression="gzip")
filtered_sc_counts[filtered_sc_counts.obs["split"] == "test"].write_h5ad(par["sc_test"], compression="gzip")
