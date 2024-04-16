import pandas as pd
import numpy as np
from scipy import sparse

import anndata as ad
import os, binascii, re

de_pert_cols = [
    'sm_name',
    'sm_lincs_id',
    'SMILES',
    'dose_uM',
    'timepoint_hr',
    'cell_type',
]
control_compound = 'Dimethyl_Sulfoxide'

def sum_by(adata: ad.AnnData, col: str) -> ad.AnnData:
    """
    Adapted from this forum post:
    https://discourse.scverse.org/t/group-sum-rows-based-on-jobs-feature/371/4
    """

    # assert pd.api.types.is_categorical_dtype(adata.obs[col])
    assert isinstance(adata.obs[col].dtypes, pd.CategoricalDtype)

    # sum `.X` entries for each unique value in `col`
    cat = adata.obs[col].values
    indicator = sparse.coo_matrix(
        (
            np.broadcast_to(True, adata.n_obs),
            (cat.codes, np.arange(adata.n_obs))
        ),
        shape=(len(cat.categories), adata.n_obs),
    )
    sum_adata = ad.AnnData(
        indicator @ adata.X,
        var=adata.var,
        obs=pd.DataFrame(index=cat.categories),
    )

    # copy over `.obs` values that have a one-to-one-mapping with `.obs[col]`
    obs_cols = list(set(adata.obs.columns) - set([col]))

    one_to_one_mapped_obs_cols = []
    nunique_in_col = adata.obs[col].nunique()
    for other_col in obs_cols:
        if len(adata.obs[[col, other_col]].drop_duplicates()) == nunique_in_col:
            one_to_one_mapped_obs_cols.append(other_col)

    joining_df = adata.obs[[col] + one_to_one_mapped_obs_cols].drop_duplicates().set_index(col)
    assert (sum_adata.obs.index == sum_adata.obs.join(joining_df).index).all()
    sum_adata.obs = sum_adata.obs.join(joining_df)
    sum_adata.obs.index.name = col
    sum_adata.obs = sum_adata.obs.reset_index()
    sum_adata.obs.index = sum_adata.obs.index.astype('str')

    return sum_adata


def create_split_mapping(df):
    # Group by 'sm_name' and 'cell_type', then select the first occurrence of 'split'
    grouped = df.groupby(['sm_name', 'cell_type']).agg({
        'split': 'first'  # Use 'first' to grab the split for each unique group
    }).reset_index()

    # Create a dictionary with tuple keys
    mapping_dict = {(row['sm_name'], row['cell_type']): row['split'] for index, row in grouped.iterrows()}

    return mapping_dict


def _run_limma_for_cell_type(bulk_adata, output_dir, rscript_path, code_dir):
    import limma_utils
    bulk_adata = bulk_adata.copy()

    compound_name_col = de_pert_cols[0]

    # limma doesn't like dashes etc. in the compound names
    rpert_mapping = bulk_adata.obs[compound_name_col].drop_duplicates() \
        .reset_index(drop=True).reset_index() \
        .set_index(compound_name_col)['index'].to_dict()

    bulk_adata.obs['Rpert'] = bulk_adata.obs.apply(
        lambda row: rpert_mapping[row[compound_name_col]],
        axis='columns',
    ).astype('str')

    compound_name_to_Rpert = bulk_adata.obs.set_index(compound_name_col)['Rpert'].to_dict()
    ref_pert = compound_name_to_Rpert[control_compound]

    random_string = binascii.b2a_hex(os.urandom(15)).decode()

    limma_utils.limma_fit(
        bulk_adata,
        design='~0+Rpert+donor_id+plate_name+row',
        output_path=f'{output_dir}/{random_string}_limma.rds',
        plot_output_path=f'{output_dir}/{random_string}_voom',
        exec_path=f'{code_dir}/limma_fit.r',
        Rscript_path=rscript_path
    )

    pert_de_dfs = []

    for pert in bulk_adata.obs['Rpert'].unique():
        if pert == ref_pert:
            continue

        pert_de_df = limma_utils.limma_contrast(
            fit_path=f'{output_dir}/{random_string}_limma.rds',
            contrast='Rpert' + pert + '-Rpert' + ref_pert,
            exec_path=f'{code_dir}/limma_contrast.r',
            Rscript_path=rscript_path
        )

        pert_de_df['Rpert'] = pert

        pert_obs = bulk_adata.obs[bulk_adata.obs['Rpert'].eq(pert)]
        for col in de_pert_cols:
            pert_de_df[col] = pert_obs[col].unique()[0]
        pert_de_dfs.append(pert_de_df)

    de_df = pd.concat(pert_de_dfs, axis=0)

    try:
        os.remove(f'{output_dir}/{random_string}_limma.rds')
        os.remove(f'{output_dir}/{random_string}_voom')
    except FileNotFoundError:
        pass

    return de_df

def make_r_safe_names(name):
    # Replace invalid characters with underscores and remove leading numbers
    safe_name = re.sub(r'\W|^(?=\d)', '_', name)
    return safe_name


def convert_de_df_to_anndata(de_df, de_sig_cutoff):
    de_df = de_df.copy()
    zero_pval_selection = de_df['P.Value'].eq(0)
    de_df.loc[zero_pval_selection, 'P.Value'] = np.finfo(np.float64).eps

    de_df['sign_log10_pval'] = np.sign(de_df['logFC']) * -np.log10(de_df['P.Value'])
    de_df['is_de'] = de_df['P.Value'].lt(de_sig_cutoff)
    de_df['is_de_adj'] = de_df['adj.P.Val'].lt(de_sig_cutoff)

    de_feature_dfs = {}
    for feature in ['is_de', 'is_de_adj', 'sign_log10_pval', 'logFC', 'P.Value', 'adj.P.Val']:
        df = de_df.reset_index().pivot_table(
            index=['gene'],
            columns=de_pert_cols,
            values=[feature],
            dropna=True,
        )
        de_feature_dfs[feature] = df

    multiindex = de_feature_dfs['sign_log10_pval'].T.index
    de_adata = ad.AnnData(de_feature_dfs['sign_log10_pval'].T.reset_index(drop=True), dtype=np.float64)
    de_adata.obs.index = multiindex
    de_adata.obs = de_adata.obs.reset_index()
    de_adata.obs = de_adata.obs.drop(columns=['level_0'])
    de_adata.obs.index = de_adata.obs.index.astype('string')

    de_adata.layers['is_de'] = de_feature_dfs['is_de'].to_numpy().T
    de_adata.layers['is_de_adj'] = de_feature_dfs['is_de_adj'].to_numpy().T
    de_adata.layers['logFC'] = de_feature_dfs['logFC'].to_numpy().T
    de_adata.layers['P.Value'] = de_feature_dfs['P.Value'].to_numpy().T
    de_adata.layers['adj.P.Val'] = de_feature_dfs['adj.P.Val'].to_numpy().T

    return de_adata


def anndata_to_dataframe(adata, split_mapping_dict):
    # Step 1: Extract Metadata
    metadata = adata.obs[['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES']]
    metadata['split'] = metadata.apply(
        lambda row: split_mapping_dict.get((row['sm_name'], row['cell_type']), 'unknown'), axis=1
    )
    metadata["control"] = metadata["split"].eq("control")

    # Step 2: Convert the main data matrix X to a DataFrame
    gene_expression = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs.index)

    # Step 3: Merge the metadata and the gene expression data
    full_data = pd.concat([metadata, gene_expression], axis=1).reset_index(drop=True)

    return full_data