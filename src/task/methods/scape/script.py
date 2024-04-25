import sys, os, fastparquet
print(sys.executable)
print(os.getcwd())
import pandas as pd
import numpy as np
import tensorflow as tf
print(f"tf version:{tf.__version__}")
print(f"Num GPUs Available:{len(tf.config.list_physical_devices('GPU'))}")
print(tf.config.list_physical_devices('GPU'))
sys.path.insert(0, "./")
import scape
print(f"scape version:{scape.__version__}")

par = dict(
	de_train = "resources/neurips-2023-data/de_train.parquet",
    lfc_train = "resources/neurips-2023-data/lfc_train.parquet",
	id_map = "resources/neurips-2023-data/id_map.csv",
	output = "resources/neurips-2023-data/output_rf.parquet",
	output_dir = "resources/neurips-2023-data/tmp_result",
)

df_de = scape.io.load_slogpvals(par['de_train'])

# create_pseudobulk_profiles(df_de, )
# rna_fc = scape.compute_lfc(rna_pseudo, rna_pseudo_meta)
df_lfc = scape.io.load_lfc(par['lfc_train'])

# Make sure rows/columns are in the same order
df_lfc = df_lfc.loc[df_de.index, df_de.columns]
df_de.shape, df_lfc.shape

# We select only a subset of the genes for the model (top most variant genes)
n_genes = 64
top_genes = scape.util.select_top_variable([df_de], k=n_genes)

cell = "NK cells"
drugs = df_de.loc[df_de.index.get_level_values("cell_type") == cell].index.get_level_values("sm_name").unique().tolist()
len(drugs)

df_id_map = pd.read_csv(par["id_map"])
df_sub_ix = df_id_map.set_index(["cell_type", "sm_name"])
df_sub_ix

base_predictions = []
for i, d in enumerate(drugs):
	print(i, d)
	scm = scape.model.create_default_model(n_genes, df_de, df_lfc)
	result = scm.train(
		val_cells=[cell], 
		val_drugs=[d],
		input_columns=top_genes,
		epochs=300,
		output_folder=f"{par["output_dir"]}/_models",
		config_file_name="config.pkl",
		model_file_name=f"drug{i}.keras",
		baselines=["zero", "slogpval_drug"],
	)
	# Collect prediction in the OOF data
	df_pred = scm.predict(df_sub_ix)
	df_pred = df_pred.loc[:, df_de.columns]
	base_predictions.append(df_pred)

df_sub = pd.DataFrame(np.median(base_predictions, axis=0), index=df_sub_ix.index, columns=df_de.columns)
# df_sub.to_csv(f"{output_dir}/base_predictions.csv")
df_sub

sub_drugs = df_sub_ix.index.get_level_values("sm_name").unique().tolist()
len(sub_drugs)

min_n_top_drugs = 50
n_genes = 256

# This time, exclude control drugs for the calculation of the top genes, in order to
# introduce more variability in the model
top_genes = scape.util.select_top_variable([df_de], k=n_genes, exclude_controls=True)

df_drug_effects = pd.DataFrame(df_de.T.pow(2).mean().pow(0.5).groupby("sm_name").mean().sort_values(ascending=False), columns=["effect"])
df_drug_effects["effect_norm"] = (df_drug_effects["effect"] / df_drug_effects["effect"].sum())*100
df_drug_effects

top_sub_drugs = df_drug_effects.loc[sub_drugs].sort_values("effect", ascending=False).head(min_n_top_drugs).index.tolist()
len(top_sub_drugs)

top_all_drugs = df_drug_effects.head(min_n_top_drugs).index.tolist()
top_drugs = set(top_all_drugs) | set(top_sub_drugs)
len(top_drugs)

df_de_c = df_de[df_de.index.get_level_values("sm_name").isin(top_drugs)]
df_de_c = df_de_c.loc[:, top_genes]
df_de_c

df_lfc_c = df_lfc.loc[df_de_c.index, df_de_c.columns]
df_lfc_c.shape

enhanced_predictions = []
for i, d in enumerate(top_drugs):
    print(i, d)
    scm = scape.model.create_default_model(n_genes, df_de_c, df_lfc_c)
    result = scm.train(
        val_cells=[cell], 
        val_drugs=[d],
        input_columns=top_genes,
        epochs=800,
        output_folder="_models",
        config_file_name="enhanced_config.pkl",
        model_file_name=f"enhanced_drug{i}.keras",
        baselines=["zero", "slogpval_drug"],
    )
    # Collect prediction in the OOF data
    df_pred = scm.predict(df_sub_ix)
    enhanced_predictions.append(df_pred)

df_sub_enhanced = pd.DataFrame(np.median(enhanced_predictions, axis=0), index=df_sub_ix.index, columns=df_de_c.columns)
# df_sub_enhanced.to_csv("enhanced_predictions.csv")
df_sub_enhanced

df_focus = df_sub.copy()
df_focus.update(df_sub_enhanced)
df_focus.loc[("Myeloid cells", "Vorinostat"), "PMF1"]

df_submission = 0.80 * df_focus + 0.20 * df_sub
df_submission

df_submission_data = df_sub_ix.join(df_submission).reset_index(drop=True).set_index("id")
df_submission_data

fastparquet.write(par['output'], df_submission_data)
