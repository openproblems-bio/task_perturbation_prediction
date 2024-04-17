workflow run_wf {
  take:
  input_ch

  main:

  output_ch = input_ch
  
    | clean_sc_counts.run(
      fromState: [
        input: "sc_counts",
        lincs_id_compound_mapping: "lincs_id_compound_mapping"
      ],
      toState: [
        sc_counts_cleaned: "output"
      ]
    )

    | compute_pseudobulk.run(
      fromState: [input: "sc_counts_cleaned"],
      toState: [pseudobulk: "output"]
    )

    | run_limma.run(
      key: "limma_train",
      fromState: { id, state ->
        [
          input: state.pseudobulk,
          input_splits: ["train", "control", "public_test"],
          output_splits: ["train", "control", "public_test"]
        ]
      },
      toState: [ de_train_h5ad: "output" ]
    )

    | run_limma.run(
      key: "limma_test",
      fromState: { id, state ->
        [
          input: state.pseudobulk,
          input_splits: ["train", "control", "public_test", "private_test"],
          output_splits: ["private_test"]
        ]
      },
      toState: [ de_test_h5ad: "output" ]
    )

    | convert_h5ad_to_parquet.run(
      fromState: [
        input_train: "de_train_h5ad",
        input_test: "de_test_h5ad"
      ],
      toState: [
        de_train_parquet: "output_train",
        de_test_parquet: "output_test",
        id_map: "output_id_map"
      ]
    )

    | setState ([
      "de_train_h5ad",
      "de_test_h5ad",
      "de_train_parquet",
      "de_test_parquet",
      "id_map"
    ])

  emit:
  output_ch
}
