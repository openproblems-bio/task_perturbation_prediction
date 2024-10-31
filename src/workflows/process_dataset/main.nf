workflow run_wf {
  take:
  input_ch

  main:
  output_ch = input_ch

    | filter_obs.run(
      fromState: [input: "sc_counts"],
      toState: [filtered_sc_counts: "output"]
    )

    | compute_pseudobulk.run(
      fromState: [input: "filtered_sc_counts"],
      toState: [pseudobulk: "output"]
    )

    | filter_vars.run(
      fromState: [input: "pseudobulk",],
      toState: [pseudobulk_filtered: "output"]
    )

    | add_uns_metadata.run(
      fromState: [
        input: "pseudobulk_filtered",
        dataset_id: "dataset_id",
        dataset_name: "dataset_name",
        dataset_summary: "dataset_summary",
        dataset_description: "dataset_description",
        dataset_url: "dataset_url",
        dataset_reference: "dataset_reference",
        dataset_organism: "dataset_organism"
      ],
      toState: [pseudobulk_filtered_with_uns: "output"]
    )

    | split_sc.run(
      fromState: [
        filtered_sc_counts: "filtered_sc_counts",
        pseudobulk_filtered_with_uns: "pseudobulk_filtered_with_uns"
      ],
      toState: [
        sc_train: "sc_train",
        sc_test: "sc_test"
      ]
    )

    | run_limma.run(
      key: "limma_train",
      fromState: { id, state ->
        [
          input: state.pseudobulk_filtered_with_uns,
          input_splits: ["train", "control", "public_test"],
          output_splits: ["train", "control", "public_test"]
        ]
      },
      toState: [de_train: "output"]
    )

    | run_limma.run(
      key: "limma_test",
      fromState: { id, state ->
        [
          input: state.pseudobulk_filtered_with_uns,
          input_splits: ["train", "control", "public_test", "private_test"],
          output_splits: ["private_test"]
        ]
      },
      toState: [de_test: "output"]
    )

    | generate_id_map.run(
      fromState: [de_test: "de_test"],
      toState: [id_map: "id_map"]
    )

    | setState([
      "de_train",
      "de_test",
      "id_map",
      "sc_train",
      "sc_test",
      "pseudobulk_filtered_with_uns"
    ])

  emit:
  output_ch
}
