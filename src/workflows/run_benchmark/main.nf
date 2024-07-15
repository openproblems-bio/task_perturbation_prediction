// construct list of methods
methods = [
  ground_truth,
  mean_outcome,
  mean_across_celltypes,
  mean_across_compounds,
  sample,
  zeros,
  lgc_ensemble,
  nn_retraining_with_pseudolabels,
  jn_ap_op2,
  scape,
  transformer_ensemble,
  pyboost
]

// construct list of metrics
metrics = [
  mean_rowwise_error,
  mean_rowwise_correlation
]

// helper workflow for starting a workflow based on lists of yaml files
workflow auto {
  findStates(params, meta.config)
    | meta.workflow.run(
      auto: [publish: "state"]
    )
}

// benchmarking workflow
workflow run_wf {
  take:
  input_ch

  main:

  /***************************
   * RUN METHODS AND METRICS *
   ***************************/
  score_ch = input_ch

    | run_benchmark_fun(
      methods: methods,
      metrics: metrics,
      methodFromState: { id, state, comp ->
        def new_args = [
          de_train_h5ad: state.de_train_h5ad,
          id_map: state.id_map,
          layer: state.layer,
          output: 'predictions/$id.$key.output.h5ad',
          output_model: null
        ]
        if (comp.config.functionality.info.type == "control_method") {
          new_args.de_test_h5ad = state.de_test_h5ad
        }
        new_args
      },
      methodToState: ["prediction": "output"],
      metricFromState: [
        de_test_h5ad: "de_test_h5ad",
        de_test_layer: "layer",
        prediction: "prediction"
      ],
      metricToState: ["metric_output": "output"],
      methodAuto: [publish: "state"]
    )
    | joinStates { ids, states ->
      def score_uns = states.collect{it.score_uns}
      def score_uns_yaml_blob = toYamlBlob(score_uns)
      def score_uns_file = tempFile("score_uns.yaml")
      score_uns_file.write(score_uns_yaml_blob)
      
      ["output", [scores: score_uns_file]]
    }

  /******************************
   * GENERATE OUTPUT YAML FILES *
   ******************************/
  // create dataset, method and metric metadata files
  metadata_ch = input_ch
    | create_metadata_files(
      datasetFromState: [input: "de_train_h5ad"],
      methods: methods,
      metrics: metrics,
      meta: meta
    )

  // merge all of the output data 
  output_ch = score_ch
    | mix(metadata_ch)
    | joinStates{ ids, states ->
      def mergedStates = states.inject([:]) { acc, m -> acc + m }
      [ids[0], mergedStates]
    }

  emit:
  output_ch
}





def run_benchmark_fun(args) {
  // required args
  def methods_ = args.methods
  def metrics_ = args.metrics
  def methodFromState = args.methodFromState
  def methodToState = args.methodToState
  def metricFromState = args.metricFromState
  def metricToState = args.metricToState

  assert methods_, "methods must be defined"
  assert metrics_, "metrics must be defined"
  assert methodFromState, "methodFromState must be defined"
  assert methodToState, "methodToState must be defined"
  assert metricFromState, "metricFromState must be defined"
  assert metricToState, "metricToState must be defined"

  // optional args
  def keyPrefix = args.keyPrefix ?: ""
  def methodAuto = args.methodAuto ?: [:]
  def metricAuto = args.metricAuto ?: [:]

  // add the key prefix to the method and metric names
  if (keyPrefix && keyPrefix != "") {
    methods_ = methods.collect{ method ->
      method.run(key: keyPrefix + method.config.functionality.name)
    }
    metrics_ = metrics.collect{ metric ->
      metric.run(key: keyPrefix + metric.config.functionality.name)
    }
  }

  workflow bench {
    take: input_ch

    main:
    output_ch = input_ch
      // run all methods
      | runEach(
        components: methods_,
        filter: { id, state, comp ->
          !state.method_ids || state.method_ids.contains(comp.config.functionality.name)
        },
        id: { id, state, comp ->
          id + "." + comp.config.functionality.name
        },
        fromState: methodFromState,
        toState: methodToState,
        auto: methodAuto
      )

      // run all metrics
      | runEach(
        components: metrics_,
        filter: { id, state, comp ->
          !state.metric_ids || state.metric_ids.contains(comp.config.functionality.name)
        },
        id: { id, state, comp ->
          id + "." + comp.config.functionality.name
        },
        fromState: metricFromState,
        toState: metricToState,
        auto: metricAuto
      )

      // extract the scores
      | extract_metadata.run(
        key: "${keyPrefix}score_uns",
        fromState: [input: "metric_output"],
        toState: { id, output, state ->
          state + [
            score_uns: readYaml(output.output).uns
          ]
        }
      )

    emit: output_ch
  }
  return bench
}


def create_metadata_files(args) {
  // required args
  def meta_ = args.meta
  def methods_ = args.methods
  def metrics_ = args.metrics
  def datasetFromState = args.datasetFromState

  assert meta_, "meta must be defined"
  assert methods_, "methods must be defined"
  assert metrics_, "metrics must be defined"
  assert datasetFromState, "datasetFromState must be defined"

  workflow metadata {
    take: input_ch

    main:
    output_ch = input_ch

      | map{ id, state ->
        [id, state + ["_meta": [join_id: id]]]
      }

      | extract_metadata.run(
        key: "dataset_uns",
        fromState: args.datasetFromState,
        toState: { id, output, state ->
          state + [
            dataset_info: readYaml(output.output).uns
          ]
        }
      )
    
      | joinStates { ids, states ->
        assert states.size() > 0, "no states found"
        assert states[0]._meta, "no _meta found in state[0]"
        assert states.every{it.dataset_info}, "not all states have dataset_info"

        // combine the dataset info into one file
        def dataset_uns = states.collect{it.dataset_info}
        def dataset_uns_yaml_blob = toYamlBlob(dataset_uns)
        def dataset_uns_file = tempFile("dataset_uns.yaml")
        dataset_uns_file.write(dataset_uns_yaml_blob)

        // store the method configs in a file
        def method_configs = methods_.collect{it.config}
        def method_configs_yaml_blob = toYamlBlob(method_configs)
        def method_configs_file = tempFile("method_configs.yaml")
        method_configs_file.write(method_configs_yaml_blob)

        // store the metric configs in a file
        def metric_configs = metrics_.collect{it.config}
        def metric_configs_yaml_blob = toYamlBlob(metric_configs)
        def metric_configs_file = tempFile("metric_configs.yaml")
        metric_configs_file.write(metric_configs_yaml_blob)

        def task_info_file = meta_.resources_dir.resolve("task_info.yaml")

        def new_state = [
          dataset_uns: dataset_uns_file,
          method_configs: method_configs_file,
          metric_configs: metric_configs_file,
          task_info: task_info_file,
          _meta: states[0]._meta
        ]
        ["output", new_state]
      }
    emit: output_ch
  }
  return metadata
}
