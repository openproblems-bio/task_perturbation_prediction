library(tidyverse)

# aws s3 sync s3://openproblems-data/resources/perturbation_prediction/results output/benchmark_results

df <- yaml::read_yaml("output/benchmark_results/kaggle_2024-06-02_22-27-09/score_uns.yaml") %>%
  map_dfr(as.data.frame) %>%
  as_tibble

df %>% filter(metric_ids == "mean_rowwise_rmse") %>% arrange(metric_values) %>% select(method_id, metric_values)

#    method_id                       metric_values
#    <chr>                                   <dbl>
#  1 ground_truth                             0   
#  2 nn_retraining_with_pseudolabels          1.29
#  3 scape                                    1.31
#  4 pyboost                                  1.32
#  5 jn_ap_op2                                1.34
#  6 lgc_ensemble                             1.41
#  7 mean_across_compounds                    1.47
#  8 transformer_ensemble                     1.55
#  9 zeros                                    1.57
# 10 mean_outcome                             1.57
# 11 mean_across_celltypes                    2.50
# 12 sample                                   3.02

#######

df <- yaml::read_yaml("output/benchmark_results/run_2024-06-02_22-27-09/score_uns.yaml") %>%
  map_dfr(as.data.frame) %>%
  as_tibble

df %>% filter(metric_ids == "mean_rowwise_rmse") %>% arrange(metric_values) %>% select(method_id, metric_values)

# # A tibble: 12 × 2
#    method_id                       metric_values
#    <chr>                                   <dbl>
#  1 ground_truth                            0    
#  2 nn_retraining_with_pseudolabels         0.757
#  3 scape                                   0.775
#  4 pyboost                                 0.795
#  5 lgc_ensemble                            0.802
#  6 mean_across_celltypes                   0.892
#  7 jn_ap_op2                               0.894
#  8 transformer_ensemble                    0.897
#  9 mean_outcome                            0.899
# 10 zeros                                   0.918
# 11 mean_across_compounds                   0.943
# 12 sample                                  1.36 