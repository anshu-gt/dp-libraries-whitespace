import numpy as np

# libraries to experiment
TUMULT_ANALYTICS = "TUMULT_ANALYTICS"
OPENDP = "OPENDP"
DIFFPRIVLIB = "DIFFPRIVLIB"
PIPELINEDP = "PIPELINEDP"

# queries to experiment
MEAN_QUERY = "MEAN"
COUNT_QUERY = "COUNT"
SUM_QUERY = "SUM"
VARIANCE_QUERY = "VARIANCE"

# number of iterations to run for each epsilon value
# value should be in [100, 500]
PER_EPSILON_ITERATIONS = 5  # 100

# TODO: configure to take the above imputs from the CLI

# synthetic dataset naming convention: synthetic_<size>_<scale>_<skew>.csv
# real dataset naming convention: real_<name>.csv
DATASET_BASE_PATH = "/Users/anshusingh/DPPCC/whitespace/differential_privacy/datasets/synthetic/skew_normal_data/"

experimental_lib = OPENDP  # TUMULT_ANALYTICS
experimental_query = MEAN_QUERY
column_name = "values"

epsilon_values = list(np.round(np.arange(0.01, 0.1, 0.01), 2)) + \
    list(np.round(np.arange(0.1, 1, 0.1), 2)) + \
    list(np.arange(1, 11, 1))
# [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

if experimental_lib == TUMULT_ANALYTICS:
    from benchmark_libraries.tumult_analytics.runner import run_tmlt_analytics_query
    run_tmlt_analytics_query(TUMULT_ANALYTICS, experimental_query, epsilon_values,
                             PER_EPSILON_ITERATIONS, DATASET_BASE_PATH, column_name)
elif experimental_lib == DIFFPRIVLIB:
    ...
elif experimental_lib == PIPELINEDP:
    ...
elif experimental_lib == OPENDP:
    from benchmark_libraries.opendp.runner import run_opendp_query
    run_opendp_query(OPENDP, experimental_query, epsilon_values,
                     PER_EPSILON_ITERATIONS, DATASET_BASE_PATH, column_name)
