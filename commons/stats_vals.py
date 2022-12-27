
TUMULT_ANALYTICS = "TUMULT_ANALYTICS"
OPENDP = "OPENDP"
DIFFPRIVLIB = "DIFFPRIVLIB"
PIPELINEDP = "PIPELINEDP"

# queries to experiment
MEAN = "MEAN"
COUNT = "COUNT"
SUM = "SUM"
VARIANCE = "VARIANCE"

BASE_PATH = "/Users/syahriikram/Documents/DPPCC/codebase/dp-libraries-whitespace/"

EPSILON_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

S3_DATA_BUCKET = "dp-experiments-data-public"
S3_OUTPUT_BUCKET = "dp-experiments-outputs"

# list(np.round(np.arange(0.01, 0.1, 0.01), 2)) + \
#     list(np.round(np.arange(0.1, 1, 0.1), 2)) + \
#     list(np.arange(1, 11, 1))
