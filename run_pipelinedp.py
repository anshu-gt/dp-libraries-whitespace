import os
import time
import psutil
import pandas as pd
from tqdm import tqdm
import pyspark
import pipeline_dp

import pipeline_dp
from pipeline_dp.private_spark import make_private


from commons.stats_vals import PIPELINEDP
from commons.stats_vals import BASE_PATH, EPSILON_VALUES, MEAN, VARIANCE, COUNT, SUM
from commons.utils import save_synthetic_data_query_ouput, update_epsilon_values

# useful resource
# https://github.com/OpenMined/PipelineDP/blob/main/examples/movie_view_ratings/run_all_frameworks.py
# https://colab.research.google.com/github/OpenMined/PipelineDP/blob/main/examples/restaurant_visits.ipynb

#-----------#
# Constants #
#-----------#
LIB_NAME = PIPELINEDP

# Framework independent function


def run_dp_metric_pipeline(data, epsilon, min_val, max_val, metric, column_name, backend):
    """
    """
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=100, total_delta=1e-7)
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    data_extractors = pipeline_dp.DataExtractors(privacy_id_extractor=lambda row: row["unique_id"],
                                                 partition_extractor=lambda row: 1,
                                                 value_extractor=lambda row: row[column_name])

    params = pipeline_dp.AggregateParams(noise_kind=pipeline_dp.NoiseKind.LAPLACE,
                                         budget_weight=epsilon,
                                         # [pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM],
                                         metrics=[metric],
                                         max_partitions_contributed=1,
                                         max_contributions_per_partition=1,
                                         min_value=min_val,
                                         max_value=max_val)
    # public_partitions=list(range(1, 8))
    rows = [index_row[1] for index_row in data.iterrows()]
    dp_result = dp_engine.aggregate(
        rows, params, data_extractors)  # , public_partitions)

    budget_accountant.compute_budgets()

    return list(dp_result)


def run_pipelinedp_query(query, epsilon_values, per_epsilon_iterations, data_path, column_name):
    """
    """

    backend = pipeline_dp.LocalBackend()

    ###############################################################
    # TODO: Setup Spark
    ###############################################################

    # Here, we use one worker thread to load the file as 1 partition.
    # For a truly distributed calculation, connect to a Spark cluster (e.g.
    # running on some cloud provider).
    # master = "local[1]"  # use one worker thread to load the file as 1 partition
    # conf = pyspark.SparkConf().setMaster(master)
    # sc = pyspark.SparkContext(conf=conf)

    # # movie_views = sc.textFile(FLAGS.input_file) \
    # #         .mapPartitions(parse_partition)
    # backend = pipeline_dp.SparkRDDBackend(sc)

    # budget_accountant = pipeline_dp.NaiveBudgetAccountant(
    #     total_epsilon=100, total_delta=1e-7)
    # dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    #------------#
    # DATASETS   #
    #------------#
    for filename in os.listdir(data_path):

        print("#"*10)
        print("Filename: ", filename)
        print("#"*10)
        if not filename.endswith(".csv"):
            continue

        df = pd.read_csv(data_path + filename)
        df['unique_id'] = range(1, len(df) + 1)

        data = df[column_name]

        #----------#
        # EPSILONS #
        #----------#
        for epsilon in epsilon_values:

            print("epsilon: ", epsilon)

            eps_time_used = []
            eps_memory_used = []
            eps_errors = []
            eps_relative_errors = []
            eps_scaled_errors = []

            for _ in tqdm(range(per_epsilon_iterations)):

                process = psutil.Process(os.getpid())

                if query == COUNT:
                    begin_time = time.time()
                    private_value = run_dp_metric_pipeline(df, epsilon, None, None, pipeline_dp.Metrics.COUNT,
                                                           column_name, backend)
                    private_value = private_value[0][1][0]
                else:
                    min_value = data.min()
                    max_value = data.max()

                    if query == MEAN:
                        begin_time = time.time()
                        private_value = run_dp_metric_pipeline(df, epsilon, min_value, max_value, pipeline_dp.Metrics.MEAN,
                                                               column_name, backend)
                        private_value = private_value[0][1][0]
                    elif query == SUM:
                        begin_time = time.time()
                        private_value = run_dp_metric_pipeline(df, epsilon, min_value, max_value, pipeline_dp.Metrics.SUM,
                                                               column_name, backend)
                        private_value = private_value[0][1][0]
                    elif query == VARIANCE:
                        begin_time = time.time()
                        private_value = run_dp_metric_pipeline(df, epsilon, min_value, max_value, pipeline_dp.Metrics.VARIANCE,
                                                               column_name, backend)
                        private_value = private_value[0][1][0]

                # compute execution time
                eps_time_used.append(time.time() - begin_time)

                # compute memory usage
                eps_memory_used.append(process.memory_info().rss)

                num_rows = data.count()

                if query == MEAN:
                    true_value = data.mean()
                elif query == SUM:
                    true_value = data.sum()
                elif query == VARIANCE:
                    true_value = data.var()
                elif query == COUNT:
                    true_value = num_rows

                # print("min_value: ", min_value)
                # print("max_value: ", max_value)
                print("true_value:", true_value)
                print("private_value:", private_value)

                # compute errors
                error = abs(true_value - private_value)

                eps_errors.append(error)
                eps_relative_errors.append(error/abs(true_value))
                eps_scaled_errors.append(error/num_rows)

            save_synthetic_data_query_ouput(LIB_NAME, query, epsilon, filename, eps_errors,
                                            eps_relative_errors, eps_scaled_errors, eps_time_used, eps_memory_used)


if __name__ == "__main__":

    #----------------#
    # Configurations #
    #----------------#
    experimental_query = MEAN  # {MEAN, VARIANCE, COUNT, SUM}

    dataset_size = 1000  # {}
    dataset_path = BASE_PATH + f"datasets/synthetic_data/size_{dataset_size}/"
    column_name = "values"

    # number of iterations to run for each epsilon value
    # value should be in [100, 500]
    per_epsilon_iterations = 3  # [100, 500]

    epsilon_values = EPSILON_VALUES

    # test whether to resume from the failed epsilon values' run
    output_file = f"outputs/synthetic/{LIB_NAME.lower()}/size_{dataset_size}/{experimental_query}.csv"
    if os.path.exists(output_file):
        epsilon_values = update_epsilon_values(output_file)

    if epsilon_values != -1:

        print("Library: ", LIB_NAME)
        print("Query: ", experimental_query)
        print("Iterations: ", per_epsilon_iterations)
        print("Dataset size: ", dataset_size)
        print("Dataset path: ", dataset_path)
        print("Epsilon Values: ", epsilon_values)

        run_pipelinedp_query(experimental_query, epsilon_values,
                             per_epsilon_iterations, dataset_path, column_name)
