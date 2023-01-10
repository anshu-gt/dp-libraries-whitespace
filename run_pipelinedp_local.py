"""Using Google's and OpenMinded's PipelineDP library to execute differentially private queries"""

import os
import time
import psutil
import pandas as pd
from tqdm import tqdm

import pipeline_dp
from commons.stats_vals import PIPELINEDP
from commons.stats_vals import BASE_PATH, EPSILON_VALUES, MEAN, VARIANCE, COUNT, SUM
from commons.utils import save_synthetic_data_query_ouput, update_epsilon_values

#-----------#
# Constants #
#-----------#
LIB_NAME = PIPELINEDP


def compute_dp_metric(rows, epsilon, metric, column_name, backend, min_val=None, max_val=None):
    """"""
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=100, total_delta=1e-7)

    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    data_extractors = pipeline_dp.DataExtractors(privacy_id_extractor=lambda row: row["id"],
                                                 partition_extractor=lambda _: 1,
                                                 value_extractor=lambda row: row[column_name])

    params = pipeline_dp.AggregateParams(noise_kind=pipeline_dp.NoiseKind.LAPLACE,
                                         budget_weight=epsilon,
                                         metrics=[metric],
                                         max_partitions_contributed=1,
                                         max_contributions_per_partition=1,
                                         min_value=min_val,
                                         max_value=max_val)

    dp_result = dp_engine.aggregate(rows, params, data_extractors)

    budget_accountant.compute_budgets()

    return list(dp_result)


def run_pipelinedp_query(query, epsilon_values, per_epsilon_iterations, data_path, column_name):
    """
    """

    backend = pipeline_dp.LocalBackend()

    #------------#
    # Datasets   #
    #------------#
    for filename in os.listdir(data_path):

        print("#"*10)
        print("Filename: ", filename)
        print("#"*10)
        if not filename.endswith(".csv"):
            continue

        df = pd.read_csv(data_path + filename)
        data = df[column_name]
        num_rows = data.count()

        # library specific setup
        # Reference: https://pipelinedp.io/key-definitions/
        df['id'] = range(1, len(df) + 1)
        rows = [index_row[1] for index_row in df.iterrows()]

        #----------#
        # Epsilons #
        #----------#
        for epsilon in epsilon_values:
            eps_time_used = []
            eps_memory_used = []
            eps_errors = []
            eps_relative_errors = []
            eps_scaled_errors = []

            #------------------------#
            # Per epsilon iterations #
            #------------------------#
            for _ in tqdm(range(per_epsilon_iterations)):

                process = psutil.Process(os.getpid())

                #----------------------------------------#
                # Compute differentially private queries #
                #----------------------------------------#
                if query == COUNT:
                    begin_time = time.time()
                    dp_result = compute_dp_metric(
                        rows, epsilon, pipeline_dp.Metrics.COUNT, column_name, backend)
                else:
                    min_value = data.min()
                    max_value = data.max()

                    if query == MEAN:
                        begin_time = time.time()
                        dp_result = compute_dp_metric(rows, epsilon, pipeline_dp.Metrics.MEAN,
                                                      column_name, backend, min_value, max_value)
                    elif query == SUM:
                        begin_time = time.time()
                        dp_result = compute_dp_metric(rows, epsilon, pipeline_dp.Metrics.SUM,
                                                      column_name, backend, min_value, max_value)
                    elif query == VARIANCE:
                        begin_time = time.time()
                        dp_result = compute_dp_metric(rows, epsilon, pipeline_dp.Metrics.VARIANCE,
                                                      column_name, backend, min_value, max_value)

                # rdd action
                private_value = dp_result[0][1][0]

                # compute execution time
                eps_time_used.append(time.time() - begin_time)

                # compute memory usage
                eps_memory_used.append(process.memory_info().rss)

                #---------------------#
                # Compute true values #
                #---------------------#
                if query == MEAN:
                    true_value = data.mean()
                elif query == SUM:
                    true_value = data.sum()
                elif query == VARIANCE:
                    true_value = data.var()
                elif query == COUNT:
                    true_value = num_rows

                # print("true_value:", true_value)
                # print("private_value:", private_value)

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
    experimental_query = COUNT  # {MEAN, VARIANCE, COUNT, SUM}

    dataset_size = 10000  # {}

    # path to the folder containing CSVs of `dataset_size` size
    dataset_path = BASE_PATH + f"datasets/synthetic_data/size_{dataset_size}/"

    # for synthetic datasets the column name is fixed (will change for real-life datasets)
    column_name = "values"

    # number of iterations to run for each epsilon value
    # value should be in [100, 500]
    per_epsilon_iterations = 100  # for the testing purpose low value is set

    epsilon_values = EPSILON_VALUES

    # get the epsilon values to resume with
    output_file = f"outputs/synthetic/{LIB_NAME.lower()}/size_{dataset_size}/{experimental_query}.csv"
    if os.path.exists(output_file):
        epsilon_values = update_epsilon_values(output_file)

    # test if all the epsilon values have NOT been experimented with
    if epsilon_values != -1:

        print("Library: ", LIB_NAME, " on Local")
        print("Query: ", experimental_query)
        print("Iterations: ", per_epsilon_iterations)
        print("Dataset size: ", dataset_size)
        print("Dataset path: ", dataset_path)
        print("Epsilon Values: ", epsilon_values)

        run_pipelinedp_query(experimental_query, epsilon_values,
                             per_epsilon_iterations, dataset_path, column_name)
