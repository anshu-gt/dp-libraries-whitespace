
import os
import psutil
import time
from tqdm import tqdm

from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.session import Session
from tmlt.analytics.protected_change import AddOneRow

from pyspark.sql import SparkSession

from commons.utils import save_synthetic_data_query_ouput, update_epsilon_values
from commons.stats_vals import BASE_PATH, EPSILON_VALUES, TUMULT_ANALYTICS, MEAN, VARIANCE, COUNT, SUM

#-----------#
# Constants #
#-----------#
LIB_NAME = TUMULT_ANALYTICS

# constants
SOURCE_ID = "synthetic_data"

# function to create a tumult analytics session with a DataFrame


def _create_tmlt_analytics_session(source_id, df):
    return Session.from_dataframe(
        privacy_budget=PureDPBudget(epsilon=float('inf')),
        source_id=source_id,
        dataframe=df,
        protected_change=AddOneRow(),
    )


def run_tmlt_analytics_query(query, epsilon_values, per_epsilon_iterations, data_path, column_name):

    spark = SparkSession.builder.getOrCreate()

    #------------#
    # DATASETS   #
    #------------#
    for filename in os.listdir(data_path):
        if not filename.endswith(".csv"):
            continue

        spark_df = spark.read.csv(
            data_path + filename, header=True, inferSchema=True)

        # for S3
        # spark_df = spark.read.csv(SparkFiles.get(data_path + filename, header=True, inferSchema=True)

        num_rows = spark_df.count()

        # session builder for tumult analytics
        session = _create_tmlt_analytics_session(SOURCE_ID, spark_df)

        #----------#
        # EPSILONS #
        #----------#
        for epsilon in epsilon_values:

            eps_time_used = []
            eps_memory_used = []
            eps_errors = []
            eps_relative_errors = []
            eps_scaled_errors = []

            #------------#
            # ITERATIONS #
            #------------#
            for _ in tqdm(range(per_epsilon_iterations)):

                process = psutil.Process(os.getpid())

                if query == "COUNT":
                    begin_time = time.time()
                    query_build = QueryBuilder(SOURCE_ID).count()
                else:
                    min_value = spark_df.agg(
                        {column_name: "min"}).collect()[0][0]
                    max_value = spark_df.agg(
                        {column_name: "max"}).collect()[0][0]

                    if query == "MEAN":
                        begin_time = time.time()
                        query_build = QueryBuilder(SOURCE_ID).average(
                            column_name, low=min_value, high=max_value)
                    elif query == "SUM":
                        begin_time = time.time()
                        query_build = QueryBuilder(SOURCE_ID).sum(
                            column_name, low=min_value, high=max_value)
                    elif query == "VARIANCE":
                        begin_time = time.time()
                        query_build = QueryBuilder(SOURCE_ID).variance(
                            column_name, low=min_value, high=max_value)

                # compute differentially private query output
                private_value = session.evaluate(
                    query_build,
                    privacy_budget=PureDPBudget(epsilon=epsilon)
                ).collect()[0][0]

                # compute execution time
                eps_time_used.append(time.time() - begin_time)

                # compute memory usage
                eps_memory_used.append(process.memory_info().rss)

                if query == "MEAN":
                    true_value = spark_df.agg(
                        {column_name: "mean"}).collect()[0][0]
                elif query == "SUM":
                    true_value = spark_df.agg(
                        {column_name: "sum"}).collect()[0][0]
                elif query == "VARIANCE":
                    true_value = spark_df.agg(
                        {column_name: "variance"}).collect()[0][0]
                elif query == "COUNT":
                    true_value = num_rows

                # print("true_value: ", true_value)
                # print("private_value: ", private_value)
                # print("memory_list: ", memory_list)
                # print("time_list: ", time_list)

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
    dataset_path = BASE_PATH + "datasets/synthetic_data/size_1000/"
    column_name = "values"

    # number of iterations to run for each epsilon value
    # value should be in [100, 500]
    per_epsilon_iterations = 100  # [100, 500]

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

        run_tmlt_analytics_query(experimental_query, epsilon_values,
                                 per_epsilon_iterations, dataset_path, column_name)
