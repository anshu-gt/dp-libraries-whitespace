from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.session import Session
from tmlt.analytics.protected_change import AddOneRow
# from pyspark import SparkFiles
from pyspark.sql import SparkSession

from utils import *
import os
import psutil
import time
import numpy as np

# function to create a tumult analytics session with a DataFrame

spark = SparkSession.builder.getOrCreate()


def _create_tmlt_analytics_session(source_id, df):
    return Session.from_dataframe(
        privacy_budget=PureDPBudget(epsilon=float('inf')),
        source_id=source_id,
        dataframe=df,
        protected_change=AddOneRow(),
    )


# constants
# update the path
BASE_PATH = "/Users/anshusingh/DPPCC/whitespace/differential_privacy/datasets/synthetic/skew_normal_data/"
SOURCE_ID = "synthetic_data"
NUM_SYNTHETIC_DATASETS = 27
QUERIES = ["MEAN", "SUM", "VARIANCE", "COUNT"]
NUM_ITERATIONS = 100
epsilons = list(np.round(np.arange(0.01, 0.1, 0.01), 2)) + \
    list(np.round(np.arange(0.1, 1, 0.1), 2)) + \
    list(np.arange(1, 11, 1))
# [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#------------#
# QUERIES    #
#------------#
for query in QUERIES:

    #------------#
    # DATASETS   #
    #------------#
    # range(1, NUM_SYNTHETIC_DATASETS + 1):
    for dataset_idx in [1, 2, 3, 4, 5]:

        spark_df = spark.read.csv(
            BASE_PATH + "dataset_{}.csv".format(dataset_idx), header=True, inferSchema=True)

        # for S3
        # spark_df = spark.read.csv(SparkFiles.get(BASE_PATH + "dataset_{}.csv".format(dataset_idx)), header=True, inferSchema=True)

        dataset_size = spark_df.count()

        # session builder for tumult analytics
        session = _create_tmlt_analytics_session(SOURCE_ID, spark_df)

        #------------#
        # EPSILONS   #
        #------------#
        for epsilon in [0.1]:  # epsilons:

            time_list = []
            memory_list = []
            err = []
            relative_err = []
            scaled_err = []

            #------------#
            # ITERATIONS #
            #------------#
            for exp_no in range(0, 1):  # NUM_ITERATIONS):

                process = psutil.Process(os.getpid())
                print("QUERYYYYY:", query)
                if query == "MEAN":
                    min_value = spark_df.agg({"values": "min"}).collect()[0][0]
                    max_value = spark_df.agg({"values": "max"}).collect()[0][0]
                    begin_time = time.time()
                    # low=0, high=120
                    query_build = QueryBuilder(SOURCE_ID).average(
                        "values", low=min_value, high=max_value)
                elif query == "SUM":
                    min_value = spark_df.agg({"values": "min"}).collect()[0][0]
                    max_value = spark_df.agg({"values": "max"}).collect()[0][0]
                    begin_time = time.time()
                    query_build = QueryBuilder(SOURCE_ID).sum(
                        "values", low=min_value, high=max_value)
                elif query == "VARIANCE":
                    min_value = spark_df.agg({"values": "min"}).collect()[0][0]
                    max_value = spark_df.agg({"values": "max"}).collect()[0][0]
                    begin_time = time.time()
                    query_build = QueryBuilder(SOURCE_ID).variance(
                        "values", low=min_value, high=max_value)
                elif query == "COUNT":
                    begin_time = time.time()
                    query_build = QueryBuilder(SOURCE_ID).count()

                private_value = session.evaluate(
                    query_build,
                    privacy_budget=PureDPBudget(epsilon=epsilon)
                ).collect()[0][0]

                # compute execution time
                time_list.append(time.time() - begin_time)

                # compute memory usage
                memory_list.append(process.memory_info().rss)

                if query == "MEAN":
                    true_value = spark_df.agg(
                        {"values": "mean"}).collect()[0][0]
                elif query == "SUM":
                    true_value = spark_df.agg(
                        {"values": "sum"}).collect()[0][0]
                elif query == "VARIANCE":
                    true_value = spark_df.agg(
                        {"values": "variance"}).collect()[0][0]
                elif query == "COUNT":
                    true_value = dataset_size

                print("true_value: ", true_value)
                print("private_value: ", private_value)
                print("memory_list: ", memory_list)
                print("time_list: ", time_list)
                # compute errors

                # TODO outputs need to be updated
                err.append(true_value - private_value)
                relative_err.append(
                    abs((true_value - private_value)/true_value))
                scaled_err.append(
                    abs((true_value - private_value)/dataset_size))

                # TODO: save outputs in a files
