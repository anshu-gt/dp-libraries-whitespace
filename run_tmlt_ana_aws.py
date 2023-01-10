"""Using Tumult Analytics by Tumult Labs to execute differentially private queries"""

import os
import sys
import psutil
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from tmlt.analytics.query_builder import QueryBuilder
from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.session import Session
from tmlt.analytics.protected_change import AddOneRow

from pyspark.sql import SparkSession

import boto3
import botocore
from awsglue.utils import getResolvedOptions

# queries to experiment
MEAN = "MEAN"
COUNT = "COUNT"
SUM = "SUM"
VARIANCE = "VARIANCE"

EPSILON_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

S3_DATA_BUCKET = "dp-experiments-data-public"
S3_OUTPUT_BUCKET = "dp-experiments-outputs"

#-------#
# Utils #
#-------#


def update_epsilon_values(output_file):
    """
    """
    out_df = pd.read_csv(output_file)
    last_epsilon = out_df["epsilon"].iloc[-1]

    index = EPSILON_VALUES.index(last_epsilon)

    try:
        return EPSILON_VALUES[index+1:]
    except:
        return -1


def strip_end(text, suffix):
    """
    """
    if suffix and text.endswith(suffix):
        return text[:-len(suffix)]
    return text


def save_synthetic_data_query_output_aws(s3_path, lib_name, query, epsilon, filename, error, relative_errors, scaled_errors, time_used, memory_used):
    """
    """

    rounding_val = 2
    out = {}

    out["epsilon"] = epsilon

    # data_<size>_<scale>_<skew>.csv
    data_feats = filename.split("_")
    out["dataset_size"] = data_feats[1]
    out["dataset_scale"] = data_feats[2]
    out["dataset_skew"] = strip_end(data_feats[3], ".csv")

    out["mean_error"] = round(np.mean(error), rounding_val)
    out["stdev_error"] = round(np.std(error), rounding_val)

    out["mean_relative_error"] = round(np.mean(relative_errors), rounding_val)
    out["stdev_relative_error"] = round(np.std(relative_errors), rounding_val)

    out["mean_scaled_error"] = round(np.mean(scaled_errors), rounding_val)
    out["stdev_scaled_error"] = round(np.std(scaled_errors), rounding_val)

    out["mean_time_used"] = round(np.mean(time_used), rounding_val)
    out["mean_memory_used"] = round(np.mean(memory_used), rounding_val)

    df = pd.DataFrame([out])

    directory = f"s3://{s3_path}/outputs/synthetic/{lib_name.lower()}/size_{out['dataset_size']}/"

    output_path = directory + f"{query.lower()}.csv"
    df.to_csv(output_path, mode="a", header=not os.path.exists(
        output_path), index=False)

    print(f"Saved results for epsilon: {epsilon}")

    return out["mean_time_used"]


#-----------#
# Constants #
#-----------#
LIB_NAME = "TUMULT_ANALYTICS"


# function to create a tumult analytics session with a DataFrame
def _create_tmlt_analytics_session(source_id, df):
    return Session.from_dataframe(
        privacy_budget=PureDPBudget(epsilon=float('inf')),
        source_id=source_id,
        dataframe=df
        # protected_change=AddOneRow()
    )


def run_tmlt_analytics_query(query, epsilon_values, per_epsilon_iterations, column_name, limiting_time_sec):
    """"""

    count_exceeded_limit = 0

    SOURCE_ID = "synthetic_data"

    # spark set-up
    spark = SparkSession.builder.getOrCreate()

    # spark = SparkSession.builder\
    #     .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
    #     .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
    #     .getOrCreate()

    #------------#
    # Datasets   #
    #------------#
    client = boto3.client('s3')
    paginator = client.get_paginator('list_objects_v2')
    result = paginator.paginate(Bucket=S3_DATA_BUCKET)
    for page in result:
        if "Contents" in page:
            for key in page["Contents"]:
                filename = key["Key"]

                # looping through the S3 on the appropriate data size subfolder
                if filename.startswith(f"{SOURCE_ID}/size_{dataset_size}/"):
                    print("#"*10)
                    print("Filename: ", filename)
                    print("#"*10)

                    if not filename.endswith(".csv"):
                        continue

                    # spark_df = spark.read.csv(
                    #     data_path + filename, header=True, inferSchema=True)

                    spark_df = spark.read.csv(
                        f"s3a://{S3_DATA_BUCKET}/{filename}", header=True, inferSchema=True)

                    num_rows = spark_df.count()

                    # session builder for tumult analytics
                    session = _create_tmlt_analytics_session(
                        SOURCE_ID, spark_df)

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

                            # generate build
                            if query == "COUNT":
                                begin_time = time.time()
                                query_build = QueryBuilder(SOURCE_ID).count()
                            else:
                                min_value = spark_df.agg(
                                    {column_name: "min"}).first()[0]
                                max_value = spark_df.agg(
                                    {column_name: "max"}).first()[0]

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

                            # compute
                            private_value = session.evaluate(
                                query_build,
                                privacy_budget=PureDPBudget(epsilon=epsilon)
                            ).first()[0]

                            # compute execution time
                            eps_time_used.append(time.time() - begin_time)

                            # compute memory usage
                            eps_memory_used.append(process.memory_info().rss)

                            #---------------------#
                            # Compute true values #
                            #---------------------#
                            if query == "MEAN":
                                true_value = spark_df.agg(
                                    {column_name: "mean"}).first()[0]
                            elif query == "SUM":
                                true_value = spark_df.agg(
                                    {column_name: "sum"}).first()[0]
                            elif query == "VARIANCE":
                                true_value = spark_df.agg(
                                    {column_name: "variance"}).first()[0]
                            elif query == "COUNT":
                                true_value = num_rows

                            # print("min_value: ", min_value)
                            # print("max_value: ", max_value)
                            # print("true_value:", true_value)
                            # print("private_value:", private_value)

                            # compute errors
                            error = abs(true_value - private_value)

                            eps_errors.append(error)
                            eps_relative_errors.append(error/abs(true_value))
                            eps_scaled_errors.append(error/num_rows)

                        mean_time_used = save_synthetic_data_query_output_aws(S3_DATA_BUCKET, LIB_NAME, query, epsilon, filename, eps_errors,
                                                                              eps_relative_errors, eps_scaled_errors, eps_time_used, eps_memory_used)

                        if mean_time_used >= limiting_time_sec:
                            if count_exceeded_limit >= 5:
                                print(
                                    f"{LIB_NAME} | {filename}: Terminated process!! Last executed epsilon {epsilon} for {query} query.")
                                sys.exit()
                            else:
                                count_exceeded_limit += 1


if __name__ == "__main__":

    #----------------#
    # Configurations #
    #----------------#

    s3 = boto3.resource('s3')

    args = getResolvedOptions(sys.argv,
                              ['JOB_NAME',
                               'tmlt_query',
                               'tmlt_iterations',
                               'tmlt_dataset_size'])

    # {MEAN, VARIANCE, COUNT, SUM}
    experimental_query = args['tmlt_query']

    dataset_size = int(args['tmlt_dataset_size'])

    # path to the folder containing CSVs of `dataset_size` size
    dataset_path = f"s3://{S3_DATA_BUCKET}/synthetic_data/size_{dataset_size}/"
    # dataset_path = BASE_PATH + f"datasets/synthetic_data/size_{dataset_size}/"

    # for synthetic datasets the column name is fixed (will change for real-life datasets)
    column_name = "values"

    limiting_time_sec = 0.5

    # number of iterations to run for each epsilon value
    # value should be in [100, 500]
    # for the testing purpose low value is set
    per_epsilon_iterations = int(args['tmlt_iterations'])

    epsilon_values = EPSILON_VALUES

    # get the epsilon values to resume with
    output_file = f"outputs/synthetic/{LIB_NAME.lower()}/size_{dataset_size}/{experimental_query.lower()}.csv"

    # check if file already exists in S3
    try:
        s3.Object(S3_DATA_BUCKET, output_file).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"The file {output_file} does not exist in {S3_DATA_BUCKET}")
        else:
            raise
    else:
        epsilon_values = update_epsilon_values(
            f"s3://{S3_DATA_BUCKET}/{output_file}")
        print(
            f"Epsilon value check before conducting experiment: {epsilon_values}")

    # if os.path.exists(output_file):
    #     epsilon_values = update_epsilon_values(output_file)

    # test if all the epsilon values have NOT been experimented with
    if epsilon_values != -1:

        print("Library: ", LIB_NAME, " on Local")
        print("Query: ", experimental_query)
        print("Iterations: ", per_epsilon_iterations)
        print("Dataset size: ", dataset_size)
        print("Dataset path: ", dataset_path)
        print("Epsilon Values: ", epsilon_values)

        run_tmlt_analytics_query(experimental_query, epsilon_values,
                                 per_epsilon_iterations, column_name, limiting_time_sec)
    else:
        print("Experiment for these params were conducted before.")
