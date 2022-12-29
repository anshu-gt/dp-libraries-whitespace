"""Using Google's and OpenMinded's PipelineDP library to execute differentially private queries"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm

import pyspark
from pipeline_dp.private_spark import make_private
import pipeline_dp
from pipeline_dp.aggregate_params import VarianceParams, MeanParams, SumParams, CountParams

# from commons.stats_vals import PIPELINEDP
# from commons.stats_vals import BASE_PATH, EPSILON_VALUES, MEAN, VARIANCE, COUNT, SUM
# from commons.utils import save_synthetic_data_query_ouput, update_epsilon_values

#-----------#
# Constants #
#-----------#
LIB_NAME = "PIPELINEDP"


#-------- Change for AWS -----------------------#
import boto3
import botocore
from awsglue.utils import getResolvedOptions

LIB_NAME = "PIPELINEDP"

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


def save_synthetic_data_query_ouput(lib_name, query, epsilon, filename, error, relative_errors, scaled_errors, time_used, memory_used):
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

    directory = f"outputs/synthetic/{lib_name.lower()}/size_{out['dataset_size']}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    output_path = directory + f"{query.lower()}.csv"
    df.to_csv(output_path, mode="a", header=not os.path.exists(
        output_path), index=False)

    print(f"Saved results for epsilon: {epsilon}")


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

#-----------------------------------------------#


def get_spark_context():
    """"""
    # run on spark
    #--------------------------------------------------------------------------------------------#
    # Setup Spark
    # Sample: https://github.com/OpenMined/PipelineDP/blob/main/examples/movie_view_ratings/run_on_spark.py
    #--------------------------------------------------------------------------------------------#

    # connect to a spark cluster for distributed calculation
    master = "local[*]"
    conf = pyspark.SparkConf().setMaster(master)
    sc = pyspark.SparkContext(conf=conf)

    # backend = pipeline_dp.SparkRDDBackend(sc)

    return sc


def _compute_dp_variance(private_rdd, epsilon, min_val, max_val):
    """"""
    # Calculate the private variance
    return private_rdd.variance(
        VarianceParams(
            budget_weight=epsilon,
            noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            min_value=min_val,
            max_value=max_val,
            partition_extractor=lambda _: 1,
            value_extractor=lambda val: val["value"]))


def _compute_dp_mean(private_rdd, epsilon, min_val, max_val):
    """"""
    # Calculate the private mean
    return private_rdd.mean(
        MeanParams(
            budget_weight=epsilon,
            noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            min_value=min_val,
            max_value=max_val,
            partition_extractor=lambda _: 1,
            value_extractor=lambda val: val["value"]))


def _compute_dp_count(private_rdd, epsilon):
    """"""
    # Calculate the private count
    return private_rdd.count(
        CountParams(
            budget_weight=epsilon,
            noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            partition_extractor=lambda _: 1))


def _compute_dp_sum(private_rdd, epsilon, min_val, max_val):
    """"""
    # Calculate the private sum
    return private_rdd.sum(
        SumParams(
            budget_weight=epsilon,
            noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            min_value=min_val,
            max_value=max_val,
            partition_extractor=lambda _: 1,
            value_extractor=lambda val: val["value"]))


def _parse_partition(iterator):
    for idx, line in enumerate(iterator):
        try:
            value = float(line)
            yield {"value": value, "id": idx + 1}
        except:
            pass


def run_pipelinedp_query(query, epsilon_values, per_epsilon_iterations, data_path, column_name):
    """
    """
    saprk_context = get_spark_context()

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
                if filename.startswith(f"synthetic_data/size_{dataset_size}/"):
                    for filename in os.listdir(data_path):

                        print("#"*10)
                        print("Filename: ", filename)
                        print("#"*10)
                        if not filename.endswith(".csv"):
                            continue

                        # df = pd.read_csv(data_path + filename)
                        df = pd.read_csv(f"s3a://{S3_DATA_BUCKET}/{filename}")
                        data = df[column_name]
                        num_rows = data.count()

                        try:
                            # load file data into Spark's RDD
                            data_rdd = saprk_context.textFile(
                                f"s3a://{S3_DATA_BUCKET}/{filename}").mapPartitions(_parse_partition)
                        except Exception as e:
                            print(e)

                        # breakpoint()

                        #----------#
                        # Epsilons #
                        #----------#
                        for epsilon in epsilon_values:

                            print("epsilon: ", epsilon)

                            eps_time_used = []
                            eps_memory_used = []
                            eps_errors = []
                            eps_relative_errors = []
                            eps_scaled_errors = []

                            #------------------------#
                            # Per epsilon iterations #
                            #------------------------#
                            for _ in tqdm(range(per_epsilon_iterations)):

                                begin_time = time.time()

                                budget_accountant = pipeline_dp.NaiveBudgetAccountant(
                                    total_epsilon=100, total_delta=1e-7)

                                # wrap Spark's RDD into its private version
                                private_rdd = make_private(
                                    data_rdd, budget_accountant, lambda val: val["id"])

                                process = psutil.Process(os.getpid())

                                #----------------------------------------#
                                # Compute differentially private queries #
                                #----------------------------------------#
                                if query == COUNT:
                                    dp_result = _compute_dp_count(private_rdd, epsilon)
                                else:
                                    min_value = data.min()
                                    max_value = data.max()

                                    if query == MEAN:
                                        dp_result = _compute_dp_mean(
                                            private_rdd, epsilon, min_value, max_value)
                                    elif query == SUM:
                                        dp_result = _compute_dp_sum(
                                            private_rdd, epsilon, min_value, max_value)
                                    elif query == VARIANCE:
                                        dp_result = _compute_dp_variance(
                                            private_rdd, epsilon, min_value, max_value)

                                budget_accountant.compute_budgets()
                                
                                # rdd action
                                private_value = dp_result.collect()[0][1]

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

    s3 = boto3.resource('s3')

    args = getResolvedOptions(sys.argv,
                              ['JOB_NAME',
                               'pipelinedp_query',
                               'pipelinedp_iterations',
                               'pipelinedp_dataset_size'])

    # {MEAN, VARIANCE, COUNT, SUM}
    experimental_query = args['pipelinedp_query']

    dataset_size = int(args['pipelinedp_dataset_size'])

    # path to the folder containing CSVs of `dataset_size` size
    dataset_path = f"s3://{S3_DATA_BUCKET}/synthetic_data/size_{dataset_size}/"
    # dataset_path = BASE_PATH + f"datasets/synthetic_data/size_{dataset_size}/"

    # for synthetic datasets the column name is fixed (will change for real-life datasets)
    column_name = "values"

    # number of iterations to run for each epsilon value
    # value should be in [100, 500]
    # for the testing purpose low value is set
    per_epsilon_iterations = int(args['pipelinedp_iterations'])

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

        print("Library: ", LIB_NAME, " on Spark")
        print("Query: ", experimental_query)
        print("Iterations: ", per_epsilon_iterations)
        print("Dataset size: ", dataset_size)
        print("Dataset path: ", dataset_path)
        print("Epsilon Values: ", epsilon_values)

        run_pipelinedp_query(experimental_query, epsilon_values,
                             per_epsilon_iterations, dataset_path, column_name)
    else:
        print("Experiment for these params were conducted before.")
