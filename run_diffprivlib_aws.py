"""Using IBM's Diffprivlib library to execute differentially private queries"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm

from diffprivlib.tools import count_nonzero, mean, sum, var

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

    s3_filename = filename.split("/")[-1]

    # data_<size>_<scale>_<skew>.csv
    data_feats = s3_filename.split("_")
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

    file_path = f"outputs/synthetic/{lib_name.lower()}/size_{out['dataset_size']}/"
    directory = f"s3://{s3_path}/{file_path}"
    output_path = directory + f"{query.lower()}.csv"
    output_file = f"{file_path}{query.lower()}.csv"

    # check if file already exists in S3
    try:
        s3.Object(S3_DATA_BUCKET, output_file).load()
        df.to_csv(output_path, mode="a", header=False, index=False)

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"Adding header for new file {output_file}")
            df.to_csv(output_path, mode="a", header=True, index=False)
        else:
            raise

    print(f"Saved results for epsilon: {epsilon}")

    return out["mean_time_used"]


#-----------#
# Constants #
#-----------#
LIB_NAME = "DIFFPRIVLIB"


def run_diffprivlib_query(query, epsilon_values, per_epsilon_iterations, column_name, limiting_time_sec):
    """
    """

    count_exceeded_limit = 0

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
                    print("#"*10)
                    print("Filename: ", filename)
                    print("#"*10)
                    if not filename.endswith(".csv"):
                        continue

                    df = pd.read_csv(f"s3a://{S3_DATA_BUCKET}/{filename}")
                    data = df[column_name]

                    num_rows = data.count()

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
                                private_value = count_nonzero(
                                    data, epsilon=epsilon)
                            else:
                                min_value = data.min()
                                max_value = data.max()

                                if query == MEAN:
                                    begin_time = time.time()
                                    private_value = mean(data, epsilon=epsilon, bounds=(
                                        min_value, max_value))
                                elif query == SUM:
                                    begin_time = time.time()
                                    private_value = sum(data, epsilon=epsilon, bounds=(
                                        min_value, max_value))
                                elif query == VARIANCE:
                                    begin_time = time.time()
                                    private_value = var(data, epsilon=epsilon, bounds=(
                                        min_value, max_value))

                            # compute execution time
                            eps_time_used.append(time.time() - begin_time)

                            # compute memory usage
                            eps_memory_used.append(
                                process.memory_info().rss)  # in bytes

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
                               'diffprivlib_query',
                               'diffprivlib_iterations',
                               'diffprivlib_dataset_size'])

    # {MEAN, VARIANCE, COUNT, SUM}
    experimental_query = args['diffprivlib_query']
    dataset_size = int(args['diffprivlib_dataset_size'])

    # path to the folder containing CSVs of `dataset_size` size
    dataset_path = f"s3://{S3_DATA_BUCKET}/synthetic_data/size_{dataset_size}/"

    # for synthetic datasets the column name is fixed (will change for real-life datasets)
    column_name = "values"

    limiting_time_sec = 0.5

    # number of iterations to run for each epsilon value
    # value should be in [100, 500]
    per_epsilon_iterations = int(args['diffprivlib_iterations'])

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

    # test if all the epsilon values have NOT been experimented with
    if epsilon_values != -1:

        print("Library: ", LIB_NAME)
        print("Query: ", experimental_query)
        print("Iterations: ", per_epsilon_iterations)
        print("Dataset size: ", dataset_size)
        print("Dataset path: ", dataset_path)
        print("Epsilon Values: ", epsilon_values)

        run_diffprivlib_query(experimental_query, epsilon_values,
                              per_epsilon_iterations, column_name, limiting_time_sec)
    else:
        print("Experiment for these params were conducted before.")
