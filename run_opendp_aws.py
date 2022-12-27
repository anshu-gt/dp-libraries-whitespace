"""Using the OpenDP library to execute differentially private queries by Harvard"""

import os
import sys
import psutil
import time
import boto3
import botocore
from tqdm import tqdm

from opendp.typing import *
# from opendp.mod import OpenDPException
from opendp.mod import binary_search, binary_search_param, binary_search_chain
from opendp.measurements import make_base_discrete_laplace, make_base_laplace
from opendp.transformations import make_count, \
    make_clamp, \
    make_cast, \
    make_cast_default, \
    make_impute_constant, \
    make_bounded_resize, \
    make_bounded_sum, \
    make_select_column, \
    make_split_dataframe, \
    make_sized_bounded_sum, \
    make_sized_bounded_mean, \
    make_sized_bounded_variance

from opendp.mod import enable_features
enable_features('contrib')
enable_features("floating-point")


from awsglue.utils import getResolvedOptions


TUMULT_ANALYTICS = "TUMULT_ANALYTICS"
OPENDP = "OPENDP"
DIFFPRIVLIB = "DIFFPRIVLIB"
PIPELINEDP = "PIPELINEDP"

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



#-----------#
# Constants #
#-----------#
LIB_NAME = OPENDP


def run_opendp_query(query, epsilon_values, per_epsilon_iterations, data_path, column_name):
    """"""

    # data_preprocessor = (
    #     # Convert data into a dataframe where columns are of type Vec<str>
    #     make_split_dataframe(separator=",", col_names=[column_name]) >>
    #     # Selects a column of df, Vec<str>
    #     make_select_column(key=column_name, TOA=float)
    # )

    #------------#
    # DATASETS   #
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
            
                    # dataset_path = os.path.join(data_path, filename)
                    # with open(dataset_path) as input_file:
                    #     data = input_file.read()
                    # data = data[len(column_name):]
            
                    df = pd.read_csv(f"s3a://{S3_DATA_BUCKET}/{filename}")
                    data = df[column_name]
                    data_list = data.tolist()
            
                    num_rows = data.count()
            
                    # setup specific to the library
                    # the greatest number of records that any one individual can influence in the dataset
                    max_influence = 1
            
                    #----------#
                    # Epsilons #
                    #----------#
                    for epsilon in epsilon_values:
            
                        # metrics for `per_epsilon_iterations`
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
            
                                compute_count = make_count(TIA=float)
            
                                discovered_scale = binary_search_param(
                                    lambda s: compute_count >> make_base_discrete_laplace(
                                        scale=s),
                                    d_in=max_influence,
                                    d_out=float(epsilon))
            
                                computation_chain = compute_count >> make_base_discrete_laplace(
                                    scale=discovered_scale)
            
                                private_value = computation_chain(data_list)
            
                            else:
            
                                min_value = data.min()
                                max_value = data.max()
            
                                if query == MEAN:
            
                                    begin_time = time.time()
            
                                    compute_bounded_mean = (
                                        make_clamp(bounds=(min_value, max_value)) >>
                                        make_bounded_resize(size=int(num_rows), bounds=(min_value, max_value), constant=0.) >>
                                        make_sized_bounded_mean(
                                            size=int(num_rows), bounds=(min_value, max_value))
                                    )
            
                                    discovered_scale = binary_search_param(
                                        lambda s: compute_bounded_mean >> make_base_laplace(
                                            scale=s),
                                        d_in=max_influence,
                                        d_out=float(epsilon))
            
                                    computation_chain = compute_bounded_mean >> make_base_laplace(
                                        scale=discovered_scale)
            
                                    private_value = computation_chain(data_list)
            
                                elif query == SUM:
                                    begin_time = time.time()
            
                                    compute_bounded_sum = (
                                        make_clamp(bounds=(min_value, max_value)) >>
                                        make_bounded_sum(bounds=(min_value, max_value))
                                    )
            
                                    discovered_scale = binary_search_param(
                                        lambda s: compute_bounded_sum >> make_base_laplace(
                                            scale=s),
                                        d_in=max_influence,
                                        d_out=float(epsilon))
            
                                    computation_chain = compute_bounded_sum >> make_base_laplace(
                                        scale=discovered_scale)
            
                                    private_value = computation_chain(data_list)
            
                                elif query == VARIANCE:
                                    begin_time = time.time()
            
                                    compute_bounded_variance = (
                                        make_clamp(bounds=(min_value, max_value)) >>
                                        make_bounded_resize(size=int(num_rows), bounds=(min_value, max_value), constant=0.) >>
                                        make_sized_bounded_variance(
                                            size=int(num_rows), bounds=(min_value, max_value))
                                    )
            
                                    discovered_scale = binary_search_param(
                                        lambda s: compute_bounded_variance >> make_base_laplace(
                                            scale=s),
                                        d_in=max_influence,
                                        d_out=float(epsilon))
            
                                    computation_chain = compute_bounded_variance >> make_base_laplace(
                                        scale=discovered_scale)
            
                                    private_value = computation_chain(data_list)
            
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
            
                        save_synthetic_data_query_output_aws(S3_DATA_BUCKET, LIB_NAME, query, epsilon, filename, eps_errors,
                                                        eps_relative_errors, eps_scaled_errors, eps_time_used, eps_memory_used)


if __name__ == "__main__":

    #----------------#
    # Configurations #
    #----------------#
    s3 = boto3.resource('s3')

    args = getResolvedOptions(sys.argv,
                              ['JOB_NAME',
                               'opendp_query',
                               'opendp_iterations',
                               'opendp_dataset_size'])

    experimental_query = args['opendp_query'].lower()  # {MEAN, VARIANCE, COUNT, SUM}
    dataset_size = int(args['opendp_dataset_size'])

    # S3 path to the folder containing CSVs of `dataset_size` size
    dataset_path = f"s3://{S3_DATA_BUCKET}/synthetic_data/size_{dataset_size}/"

    # for synthetic datasets the column name is fixed (will change for real-life datasets)
    column_name = "values"

    # number of iterations to run for each epsilon value
    # value should be in [100, 500]
    per_epsilon_iterations = int(args['opendp_iterations'])

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
        print("updating epsilon values...")
        epsilon_values = update_epsilon_values(f"s3://{S3_DATA_BUCKET}/{output_file}")
        print(f"value is: {epsilon_values}")

    # test if all the epsilon values have NOT been experimented with
    if epsilon_values != -1:

        print("Library: ", LIB_NAME)
        print("Query: ", experimental_query)
        print("Iterations: ", per_epsilon_iterations)
        print("Dataset size: ", dataset_size)
        print("Dataset path: ", dataset_path)
        print("Epsilon Values: ", epsilon_values)
        run_opendp_query(experimental_query, epsilon_values,per_epsilon_iterations, dataset_path, column_name)
    else:
        print("Experiment for these params were conducted before, check: {output_file} in {S3_DATA_BUCKET}")
