
import os
import psutil
import time
import pandas as pd

from opendp.typing import *
from opendp.mod import OpenDPException
# from opendp.transformations import make_cast_default, make_bounded_resize, make_sized_bounded_mean
from opendp.measurements import make_base_discrete_laplace
# from opendp.transformations import make_cast, make_impute_constant
import os
from opendp.transformations import make_split_dataframe, make_select_column
from opendp.transformations import make_count, \
    make_clamp, \
    make_cast_default, \
    make_bounded_sum, \
    make_sized_bounded_sum, \
    make_sized_bounded_mean, \
    make_sized_bounded_variance
from opendp.measurements import make_base_laplace
# TODO: explore between make_sized_bounded_sum and make_bounded_sum

from opendp.mod import binary_search, binary_search_param, binary_search_chain
from opendp.transformations import \
    (make_bounded_resize,
     make_cast,
     make_clamp,
     make_impute_constant,
     make_select_column,
     make_split_dataframe, make_sized_bounded_sum)

from commons.stats_vals import OPENDP
from commons.stats_vals import BASE_PATH, EPSILON_VALUES, MEAN, VARIANCE, COUNT, SUM
from commons.utils import save_synthetic_data_query_ouput, update_epsilon_values


from opendp.mod import enable_features
enable_features('contrib')
enable_features("floating-point")

#-----------#
# Constants #
#-----------#
LIB_NAME = OPENDP

# the greatest number of records that any one individual can influence in the dataset
max_influence = 1

# cast_str_int = (
#     # Cast Vec<str> to Vec<Option<int>>
#     make_cast(TIA=str, TOA=int) >>
#     # Replace any elements that failed to parse with 0, emitting a Vec<int>
#     make_impute_constant(0)
# )


def run_opendp_query(query, epsilon_values, per_epsilon_iterations, data_path, column_name):

    #------------#
    # DATASETS   #
    #------------#

    data_preprocessor = (
        # Convert data into a dataframe where columns are of type Vec<str>
        make_split_dataframe(separator=",", col_names=[column_name]) >>
        # Selects a column of df, Vec<str>
        make_select_column(key=column_name, TOA=float)
    )

    for filename in os.listdir(data_path):
        if not filename.endswith(".csv"):
            continue

        # dataset_path = os.path.join(data_path, filename)
        # with open(dataset_path) as input_file:
        #     data = input_file.read()
        # data = data[len(column_name):]

        # preprocess_dataframe = (make_split_dataframe(separator=",",
        #                                              col_names=['values']) >>
        #                         make_select_column(key="values", TOA=float))

        df = pd.read_csv(data_path + filename)
        data = df[column_name]
        data_list = data.tolist()

        num_rows = data.count()

        # the greatest number of records that any one individual can influence in the dataset
        max_influence = 1

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
            for _ in range(per_epsilon_iterations):

                process = psutil.Process(os.getpid())

                if query == COUNT:
                    begin_time = time.time()

                    true_value = make_count(TIA=float)

                    discovered_scale = binary_search_param(
                        lambda s: true_value >> make_base_discrete_laplace(
                            scale=s),
                        d_in=max_influence,
                        d_out=float(epsilon))

                    computation_chain = true_value >> make_base_discrete_laplace(
                        scale=discovered_scale)

                    private_value = computation_chain(data_list)

                else:

                    min_value = float(data.min())
                    max_value = float(data.max())

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
    experimental_query = VARIANCE  # {MEAN, VARIANCE, COUNT, SUM}

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

        run_opendp_query(experimental_query, epsilon_values,
                         per_epsilon_iterations, dataset_path, column_name)
