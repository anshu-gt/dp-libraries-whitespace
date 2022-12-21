
import os
import psutil
import time

from opendp.mod import OpenDPException
# from opendp.transformations import make_cast_default, make_bounded_resize, make_sized_bounded_mean
from opendp.measurements import make_base_discrete_laplace
# from opendp.transformations import make_cast, make_impute_constant
import os
from opendp.transformations import make_split_dataframe, make_select_column
from opendp.transformations import make_count, \
    make_clamp, \
    make_sized_bounded_sum, \
    make_sized_bounded_mean, \
    make_sized_bounded_variance
from opendp.measurements import make_base_laplace
# TODO: explore between make_sized_bounded_sum and make_bounded_sum


from opendp.mod import enable_features
enable_features('contrib')
enable_features("floating-point")

# the greatest number of records that any one individual can influence in the dataset
max_influence = 1

# cast_str_int = (
#     # Cast Vec<str> to Vec<Option<int>>
#     make_cast(TIA=str, TOA=int) >>
#     # Replace any elements that failed to parse with 0, emitting a Vec<int>
#     make_impute_constant(0)
# )


def run_opendp_query(lib_name, query, epsilon_values, per_epsilon_iterations, data_path, column_name):

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

        dataset_path = os.path.join(data_path, filename)
        with open(dataset_path) as input_file:
            data = input_file.read()

        # transformed = data_preprocessor(data)
        # print(type(transformed))
        # print(transformed[:6])

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

                if query == "COUNT":
                    begin_time = time.time()

                    count = data_preprocessor >> make_count(TIA=int)
                    # NOT a DP release!
                    true_count = count(data)
                    private_count = count >> make_base_discrete_laplace(
                        scale=1.)

                else:
                    min_value = ...
                    max_value = ...

                    bounds = (min_value, max_value)

                    if query == "MEAN":
                        begin_time = time.time()

                        mean_age_preprocessor = (
                            data_preprocessor >>
                            make_clamp(bounds=bounds) >>

                        )

                    elif query == "SUM":
                        begin_time = time.time()

                        bounded_income_sum = (
                            data_preprocessor >>
                            # Clamp income values
                            make_clamp(bounds=bounds) >>
                            # These bounds must be identical to the clamp bounds, otherwise chaining will fail
                            make_bounded_sum(bounds=bounds) >>
                            make_bounded_resize(size=count_release, bounds=float_age_bounds, constant=20.) >>
                            # Compute the mean
                            make_sized_bounded_mean(
                                size=count_release, bounds=float_age_bounds)
                        )

                        discovered_scale = binary_search_param(
                            lambda s: bounded_income_sum >> make_base_discrete_laplace(
                                scale=s),
                            d_in=max_influence,
                            d_out=1.)

                        private_sum = bounded_income_sum >> make_base_discrete_laplace(
                            scale=discovered_scale)

                    elif query == "VARIANCE":
                        begin_time = time.time()

    # mean_age_preprocessor = (
    #     # Convert data into a dataframe of string columns
    #     make_split_dataframe(separator=",", col_names=col_names) >>
    #     # Selects a column of df, Vec<str>
    #     make_select_column(key="age", TOA=str) >>
    #     # Cast the column as Vec<float>, and fill nulls with the default value, 0.
    #     make_cast_default(TIA=str, TOA=float) >>
    #     # Clamp age values
    #     make_clamp(bounds=age_bounds)
    # )

    #     # add laplace noise
    #     dp_mean = mean_age_preprocessor >> make_base_laplace(scale=1.0)

    #     mean_release = dp_mean(data)
    #     print("DP mean:", mean_release)
