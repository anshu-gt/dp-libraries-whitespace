"""Using IBM's Diffprivlib library to execute differentially private queries"""

import os
import time
import psutil
import pandas as pd
from tqdm import tqdm

from diffprivlib import BudgetAccountant
from diffprivlib.tools import count_nonzero, mean, sum, var

from commons.stats_vals import DIFFPRIVLIB
from commons.stats_vals import BASE_PATH, EPSILON_VALUES, MEAN, VARIANCE, COUNT, SUM
from commons.utils import save_synthetic_data_query_ouput, update_epsilon_values

#-----------#
# Constants #
#-----------#
LIB_NAME = DIFFPRIVLIB


def run_diffprivlib_query(query, epsilon_values, per_epsilon_iterations, data_path, column_name):
    """
    """

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

        # setup specific to the library
        # budget_acc = BudgetAccountant()

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
                    private_value = count_nonzero(data, epsilon=epsilon)
                else:
                    min_value = data.min()
                    max_value = data.max()

                    if query == MEAN:
                        begin_time = time.time()
                        private_value = mean(
                            data, epsilon=epsilon, bounds=(min_value, max_value))
                    elif query == SUM:
                        begin_time = time.time()
                        private_value = sum(
                            data, epsilon=epsilon, bounds=(min_value, max_value))
                    elif query == VARIANCE:
                        begin_time = time.time()
                        private_value = var(
                            data, epsilon=epsilon, bounds=(min_value, max_value))

                # compute execution time
                eps_time_used.append(time.time() - begin_time)

                # compute memory usage
                eps_memory_used.append(process.memory_info().rss)  # in bytes

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

    dataset_size = 1000  # {}

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

        print("Library: ", LIB_NAME)
        print("Query: ", experimental_query)
        print("Iterations: ", per_epsilon_iterations)
        print("Dataset size: ", dataset_size)
        print("Dataset path: ", dataset_path)
        print("Epsilon Values: ", epsilon_values)

        run_diffprivlib_query(experimental_query, epsilon_values,
                              per_epsilon_iterations, dataset_path, column_name)
