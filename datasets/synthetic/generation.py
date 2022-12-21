import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm
import itertools
import pandas as pd

# ## Goal: We would like to create different arrays that represent different characteristic combinations in terms of dataset size (number of elements, sample size), spread (the range of values the elements of the array take, scale), and shape (how symmetric the dataset is, skewness)

# ## Levels:
# ### Dataset size: 1000, 10000, 100000
# ### Spread: scale= 50, scale=250, scale=500
# ### Shape: skew = 0, skew = 5, skew = 50
# ## We define three levels for each of these terms, yielding 27 possible combinations. We will later on apply differencial privacy mechanism to the arrays to test how the error is affected by these characteristics (dataset size, scale, skew).
# ## This notebook only shows how to tune each of the parameters of a skew-normal distribution created with Scipy
# ### Funciton to be used:  skewnorm.rvs(a => for skewness, loc=> for location, scale => for scale, size => datset size)


def dataset_generation(skew, scale, size, f):
    data_points = skewnorm.rvs(a=skew, loc=0, scale=scale, size=size)
    df = pd.DataFrame()
    df["values"] = [round(point, 2) for point in data_points]
    df.to_csv(f"skew_normal_data/synthetic_{size}_{scale}_{skew}.csv", index=False)

    # for i in data:
    #     guestFile = open('.\skew_normal_data\dataset_{}.csv'.format(f), "a")
    #     guestFile.write(str("%.4f" % (i)))
    #     guestFile.write("\n")
    #     guestFile.close()

# a = [[0,5,50],[50,250,500],[1000,10000,100000]]


a = [[0, 5, 50], [50, 250, 500], [1000, 100000, 10_000_000]]  # 1_000_000_000]]

all_combinations = list(itertools.product(*a))

for v in range(len(all_combinations)):
    c = all_combinations[v]
    s = c[0]
    t = c[1]
    u = c[2]
    print('Generating dataset for {}'.format(all_combinations[v]))
    dataset_generation(s, t, u, v+1)
