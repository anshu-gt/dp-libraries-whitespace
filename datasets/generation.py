import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
# ## Goal: We would like to create different arrays that represent different characteristic combinations in terms of dataset size (number of elements, sample size), spread (the range of values the elements of the array take, scale), and shape (how symmetric the dataset is, skewness)

# ## Levels:
# ### Dataset size: 1000, 10000, 100000
# ### Spread: scale= 50, scale=250, scale=500
# ### Shape: skew = 0, skew = 5, skew = 50
# ## We define three levels for each of these terms, yielding 27 possible combinations. We will later on apply differencial privacy mechanism to the arrays to test how the error is affected by these characteristics (dataset size, scale, skew).
# ## This notebook only shows how to tune each of the parameters of a skew-normal distribution created with Scipy
# ### Funciton to be used:  skewnorm.rvs(a => for skewness, loc=> for location, scale => for scale, size => datset size)


def dataset_generation(size, scale, skew):
    """
    """
    data_points = skewnorm.rvs(a=skew, loc=0, scale=scale, size=size)
    df = pd.DataFrame()
    df["values"] = [round(point, 2) for point in data_points]

    directory = f"synthetic_data/size_{size}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    df.to_csv(f"{directory}/data_{size}_{scale}_{skew}.csv", index=False)


scale_diversity = [50, 250, 500]
skew_diversity = [0, 5, 50]
size_diversity =  [10000] #[1000, 100000, 10_000_000]

all_combinations = list(itertools.product(*[scale_diversity, skew_diversity]))

for size in size_diversity:
    for comb in all_combinations:
        scale = comb[0]
        skew = comb[1]
        print('Generating dataset for {}'.format(comb))
        dataset_generation(size, scale, skew)
