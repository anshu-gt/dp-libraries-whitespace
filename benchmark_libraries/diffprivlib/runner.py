from diffprivlib import BudgetAccountant
from diffprivlib.tools.histograms import histogram, histogramdd, histogram2d
# from diffprivlib.tools.quantiles import quantile, median, percentile
from diffprivlib.tools.utils import count_nonzero, mean, std, sum, var, nanmean, nanstd, nansum, nanvar

# import diffprivlib as dp

dp_mean = mean(X, epsilon=1,, bounds=(0, 1), accountant=acc)
dp_std = var(X, epsilon=1, bounds=(0, 1), accountant=acc)
dp_count = count_nonzero(X, epsilon=1, bounds=(0, 1), accountant=acc)
dp_sum = sum(X, epsilon=1, bounds=(0, 1), accountant=acc)