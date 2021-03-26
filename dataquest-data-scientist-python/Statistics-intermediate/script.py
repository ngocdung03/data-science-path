## The mean 
import numpy as np
distribution = [0,2,3,3,3,4,13]
mean = np.mean(distribution)
center = False
a= (distribution - mean).sum()
equal_distances = True

### Solution ###
above = []
below = []

for value in distribution:
    if value < mean:
        below.append(mean - value)
    if value > mean:
        above.append(value - mean)
        
equal_distances = (sum(above) == sum(below))
######

# Generate 5000 different distributions, measure the total distances above and below the mean, and check whether they are equal
from numpy.random import randint, seed
import numpy as np 

equal_distances = 0
round_v = np.vectorize(round)
for i in range(5000):
    seed(1)
    distribution = randint(0, 1000, 10)
    mean = distribution.mean()
    distance = distribution - mean
    total_distance = distance.sum()              #(round_v(distance, 1))
    if round(total_distance, 1) == 0:   # This will prevent rounding errors at the 13th or 14th decimal place?
        equal_distances += 1

### Solution ###
for i in range(5000):
    seed(i)
    distribution = randint(0,1000,10)
    mean = sum(distribution) / len(distribution)
    
    above = []
    below = []
    for value in distribution:
        if value == mean:
            continue # continue with the next iteration because the distance is 0
        if value < mean:
            below.append(mean - value)
        if value > mean:
            above.append(value - mean)
    
    sum_above = round(sum(above),1)
    sum_below = round(sum(below),1)
    if (sum_above == sum_below):
        equal_distances += 1
######

# Sigma as iteration
distribution_1 = [42, 24, 32, 11]
distribution_2 = [102, 32, 74, 15, 38, 45, 22]
distribution_3 = [3, 12, 7, 2, 15, 1, 21]

def sigma(arr):
    sum_s = 0
    for i in range(len(arr)):
        sum_s += arr[i]
    return sum_s/len(arr)

mean_1 = sigma(distribution_1)
mean_2 = sigma(distribution_2)
mean_3 = sigma(distribution_3)

# Example: house prices
import pandas as pd
houses = pd.read_table('AmesHousing_1.txt')  #or pd.read_csv('AmesHousing_1', sep='\t')

# Mean of sale price
print(houses['SalePrice'].describe())
print(houses['SalePrice'].mean())

def mean(distribution):
    sum_distribution = 0
    for value in distribution:
        sum_distribution += value
        
    return sum_distribution / len(distribution)
function_mean = mean(houses['SalePrice'])
pandas_mean = houses['SalePrice'].mean()         #Series.mean()
means_are_equal = function_mean == pandas_mean

# Visualize on a scatter plot how the sampling error changes as we increase the sample size
import matplotlib.pyplot as plt

pop_mean = houses['SalePrice'].mean()
sampling_errors = []
sample_sizes = [5]
for i in range(101):
    sample = houses['SalePrice'].sample(sample_sizes[i], random_state=i)
    sample_sizes.append(sample_sizes[i] + 29)
    samp_mean = sample.mean()
    sampling_error = pop_mean - samp_mean
    sampling_errors.append(sampling_error)
sample_sizes.pop()            #Remove last element

plt.scatter(sample_sizes, sampling_errors)
plt.axhline(0)
plt.axvline(2930)
plt.xlabel('Sample size')
plt.ylabel('Sampling error')
plt.show()

### Solution ###
arameter = houses['SalePrice'].mean()
sample_size = 5

sample_sizes = []
sampling_errors = []

for i in range(101):
    sample = houses['SalePrice'].sample(sample_size , random_state = i)
    statistic = sample.mean()
    sampling_error = parameter - statistic
    sampling_errors.append(sampling_error)
    sample_sizes.append(sample_size)   #Pay attention
    sample_size += 29
    
import matplotlib.pyplot as plt
plt.scatter(sample_sizes, sampling_errors)
plt.axhline(0)
plt.axvline(2930)
plt.xlabel('Sample size')
plt.ylabel('Sampling error')
######

# Sample means cluster around the population mean
pop_mean = houses['SalePrice'].mean()
sample_means = []
for i in range(10000):
    sample = houses['SalePrice'].sample(100, random_state=i)
    sample_mean = sample.mean()
    sample_means.append(sample_mean)
plt.hist(sample_means)
plt.axvline(pop_mean)
plt.xlabel('Sample mean')
plt.ylabel('Frequency')
plt.xlim(0, 500000)
plt.show()

# Check whether the population mean is equal to the mean of all the sample means of size 2 that we can get if we do sampling without replacement.
population = pd.Series([3, 7, 2])
sample_replace = (population.sample(2, replace=True)).mean()
unbiased = sample_replace == population.mean()

### Solution ###
population = [3, 7, 2]
samples = [[3, 7], [3, 2],
           [7, 2], [7, 3],
           [2, 3], [2, 7]
          ]

sample_means = []
for sample in samples:
    sample_means.append(sum(sample) / len(sample))
    
population_mean = sum(population) / len(population)
mean_of_sample_means = sum(sample_means) / len(sample_means)

unbiased = (population_mean == mean_of_sample_means)
######

## The Weighted Mean and the Median
#houses_per_year = pd.read_table('houses_per_year.txt')
mean_new = houses_per_year['Mean Price'].mean()
mean_original = houses['SalePrice'].mean()
difference = mean_original - mean_new

# Weighted mean by number of houses sold 
weighted_mean = (houses_per_year['Mean Price']*houses_per_year['Houses Sold']).sum()/houses_per_year['Houses Sold'].sum()
mean_original = houses['SalePrice'].mean()
difference = round(mean_original, 10) - round(weighted_mean, 10)

# Weighted by np.average() comparison
import numpy as np
def weighted_mean(mean_arr, weight_arr):
    return (mean_arr*weight_arr).sum()/weight_arr.sum()

weighted_mean_function = weighted_mean(houses_per_year['Mean Price'], houses_per_year['Houses Sold'])
weighted_mean_numpy = np.average(houses_per_year['Mean Price'], weights = houses_per_year['Houses Sold'])
equal = round(weighted_mean_function,10) == round(weighted_mean_numpy,10)

# Median
distribution1 = sorted(map(str, [23, 24, 22, '20 years or lower,', 23, 42, 35]))
distribution2 = sorted([55, 38, 123, 40, 71])
distribution3 = sorted(map(str, [45, 22, 7, '5 books or lower', 32, 65, '100 books or more']))

median1 = 23
median2 = 55
median3 = 32

# Finding median value of one column of houses data
# '10 or more' is replaced by integer 10 only for sorting purposes
new_array = houses['TotRms AbvGrd'].copy().replace('10 or more', 10).astype(int).unique()
pd.Series(new_array).sort_values()
median = 6

### Solution ###
# Sort the values
rooms = houses['TotRms AbvGrd'].copy()
rooms = rooms.replace({'10 or more': 10})
rooms = rooms.astype(int)
rooms_sorted = rooms.sort_values()

# Find the median
middle_indices = [int((len(rooms_sorted) / 2) - 1),
                  int((len(rooms_sorted) / 2))
                 ] # len - 1 and len because Series use 0-indexing 
middle_values = rooms_sorted.iloc[middle_indices] # make sure you don't use loc[]
median = middle_values.mean()
######

# Compare mean and median in distribution with outliers
houses['Lot Area'].plot.box()
plt.show()
houses['SalePrice'].plot.box()
plt.show()

mean_lot = houses['Lot Area'].mean()
median_lot = houses['Lot Area'].median()
lotarea_difference = mean_lot - median_lot

mean_price= houses['SalePrice'].mean()
median_price = houses['SalePrice'].median()
saleprice_difference = mean_price - median_price

# Compare mean and median 2
mean = houses['Overall Cond'].mean()
median = houses['Overall Cond'].median()
houses['Overall Cond'].plot.hist()
more_representative = 'mean'
# Although it can be argued that it's theoretically unsound to compute the mean for ordinal variables, in the last exercise we found the mean more informative and representative than the median. The truth is that in practice many people get past the theoretical hurdles and use the mean nonetheless because in many cases it's much richer in information than the median.
