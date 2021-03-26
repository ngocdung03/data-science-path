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
