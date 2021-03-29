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

## The Mode
houses = pd.read_table('AmesHousing_1.txt')
scale_land = 'ordinal'
scale_roof = 'nominal'
kitchen_variable = 'discrete'

# Def mode function
def return_mode(array):
    count = {}
    for i in array:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    # return the key with the highest count
    return max(count, key = count.get)

mode_function = return_mode(houses['Land Slope'])
mode_method = houses['Land Slope'].mode()
same = mode_function == mode_method

# Def value_counts function
def mode(array):
    counts = {}
    
    for value in array:
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1
    
    return (max(counts, key = counts.get), counts)    #error when returning list?

mode, value_counts = mode(houses['Roof Style'])

print(houses['Roof Style'].value_counts())

# Mode of discrete variable
bedroom_variable = 'discrete'
bedroom_mode = houses['Bedroom AbvGr'].mode()
price_variable = 'continuous'

# Mean, median, mode in skewed  continuous distribution
#import matplotlib.pyplot as plt
houses['Mo Sold'].plot.kde(xlim=(1,12))
plt.axvline(houses['Mo Sold'].mode()[0], color='green', label='Mode')   #mode() return series? error without [0]
plt.axvline(houses['Mo Sold'].median(), color='Orange', label='Median')
plt.axvline(houses['Mo Sold'].mean(), color='black', label='Mean')
plt.legend()

## Measures of Variablity
import pandas as pd
houses = pd.read_table('AmesHousing_1.txt')

def ret_range(arr):
    return arr.max() - arr.min()

range_by_year = {}
for year in houses['Yr Sold'].unique():
    data_by_year = houses[houses['Yr Sold'] == year]
    range_by_year[year] = ret_range(data_by_year['SalePrice'])

# Find the year with the greatest/lowest variability of prices 
def standard_deviation(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
        
    variance = sum(distances) / len(distances)
    
    return sqrt(variance)

dict = {}
for year in houses['Yr Sold'].unique():
    data = houses['SalePrice'][houses['Yr Sold'] == year]
    dict[year] = standard_deviation(data)

greatest_variability = 2006
lowest_variability = 2010

### Solution ###
for year in houses['Yr Sold'].unique():
    year_segment = houses[houses['Yr Sold'] == year]
    st_dev = standard_deviation(year_segment['SalePrice'])
    years[year] = st_dev

# Get years of max and min variability
greatest_variability = max(years, key = years.get) # outputs directly the year with the maximum variability
lowest_variability = min(years, key = years.get) # outputs directly the year with the minimum variability
######

# Sampling repeatedly a known population and see how the sample standard deviations compare on average to the population standard deviation
def standard_deviation(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
    
    variance = sum(distances) / len(distances)
    
    return sqrt(variance)

sample_sd = []
import matplotlib.pyplot as plt
for i in range(5000):
    sample = houses['SalePrice'].sample(10, random_state=i)
    sd = standard_deviation(sample)
    sample_sd.append(sd)
plt.hist(sample_sd)
plt.axvline(standard_deviation(houses['SalePrice']))
# Sample standard deviation usually underestimates the population standard deviation

# Bessel's correction
from math import sqrt
def standard_deviation(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
    
    variance = sum(distances) / (len(distances)-1)
    
    return sqrt(variance)

import matplotlib.pyplot as plt
st_devs = []

for i in range(5000):
    sample = houses['SalePrice'].sample(10, random_state = i)
    st_dev = standard_deviation(sample)
    st_devs.append(st_dev)

plt.hist(st_devs)
plt.axvline(pop_stdev)  # pop_stdev is pre-saved from the last screen

# Compare pandas method and numpy function
sample = houses.sample(100, random_state = 1)
from numpy import std, var
pandas_stdev = sample['SalePrice'].std(ddof=1)  #default, pandas Series.std() method 
numpy_stdev = numpy.std(sample['SalePrice'], ddof=1)  # numpy.std() function
equal_stdevs = pandas_stdev == numpy_stdev
pandas_var = sample['SalePrice'].var(ddof=1)
numpy_var = numpy.var(sample['SalePrice'], ddof=1)  #default ddpf=0
equal_vars = pandas_var == numpy_var

# Assess sample var and std when sampling without replacement.
population = [0, 3, 6]

samples = [[0,3], [0,6],
           [3,0], [3,6],
           [6,0], [6,3]
          ]
sample_vars = []
sample_sd = []
for sample in samples:
    variance = numpy.var(sample, ddof=0)
    sd = numpy.std(sample, ddof=0)
    sample_vars.append(variance)
    sample_sd.append(sd)
    
equal_var = sum(sample_vars)/2 == numpy.var(population,ddof=0)
equal_stdev = sum(sample_sd)/2 == numpy.std(population,ddof=0)

## Z-score
# Generate a kernel density plot for the SalePrice variable to find out how far off $220,000 is from the mean.
import pandas as pd
houses = pd.read_table('AmesHousing_1.txt')
houses['SalePrice'].plot.kde()
plt.axvline(houses['SalePrice'].mean(), color='black', label='Mean')
plt.axvline(houses['SalePrice'].mean()+ houses['SalePrice'].std(ddof=0), color='red', label='Standard deviation')
plt.axvline(220000, color='orange', label = '220000')
plt.legend()
plt.xlim(houses['SalePrice'].min(), houses['SalePrice'].max())

very_expensive = False

# Write a function that takes in a value, the array the value belongs to, and returns the z-score of that value. Make sure your function is flexible enough to compute z-scores for both samples and populations.
import numpy
min_val = houses['SalePrice'].min()
mean_val = houses['SalePrice'].mean()
max_val = houses['SalePrice'].max()

def z_score(value, array, ddof):
    mean = numpy.mean(array)
    std = numpy.std(array, ddof=ddof)
    return (value - mean)/std             #have to have return otherwise error
    
min_z = z_score(min_val, houses['SalePrice'], 0)
mean_z = z_score(mean_val, houses['SalePrice'], 0)
max_z = z_score(max_val, houses['SalePrice'], 0)

# Find out the location for which $200,000 has the z-score closest to 0
def z_score(value, array, bessel = 0):
    mean = sum(array) / len(array)
    
    from numpy import std
    st_dev = std(array, ddof = bessel)
    
    distance = value - mean
    z = distance / st_dev
    
    return z
NAmes = z_score(220000, houses[houses['Neighborhood'] == 'NAmes']['SalePrice'])
CollgCr = z_score(220000, houses[houses['Neighborhood'] == 'CollgCr']['SalePrice'])
OldTown = z_score(220000, houses[houses['Neighborhood'] == 'OldTown']['SalePrice'])
Edwards = z_score(220000, houses[houses['Neighborhood'] == 'Edwards']['SalePrice'])
Somerst = z_score(220000, houses[houses['Neighborhood'] == 'Somerst']['SalePrice'])

best_investment = 'College Creek'
### Solution ###
# Segment the data by location
north_ames = houses[houses['Neighborhood'] == 'NAmes']
clg_creek = houses[houses['Neighborhood'] == 'CollgCr']
old_town = houses[houses['Neighborhood'] == 'OldTown']
edwards = houses[houses['Neighborhood'] == 'Edwards']
somerset = houses[houses['Neighborhood'] == 'Somerst']

# Find the z-score for 200000 for every location
z_by_location = {}
for data, neighborhood in [(north_ames, 'NAmes'), (clg_creek, 'CollgCr'),
                     (old_town, 'OldTown'), (edwards, 'Edwards'),
                     (somerset, 'Somerst')]:
    
    z_by_location[neighborhood] = z_score(200000, data['SalePrice'],
                                          bessel = 0)

# Find the location with the z-score closest to 0
print(z_by_location)
best_investment = 'College Creek'
######


st_devs_away = (220000 - houses['SalePrice'].mean())/houses['SalePrice'].std(ddof=0)

# Convert all values to z-scores
mean = houses['SalePrice'].mean()
st_dev = houses['SalePrice'].std(ddof = 0)
# Transform to z-score distribution
houses['z_prices'] = houses['SalePrice'].apply(
    lambda x: ((x - mean) / st_dev)
    )
z_mean_price = numpy.mean(houses['z_prices'])
z_stdev_price = numpy.std(houses['z_prices'], ddof=0)

houses['z_area'] = houses['Lot Area'].apply(
    lambda x: ((x - houses['Lot Area'].mean()) / houses['Lot Area'].std(ddof = 0))
    )
z_mean_area = numpy.mean(houses['z_area'])
z_stdev_area = numpy.std(houses['z_area'], ddof=0)
# Mean values were both extremely close to 0

# Standardize the population of values stored in the population variable
from numpy import std, mean
population = pd.Series([0,8,0,8])
standardized_pop = population.apply(lambda x: ((x - mean(population)) / std(population)))
mean_z = mean(standardized_pop)
stdev_z = std(standardized_pop) 

# Standardized sample with Bessel's correction
from numpy import std, mean
sample = [0,8,0,8]

x_bar = mean(sample)
s = std(sample, ddof = 1)

standardized_sample = []
for value in sample:
    z = (value - x_bar) / s
    standardized_sample.append(z)
stdev_sample = std(standardized_sample, ddof=1)  #1

# Choose 'better' houses by different grading systems
# Standardize the distributions of the index_1 and index_2 variables. We've coded these columns under the hood, and they're already part of the houses data set.
standardized_index1 = houses['index_1'].apply(lambda x: ((x - mean(houses['index_1'])) / std(houses['index_1'])))
standardized_index2 = houses['index_2'].apply(lambda x: ((x - mean(houses['index_2'])) / std(houses['index_2'])))
print(standardized_index1.head(2))
print(standardized_index2.head(2))
better = 'first'

### Solution ###
mean_index1 = houses['index_1'].mean()
stdev_index1 = houses['index_1'].std(ddof = 0)
houses['z_1'] = houses['index_1'].apply(lambda x: 
                                      (x - mean_index1) / stdev_index1
                                     )

mean_index2 = houses['index_2'].mean()
stdev_index2 = houses['index_2'].std(ddof = 0)
houses['z_2'] = houses['index_2'].apply(lambda x: 
                                      (x - mean_index2) / stdev_index2
                                     )

print(houses[['z_1', 'z_2']].head(2))
better = 'first'
######

# transform back to original values
# We merged the two columns of z-scores together into a new column named z_merged
transformed = houses['z_merged'].apply(lambda z: z*10+50)
mean_transformed = transformed.mean()
stdev_transformed = transformed.std(ddof=0)
