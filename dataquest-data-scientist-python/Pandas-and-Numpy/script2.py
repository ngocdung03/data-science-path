# Boolean indexing witht NumPy
import numpy as np
taxi = np.genfromtxt('nyc_taxis.csv', delimiter = ',', skip_header = 1)
taxi_shape = taxi.shape

# Pickup in January
pickup_month = taxi[:,1]
january_bool = pickup_month == 1
january = pickup_month[january_bool]
january_rides = january.shape[0]

# Select all rows from taxi with values tip amounts of more than 50, and the columns from indexes 5 to 13 inclusive. 
tip_amount = taxi[:,12]
tip_bool = tip_amount > 50
top_tips = taxi[:,5:14][tip_bool]

# Modify data using indexing and slicing
taxi_modified = taxi.copy()
taxi_modified[1066, 5] = 1
taxi_modified[:,0] = 16
taxi_modified[550:551, 7] = taxi_modified[:,7].mean()

# Use shortcut method for indexing and changing values
taxi_copy = taxi.copy()
total_amount = taxi_copy[:,13]
taxi_copy[total_amount<0] = 0

# Use shortcut to modify one column based on another column
# create a new column filled with `0`.
zeros = np.zeros([taxi.shape[0], 1])
taxi_modified = np.concatenate([taxi, zeros], axis=1)
print(taxi_modified)

taxi_modified[taxi_modified[:,5] == 2, 15] = 1
taxi_modified[taxi_modified[:,5] == 3, 15] = 1
taxi_modified[taxi_modified[:,5] == 5, 15] = 1

# Calculate how many trips had JFK Airport as their destination:
jfk = taxi[taxi[:,6]==2]
jfk_count = jfk.shape[0]

# Calculate how many trips from taxi had Laguardia Airport as their destination:
laguardia = taxi[taxi[:,6] == 3]
laguardia_count = laguardia.shape[0]
                 
# Calculate how many trips from taxi had Newark Airport as their destination:
newark = taxi[taxi[:, 6] == 5]
newark_count = newark.shape[0]

# Using boolean indexing to remove any rows that have an average speed for the trip greater than 100 mph (160 kph) which should remove the questionable data we have worked with over the past two missions. Then, we'll use array methods to calculate the mean for specific columns of the remaining data. 
trip_mph = taxi[:,7] / (taxi[:,8] / 3600)
# trip_mph = np.array(trip_mph)
# taxi_modified = np.concatenate([taxi, trip_mph], axis=1)
cleaned_taxi = taxi[trip_mph < 100]

mean_distance = cleaned_taxi[:, 7].mean()

mean_length = cleaned_taxi[:, 8].mean()

mean_total_amount = cleaned_taxi[:, 13].mean()