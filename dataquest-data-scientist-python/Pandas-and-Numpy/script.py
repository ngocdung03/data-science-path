# Dataset data dictionary: https://s3.amazonaws.com/dq-content/289/nyc_taxi_data_dictionary.md
import csv
import numpy as np

# import nyc_taxi.csv as a list of lists
f = open("nyc_taxis.csv", "r")
taxi_list = list(csv.reader(f))

# remove the header row
taxi_list = taxi_list[1:]

# convert all values to floats
converted_taxi_list = []
for row in taxi_list:
    converted_row = []
    for item in row:
        converted_row.append(float(item))
    converted_taxi_list.append(converted_row)

# Convert list of lists into numpy array
taxi = np.array(converted_taxi_list)

# Number of rows and columns
taxi_shape = np.shape(taxi)
taxi_shape #tuple

trip_distance_miles = taxi[:,7]
trip_length_seconds = taxi[:,8]

trip_length_hours = trip_length_seconds / 3600 # 3600 seconds is one hour
trip_mph = trip_distance_miles/trip_length_hours

mph_min = trip_mph.min()
mph_max = trip_mph.max()
mph_mean = trip_mph.mean()

# we'll compare against the first 5 rows only
taxi_first_five = taxi[:5]
# select these columns: fare_amount, fees_amount, tolls_amount, tip_amount
fare_components = taxi_first_five[:,9:13]
# Check that the sum of each row in fare_components equals the value in the total_amount column.
fare_sums = fare_components.sum(1)
fare_totals = taxi_first_five[:,13]

fare_sums 
fare_totals