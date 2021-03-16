# Scatter plots and correlations
import pandas as pd
import matplotlib.pyplot as plt
bike_sharing['dteday'] = pd.to_datetime(bike_sharing['dteday'])

bike_sharing = pd.read_csv("day.csv")
bike_sharing.head(5)
bike_sharing.tail(5)
bike_sharing.info()
# How many rows and columns does this dataset have? Do you see any missing values?

# Plot two line graphs sharing the same x- and y-axis.
plt.plot(bike_sharing['dteday'], bike_sharing['casual'], label='Casual')
plt.plot(bike_sharing['dteday'], bike_sharing['registered'], label='Registered')
plt.xticks(rotation=30)   #rotate x-tick labels
plt.xlabel('Date')
plt.ylabel('Bikes Rented')
plt.title('Bikes Rented: Casual vs. Registered')
plt.legend()
plt.show()

# Plot 2: Seasonal changes
plt.plot(bike_sharing['dteday'], bike_sharing['temp'])
plt.xticks(rotation=45)
plt.show() 

# Plot 3: Scatter plot
plt.scatter(bike_sharing['windspeed'], bike_sharing['cnt'])
plt.xlabel('Wind Speed')
plt.ylabel('Bikes Rented')
plt.show()

# Plot 4: Positive relationship
plt.scatter(bike_sharing['atemp'], bike_sharing['registered'])
plt.show()
correlation = 'positive'

# Calculating Pearson's r
temp_atemp_corr = bike_sharing['temp'].corr(bike_sharing['atemp'])
wind_hum_corr = bike_sharing['windspeed'].corr(bike_sharing['hum'])

plt.scatter(bike_sharing['temp'], bike_sharing['atemp'])
plt.xlabel('Air Temperature')
plt.ylabel('Feeling Temperature')
plt.show()

plt.scatter(bike_sharing['windspeed'], bike_sharing['hum'])
plt.xlabel('Wind Speed')
plt.ylabel('Humidity')
plt.show()

## Bar plots, histograms, and distributions
working_days = ['Non-Working Day', 'Working Day']
registered_avg = [2959, 3978]

plt.bar(working_days, registered_avg)
plt.show()

# Plot 2: histogram by weekdays
weekday_averages = bike_sharing.groupby('weekday').mean()[['casual', 'registered']].reset_index() 
plt.bar(weekday_averages['weekday'], weekday_averages['registered'])
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6],
          labels=['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday'],
          rotation=30)
plt.show()

# Generate a grouped frequency table with 10 intervals, sorted ascending
registered_freq = bike_sharing['registered'].value_counts(bins = 10).sort_index()

casual_freq = bike_sharing['casual'].value_counts(bins = 10).sort_index()
