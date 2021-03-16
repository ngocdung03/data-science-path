## Line graphs and time series
# Plotting line graphs
month_number = [1, 2, 3, 4, 5, 6, 7]
new_deaths = [213, 2729, 37718, 184064, 143119, 136073, 165003]
import matplotlib.pyplot as plt
plt.plot(month_number, new_deaths)
plt.title('New Reported Deaths By Month (Globally)')
plt.xlabel('Month Number')
plt.ylabel('Number Of Deaths')
plt.show()

import pandas as pd
who_time_series = pd.read_csv("WHO_time_series.csv")
# Transform to datetime data type
who_time_series["Date_reported"] = pd.to_datetime(who_time_series["Date_reported"])
who_time_series.head(5)
who_time_series.tail(5)
who_time_series.info()

# Plot
def plot_cumulative_cases(country_name):
    country = who_time_series[who_time_series['Country'] == country_name]
    plt.plot(country['Date_reported'], country['Cumulative_cases'])
    plt.title('{}: Cumulative Reported Cases'.format(country_name))
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.show()
    
plot_cumulative_cases('Brazil')
plot_cumulative_cases('Iceland')
plot_cumulative_cases('Argentina')

brazil = 'exponential'
iceland = 'logarithmic'
argentina = 'exponential'

# Plot the evolution of cumulative cases for France, the United Kingdom, and Italy on the same graph.
france = who_time_series[who_time_series['Country'] == 'France']
uk = who_time_series[who_time_series['Country'] == 'The United Kingdom']
italy = who_time_series[who_time_series['Country'] == 'Italy']
plt.plot(france['Date_reported'], france['Cumulative_cases'], label='France')
plt.plot(uk['Date_reported'], uk['Cumulative_cases'], label='The UK')
plt.plot(italy['Date_reported'], italy['Cumulative_cases'], label='Italy')
plt.legend()
plt.show()
greatest_july = 'UK'
lowest_july = 'France'
increase_march = 'Italy'

