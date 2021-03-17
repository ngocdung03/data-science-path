## Pandas visualization and grid charts
import pandas as pd
import matplotlib.pyplot as plt
traffic = pd.read_csv("traffic_sao_paulo.csv", sep=';')
traffic.head(5)
traffic.tail(5)
traffic.info()

# Convert outcome varialbe into float then inspect
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].str.replace(',', '.')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].astype(float)

traffic['Slowness in traffic (%)'].describe()
plt.hist(traffic['Slowness in traffic (%)'])
plt.show()

# Generate histograme using Series.plot.hist()
traffic['Slowness in traffic (%)'].plot.hist()
plt.title('Distribution of Slowness in traffic (%)')
plt.xlabel('Slowness in traffic (%)')
plt.show()

# Generate a horizontal bar plot for the incidents.sum() table.
incidents = traffic.drop(['Hour (Coded)', 'Slowness in traffic (%)'],
                        axis=1)
incidents.sum().plot.barh()
plt.show()

# Generate scatter plots
traffic.plot.scatter(x='Slowness in traffic (%)',
                     y='Lack of electricity')
plt.show()

traffic.plot.scatter(x='Slowness in traffic (%)',
                     y='Point of flooding')
plt.show()

traffic.plot.scatter(x='Slowness in traffic (%)',
                     y='Semaphore off')
plt.show()

# Isolate all the rows where traffic slowness is 20% or more. Then, we're going to calculate and visualize the incident frequency.
slowness_20_or_more = traffic[traffic['Slowness in traffic (%)'] >=20].drop(['Slowness in traffic (%)', 'Hour (Coded)'], 
                          axis = 1)
incident_frequencies = slowness_20_or_more.sum()
incident_frequencies.plot.barh()
plt.show()

# Plot traffic slowness by hour in each week day
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
traffic_per_day = {}
for i, day in zip(range(0, 135, 27), days):    #For each key, we have a DataFrame containing only the data for that specific day
    each_day_traffic = traffic[i:i+27]
    traffic_per_day[day] = each_day_traffic
    
for day in days:
    traffic_per_day[day].plot.line(x='Hour (Coded)',
                                   y='Slowness in traffic (%)')
    plt.title(day)
    plt.ylim([0, 25])   # To better compare the graphs, we brought the y-axis to the same range for each plot
    plt.show()

# Put all five line plots on the same graph.
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
traffic_per_day = {}
for i, day in zip(range(0, 135, 27), days):
    each_day_traffic = traffic[i:i+27]
    traffic_per_day[day] = each_day_traffic
    
# use plt.plot() instead of the DataFrame.plot.line() method. That's because DataFrame.plot.line() plots separate graphs by default, which means we won't be able to put all the lines on the same graph.
for day in days:
    plt.plot(traffic_per_day[day]['Hour (Coded)'], 
             traffic_per_day[day]['Slowness in traffic (%)'],
             label = day)
plt.legend()
plt.show()

# Grid chart — also known as a small multiple.
plt.figure(figsize=(10, 12))
for i, day in zip(range(1, 6), days):
    plt.subplot(3, 2, i)
    plt.plot(traffic_per_day[day]['Hour (Coded)'],
             traffic_per_day[day]['Slowness in traffic (%)'])
    plt.title(day)
    plt.ylim(0, 25)
# Add the last chart that contains all the five line graphs
plt.subplot(3, 2, 6)
for day in days:
    plt.plot(traffic_per_day[day]['Hour (Coded)'],
            traffic_per_day[day]['Slowness in traffic (%)'],
            label = day)
    plt.ylim(0, 25)
plt.legend()
plt.show()

## Relational plots and multiple variables





