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





