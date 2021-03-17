## Relational plots and multiple variables
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Inspect data
housing = pd.read_csv("housing.csv")
housing.head(5)
housing.tail(5)
housing.info()

# Set up environment of Seaborn and plot
sns.set_theme()
sns.relplot(data=housing, x='Gr Liv Area', y='SalePrice',
           hue='Overall Qual', palette='RdYlGn',
           size='Garage Area', sizes=(1,300))   #The min/max value in the range Tuple maps to the min/max value in the variable.
           #size='Rooms', sizes=[200,50])   #To control the sizes for a categorical variable, we need to use a list or a dict. Instead of specifying the range, we need to specify the sizes for each unique value in the variable.

plt.show()

