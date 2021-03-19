## Data aggregration
import pandas as pd
happiness2015 = pd.read_csv("World_Happiness_2015.csv")
first_5 = happiness2015.head(5)
print(happiness2015.info())

# Calculate mean happiness by region
mean_happiness = {}
regions = happiness2015['Region'].unique()

for region in regions:
    region_group = happiness2015[happiness2015['Region']==region]
    mean = region_group['Happiness Score'].mean()
    mean_happiness[region] = mean
# Built-in methods for groupby and getgroup
grouped = happiness2015.groupby('Region')
aus_nz = grouped.get_group('Australia and New Zealand')

# Inspect the results of .groups:
grouped = happiness2015.groupby('Region')
#grouped.groups     # get more information about the GroupBy object
north_america = happiness2015.iloc[[4,14]]
na_group = grouped.get_group('North America')
equal = north_america == na_group

# Common aggregration method(s)
means = grouped.mean()

# Calculate mean of happiness only in groups by region
# grouped = happiness2015.groupby('Region')
happy_grouped = grouped['Happiness Score']
happy_mean = happy_grouped.mean() 

# Multiple aggregration at once
import numpy as np
grouped = happiness2015.groupby('Region')
happy_grouped = grouped['Happiness Score']
def dif(group):
    return (group.max() - group.mean())
happy_mean_max = happy_grouped.agg([np.mean, np.max])

mean_max_dif = happy_grouped.agg(dif)

# Chaining methods
happiness_means = happiness2015.groupby('Region')['Happiness Score'].mean()

# Apply df.pivot_table() for aggregration
pv_happiness = happiness2015.pivot_table(values='Happiness Score', index='Region', aggfunc=np.mean, margins=True)  #margins: add row for Total
pv_happiness.plot(kind='barh', xlim=(0,10), title='Mean Happiness Scores by Region', legend=False)
world_mean_happiness = happiness2015['Happiness Score'].mean()

# Aggregate multiple functions by df.pivot_table()
grouped = happiness2015.groupby('Region')['Happiness Score', 'Family']
happy_family_stats = grouped.agg([np.min, np.max, np.mean])
pv_happy_family_stats = happiness2015.pivot_table(
    ['Happiness Score', 'Family'],
    'Region',
    aggfunc=[np.min, np.max, np.mean],
    margins=True)