## Transforming data with pandas
import pandas as pd
happiness2015 = pd.read_csv("World_Happiness_2015.csv")
mapping = {'Economy (GDP per Capita)': 'Economy', 'Health (Life Expectancy)': 'Health', 'Trust (Government Corruption)': 'Trust' }
happiness2015 = happiness2015.rename(mapping, axis=1)

# Series.map() vs Series.apply()
def label(element):
    if element > 1:
        return 'High'
    else:
        return 'Low'
economy_impact_map = happiness2015['Economy'].map(label)

economy_impact_apply = happiness2015['Economy'].apply(label)

equal=economy_impact_map.equals(economy_impact_apply)

# only use the Series.apply() method to apply a function with additional arguments element-wise 
def label(element, x):
    if element > x:
        return 'High'
    else:
        return 'Low'
economy_impact_apply = happiness2015['Economy'].apply(label, x=0.8)

# DataFrame.applymap() to multiple columns
factors = ['Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity']
factors_impact = happiness2015[factors].applymap(label)

# we used the df.apply() method to transform multiple columns. This is only possible because the paraemter function operates on a series.
def v_counts(col):
    num = col.value_counts()
    den = col.size
    return num/den

v_counts_pct = factors_impact.apply(v_counts)

# Create a function that converts each of the six factor columns and the Dystopia Residual column to percentages.
factors = ['Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia Residual']

def percentages(col):
    div = col/happiness2015['Happiness Score']
    return div*100

factor_percentages = happiness2015[factors].apply(percentages)

# Melt the dataframe then use vectorized operations to transform the value column at once
main_cols = ['Country', 'Region', 'Happiness Rank', 'Happiness Score']
factors = ['Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia Residual']   #these factors will be included in one column named 'variable' in the new dataframe
melt = pd.melt(happiness2015, id_vars=main_cols, value_vars=factors)
melt['Percentage'] = round(melt['value']/melt['Happiness Score']*100, 2)
#The melt function moved the values in the seven columns - Economy, Health, Family, Freedom, Generosity, Trust, and Dystopia Residual - to the same column, which meant we could transform them all at once.

# Group the data by the 'variable' column in melted table, find the mean value of each variable (or factor), and plot the results 
melt = pd.melt(happiness2015, id_vars = ['Country', 'Region', 'Happiness Rank', 'Happiness Score'], value_vars= ['Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia Residual'])
melt['Percentage'] = round(melt['value']/melt['Happiness Score'] * 100, 2)
pv_melt = melt.pivot_table(values='value', index='variable')  #mean as default
pv_melt.plot(kind='pie', y='value', legend=False)
