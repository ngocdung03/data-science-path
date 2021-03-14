# Introduction to Panda
import pandas as pd
f500 = pd.read_csv('f500.csv',index_col=0)
f500.index.name = None

# Inspect the data
f500_type = type(f500)
f500_shape = f500.shape

f500_head = f500.head(6)
f500_tail = f500.tail(8)

print(f500.info())

# Selecting data by labels
industries = f500.loc[:,'industry']
industries_type = type(industries)

# Slice columns with labels:
countries = f500.loc[:,'country']
revenues_years = f500.loc[:,['revenues', 'years_on_global_500_list']]
ceo_to_sector = f500.loc[:,'ceo':'sector']

# Selecting rows
toyota = f500.loc['Toyota Motor']
drink_companies = f500.loc[['Anheuser-Busch InBev', 'Coca-Cola', 'Heineken Holding']]
middle_companies = f500['Tata Motors':'Nationwide']
middle_companies = f500.loc['Tata Motors':'Nationwide', 'rank':'country']

# Using .value_count() to make a frequency table
countries = f500_sel["country"]
country_counts = countries.value_counts()

# Select items from series
countries = f500['country']
countries_counts = countries.value_counts()

india = countries_counts["India"]
north_america = countries_counts[["USA", "Canada", "Mexico"]]

big_movers = f500.loc[["Aviva", "HP", "JD.com", "BHP Billiton"], ["rank","previous_rank"]]
bottom_companies = f500.loc["National Grid":"AutoNation", ["rank","sector","country"]]

## Pandas fundamental
f500_head = f500.head(10)
print(f500_head.info())

rank_change =  f500["previous_rank"] - f500["rank"]

rank_change_max = rank_change.max()
rank_change_min = rank_change.min()

# Varible desribe for numeric and non-numeric ones
rank = f500["rank"]
rank_desc = rank.describe()

prev_rank = f500["previous_rank"]
prev_rank_desc = prev_rank.describe()

# Method chaining
zero_previous_rank = f500["previous_rank"].value_counts().loc[0]

#  find the maximum value for only the numeric columns 
max_f500 = f500.max(axis = 0,  numeric_only = True)

# Describe numeric and non-numeric variables
f500.describe()
f500.describe(include=['O'])

# Assigning in Pandas
f500.loc["Dow Chemical", "ceo"] = "Jim Fitterling"

# Boolean indexing 
motor_bool = f500["industry"] == "Motor Vehicles and Parts"
motor_countries = f500["country"][motor_bool]

# Combine
import numpy as np
prev_rank_before = f500["previous_rank"].value_counts(dropna=False).head()
f500.loc[f500["previous_rank"]==0,"previous_rank"] = np.nan
prev_rank_after = f500["previous_rank"].value_counts(dropna=False).head()

# Add a new column named rank_change to the f500 dataframe and return a series of descriptive statistics for the column
# rank_change = f500["previous_rank"] - f500["rank"]
# rank_change = rank_change.to_frame()
# f500 = np.concatenate([f500, rank_change], axis=1)
# rank_change_desc = rank_change.describe()

f500["rank_change"] = f500["previous_rank"] - f500["rank"]
rank_change_desc = f500["rank_change"].describe()

# Counts of the most common values in the industry column for companies headquartered in the USA or China.
industry_usa = f500["industry"][f500["country"]=="USA"].value_counts().head(2)
sector_china = f500["sector"][f500["country"]=="China"].value_counts().head(3)
