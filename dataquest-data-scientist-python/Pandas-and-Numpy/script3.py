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
