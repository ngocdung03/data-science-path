## Missing and duplicate data
import pandas as pd
happiness2015 = pd.read_csv("wh_2015.csv")
happiness2016 = pd.read_csv("wh_2016.csv")
happiness2017 = pd.read_csv("wh_2017.csv")

# 1. Check for errors in data cleaning/transformation.
shape_2015 = happiness2015.shape
shape_2016 = happiness2016.shape
shape_2017 = happiness2017.shape

missing_2016 = happiness2016.isnull().sum()
missing_2017 = happiness2017.isnull().sum()

# Combine 3 data
# Update column names
happiness2017.columns = happiness2017.columns.str.replace('.', ' ').str.replace('\s+', ' ').str.strip().str.upper()
happiness2015.columns = happiness2015.columns.str.replace('(', ' ').str.replace(')', ' ').str.replace('\s+', ' ').str.strip().str.upper()
happiness2016.columns = happiness2016.columns.str.replace('(', ' ').str.replace(')', ' ').str.replace('\s+', ' ').str.strip().str.upper()

combined = pd.concat([happiness2015, happiness2016, happiness2017], ignore_index=True)
missing = combined.isnull().sum()

# Drawing heatmap to see the pattern of missing data
import seaborn as sns
combined_updated = combined.set_index('YEAR')
sns.heatmap(combined_updated.isnull(), cbar=False)
# Confirm that the REGION column is missing from the 2017 data
regions_2017 = combined[combined['YEAR']==2017]['REGION']
missing = regions_2017.isnull().sum()

# 2. Use data from additional sources to fill missing values.
# Since the regions are fixed values - the region a country was assigned to in 2015 or 2016 won't change - we should be able to assign the 2015 or 2016 region to the 2017 row.
regions = pd.read_csv('regions.csv')
combined = pd.merge(left=combined, right=regions, on='COUNTRY', how='left')
combined = combined.drop('REGION_x', axis = 1)
missing = combined.isnull().sum()

# Checking for duplicates
combined['COUNTRY'] = combined['COUNTRY'].str.upper()
dups = combined.duplicated(['COUNTRY', 'YEAR'])
combined[dups]
combined = combined.drop_duplicates(['COUNTRY', 'YEAR'])

# 3. Dropping rows/columns with missing data:
columns_to_drop = ['LOWER CONFIDENCE INTERVAL', 'STANDARD ERROR', 'UPPER CONFIDENCE INTERVAL', 'WHISKER HIGH', 'WHISKER LOW']
combined = combined.drop(columns_to_drop, axis=1)
missing = combined.isnull().sum()
# drop all columns in combined with 159 or less non null values.
combined = combined.dropna(thresh=159, axis=1)
missing = combined.isnull().sum()

# Replace the missing happiness scores with the mean.
happiness_mean = combined['HAPPINESS SCORE'].mean()
print(happiness_mean)
combined['HAPPINESS SCORE UPDATED'] = combined['HAPPINESS SCORE'].fillna(happiness_mean)
print(combined['HAPPINESS SCORE UPDATED'].mean())

# The mean for the whole world wouldn't be a good estimate for Sub-Saharan Africa region, where most of missing values arise from.
# Maybe it's better to drop the rows with missing values
combined = combined.dropna()
missing = combined.isnull().sum()