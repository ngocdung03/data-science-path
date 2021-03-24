## Working with missing data
import pandas as pd
mvc = pd.read_csv("nypd_mvc_2018.csv")
# Create a series that counts the number of null values in each of the columns
null_counts = mvc.isnull().sum()

# Check if total_killed != pedestrians_killed + cyclist_killed + motorist_killed
killed_cols = [col for col in mvc.columns if 'killed' in col]
killed = mvc[killed_cols].copy()
killed_manual_sum = killed.iloc[:, 0:3].sum(axis=1)
killed_mask = killed_manual_sum != killed['total_killed']
killed_non_eq = killed[killed_mask]
# filling null values with the sum of the columns is a fairly good choice for our imputation, given that only six rows out of around 58,000 don't match this pattern.

import numpy as np

# In 'total_killed', five null values to one, and flagged some suspicious data as NaN
killed['total_killed'] = killed['total_killed'].mask(killed['total_killed'].isnull(), killed_manual_sum)
killed['total_killed'] = killed['total_killed'].mask(killed['total_killed'] != killed_manual_sum, np.nan)

# Create an injured dataframe and manually sum values
injured = mvc[[col for col in mvc.columns if 'injured' in col]].copy()
injured_manual_sum = injured.iloc[:,:3].sum(axis=1)
injured['total_injured'] = injured['total_injured'].mask(injured['total_injured'].isnull(), injured_manual_sum)
injured['total_injured'] = injured['total_injured'].mask(injured['total_injured'] != injured_manual_sum, np.nan)

# Summarize the count of null values before and after our changes
summary = {
    'injured': [
        mvc['total_injured'].isnull().sum(),
        injured['total_injured'].isnull().sum()
    ],
    'killed': [
        mvc['total_killed'].isnull().sum(),
        killed['total_killed'].isnull().sum()
    ]
}
print(pd.DataFrame(summary, index=['before','after']))

# Assign the values from the killed and injured dataframe back to the main mvc dataframe
mvc['total_injured'] = injured['total_injured']
mvc['total_killed'] = killed['total_killed']killed['total_killed'] = mvc['total_killed']

# Visualize the missing values by heatmap
def plot_null_matrix(df, figsize=(18,15)):
    # initiate the figure
    plt.figure(figsize=figsize)
    # create a boolean dataframe based on whether values are null
    df_null = df.isnull()
    # create a heatmap of the boolean dataframe
    sns.heatmap(~df_null, cbar=False, yticklabels=False)
    plt.xticks(rotation=90, size='x-large')
    plt.show()
plot_null_matrix(mvc)
# The last 10 columns seem to break into two groups of five, with each group of five having similar patterns of null/non-null values.

# Examine the pattern in the last 10 columns 
# Correlation table
cols_with_missing_vals = mvc.columns[mvc.isnull().sum() > 0]
missing_corr = mvc[cols_with_missing_vals].isnull().corr()
print(missing_corr)
# Correlation plot
import matplotlib.pyplot as plt
import seaborn as sns

#  Plot correlations between null values in a dataframe.
def plot_null_correlations(df):
    # create a correlation matrix only for columns with at least
    # one missing value
    cols_with_missing_vals = df.columns[df.isnull().sum() > 0]
    missing_corr = df[cols_with_missing_vals].isnull().corr()
    
    # create a mask to avoid repeated values and make
    # the plot easier to read
    missing_corr = missing_corr.iloc[1:, :-1]
    mask = np.triu(np.ones_like(missing_corr), k=1)
    
    # plot a heatmap of the values
    plt.figure(figsize=(20,14))
    ax = sns.heatmap(missing_corr, vmin=-1, vmax=1, cbar=False,
                     cmap='RdBu', mask=mask, annot=True)
    
    # format the text in the plot to make it easier to read
    for text in ax.texts:
        t = float(text.get_text())
        if -0.05 < t < 0.01:
            text.set_text('')
        else:
            text.set_text(round(t, 2))
        text.set_fontsize('x-large')
    plt.xticks(rotation=90, size='x-large')
    plt.yticks(rotation=0, size='x-large')

    plt.show()
        
veh_cols = [col for col in mvc.columns if 'vehicle' in col]
plot_null_correlations(mvc[veh_cols])

# When a vehicle is in an accident, there is likely to be a cause, and vice-versa.
# Count when the vehicle is missing and the cause is not missing, and vice-versa
col_labels = ['v_number', 'vehicle_missing', 'cause_missing']

vc_null_data = []

for v in range(1,6):
    v_col = 'vehicle_{}'.format(v)
    c_col = 'cause_vehicle_{}'.format(v)
    #Count the number of rows where the v_col column is null and the c_col column is not null
    v_null = (mvc[v_col].isnull() & mvc[c_col].notnull()).sum()
    c_null = (mvc[v_col].notnull() & mvc[c_col].isnull()).sum()
    vc_null_data.append([v, v_null, c_null])
vc_null_df = pd.DataFrame(vc_null_data, columns=col_labels)

# If dropping rows with missing values - lost 10% of the total data
# identify the most common non-null value for the vehicle columns.
cause_cols = [c for c in mvc.columns if "cause_" in c]
cause = mvc[cause_cols]
cause_1d = cause.stack()
top10_causes = cause_1d.value_counts().head(10)

v_cols = [c for c in mvc.columns if c.startswith("vehicle")]
top10_vehicles = mvc[v_cols].stack().value_counts().head(10)

print(top10_vehicles)
print(top_10_causes)   #Unspecified 57481
# The top "cause" is an "Unspecified" placeholder. This is useful instead of a null value as it makes the distinction between a value that is missing because there were only a certain number of vehicles in the collision versus one that is because the contributing cause for a particular vehicle is unknown.

# For values where the vehicle is null and the cause is non-null, set the vehicle to Unspecified, and vice-versa
def summarize_missing():
    v_missing_data = []

    for v in range(1,6):
        v_col = 'vehicle_{}'.format(v)
        c_col = 'cause_vehicle_{}'.format(v)

        v_missing = (mvc[v_col].isnull() & mvc[c_col].notnull()).sum()
        c_missing = (mvc[c_col].isnull() & mvc[v_col].notnull()).sum()

        v_missing_data.append([v, v_missing, c_missing])

    col_labels = columns=["vehicle_number", "vehicle_missing", "cause_missing"]
    return pd.DataFrame(v_missing_data, columns=col_labels)

summary_before = summarize_missing()

for v in range(1,6):
    v_col = 'vehicle_{}'.format(v)
    c_col = 'cause_vehicle_{}'.format(v)
    v_missing_mask = mvc[v_col].isnull() & mvc[c_col].notnull()
    c_missing_mask = mvc[c_col].isnull() & mvc[v_col].notnull()
    mvc[v_col] = mvc[v_col].mask(v_missing_mask, "Unspecified")
    mvc[c_col] = mvc[c_col].mask(c_missing_mask, "Unspecified")

summary_after = summarize_missing()

# Looking at missing value for location
loc_cols = ['borough', 'location', 'on_street', 'off_street', 'cross_street']
location_data = mvc[loc_cols]
print(location_data.isnull().sum()) # a lot of missing value
plot_null_correlations(location_data)
# off_street and on_street have a near perfect negative correlation. That means for almost every row that has a null value in one column, the other has a non-null value and vice-versa.
sorted_location_data = location_data.sort_values(loc_cols)
plot_null_matrix(sorted_location_data)  #heatmap
# We will be able to impute a lot of the missing values by using the other columns in each row. To do this, we can use geolocation APIs that take either an address or location coordinates, and return information about that location.

# pre-prepared supplemental data using geocoding APIs.
sup_data = pd.read_csv('supplemental_data.csv')

mvc_keys = mvc['unique_key']
sup_keys = sup_data['unique_key']

is_equal = mvc_keys.equals(sup_keys)
print(is_equal) #True

# Use Series.mask() to add our supplemental data to our original data
location_cols = ['location', 'on_street', 'off_street', 'borough']
null_before = mvc[location_cols].isnull().sum()
for col in location_cols:
    mvc[col] = mvc[col].mask(mvc[col].isnull(), sup_data[col])
null_after = mvc[location_cols].isnull().sum()

# If you'd like to continue working with this data, you can:
# Drop the rows that had suspect values for injured and killed totals.
# Clean the values in the vehicle_1 through vehicle_5 columns by analyzing the different values and merging duplicates and near-duplicates.
# Analyze whether collisions are more likely in certain locations, at certain times, or for certain vehicle types.

