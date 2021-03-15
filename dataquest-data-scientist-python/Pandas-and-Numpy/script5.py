## Data cleaning basics
import pandas as pd
laptops = pd.read_csv("laptops.csv", encoding = "Latin-1")
print(laptops.info())
print(laptops.columns)

# Reformat column names
import pandas as pd
laptops = pd.read_csv('laptops.csv', encoding='Latin-1')

def clean_col(col):
    col = col.strip()    #remove whitespaces at the end/beginning
    col = col.replace("Operating System", "os")
    col = col.replace(" ", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.lower()
    return col

new_columns = []
for column in laptops.columns:
    column = clean_col(column)
    new_columns.append(column)
    
laptops.columns = new_columns

# Convert text to numeric variable
# Identify patterns and special cases
unique_ram = laptops["ram"].unique()
# Remove non-digit characters
laptops['ram'] = laptops['ram'].str.replace('GB', '')
unique_ram = laptops['ram'].unique()
# Convert (or cast) the columns to a numeric dtype
laptops['ram'] = laptops['ram'].astype(int)
dtypes = laptops.dtypes
# Rename the column
laptops["ram"] = laptops["ram"].str.replace('GB','').astype(int)
laptops.rename({"ram":"ram_gb"}, axis=1, inplace = True)
ram_gb_desc = laptops["ram_gb"].describe()

# Extract the manufacturer from the cpu column 
print(laptops["cpu"].head())
laptops["cpu_manufacturer"] = (laptops["cpu"]
                               .str.split()
                               .str[0]
                              )

cpu_manufacturer_counts = laptops["cpu_manufacturer"].value_counts()

# Use Series.map() to correct multiple values
# both the correct and incorrect spelling of macOS were included as keys, otherwise we'll end up with null values.
mapping_dict = {
    'Android': 'Android',
    'Chrome OS': 'Chrome OS',
    'Linux': 'Linux',
    'Mac OS': 'macOS',
    'No OS': 'No OS',
    'Windows': 'Windows',
    'macOS': 'macOS'
}
laptops["os"] = laptops["os"].map(mapping_dict)

# Remove rows/columns that have null values
laptops_no_null_rows = laptops.dropna()
laptops_no_null_cols = laptops.dropna(axis=1)

# Use knowledge to fill the missing values
value_counts_before = laptops.loc[laptops["os_version"].isnull(), "os"].value_counts()
laptops.loc[laptops["os"] == "macOS", "os_version"] = "X"

laptops.loc[laptops["os"] == "No OS" , "os_version"] = "Version Unknown"

value_counts_after = laptops.loc[laptops["os_version"].isnull(), "os"].value_counts()

# Assignment
# Explore
print(laptops["weight"].unique())
# Replace 
laptops["weight"] = laptops["weight"].str.replace('kgs', '')
                                     .str.replace('kg', '')
print(laptops["weight"].unique())
# Float convert
laptops["weight"] = laptops["weight"].astype(float)
# Rename column
laptops.rename({"weight": "weight_kg"}, axis = 1, inplace =True)
# Save as new csv
laptops.to_csv("laptops_cleaned.csv", index = False)