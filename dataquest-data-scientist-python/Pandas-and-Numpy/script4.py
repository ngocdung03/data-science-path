## Pandas Intermediate
import pandas as pd
# read the data set into a pandas dataframe
f500 = pd.read_csv("f500.csv", index_col=0)
f500.index.name = None

# replace 0 values in the "previous_rank" column with NaN
f500.loc[f500["previous_rank"] == 0, "previous_rank"] = np.nan

# Select columns
f500_selection = f500.loc[:, ["rank", "revenues", "revenue_change"]].head(5)

# f500 = pd.read_csv("f500.csv", index_col=0)
# f500.index.name = None
# Read dataframw without index_col paramemter
f500 = pd.read_csv("f500.csv")
f500.loc[f500["previous_rank"] == 0, "previous_rank"] = np.nan

# .iloc 
fifth_row = f500.iloc[4]
company_value = f500.loc[0, "company"]

first_three_rows = f500[:3]
first_seventh_row_slice = f500.iloc[[0, 6], 0:5]

# Boolean indexing
prev_isnull = f500["previous_rank"].isnull()
null_previous_rank = f500[prev_isnull][["company", "rank", "previous_rank"]]

null_previous_rank = f500[f500["previous_rank"].isnull()]
top5_null_prev_rank = null_previous_rank.iloc[0:5]

# Concatenate series/dataframes by mutual index labels
previously_ranked = f500[f500["previous_rank"].notnull()]
rank_change = previously_ranked["previous_rank"] - previously_ranked["rank"]
# Concatenate by mutual row indices
f500["rank_change"] = rank_change

# Combining boolean operators
large_revenue = f500["revenues"] > 100000
negative_profits = f500["profits"] < 0
combined = large_revenue & negative_profits
big_rev_neg_profit = f500[combined]

brazil_venezuela = f500[(f500["country"] == "Brazil") | (f500["country"] == "Venezuela")]
tech_outside_usa = f500[(f500["sector"] == "Technology") & ~(f500["country"] == "USA")].head(5)

# DataFrame.sort_value()
japan_companies = f500[f500["country"] == "Japan"]
top_japanese_employer = japan_companies.sort_values("employees", ascending = False).iloc[0]["company"]

# Aggregration: dictionary of the top employer in each country
top_employer_by_country = {}
countries = f500["country"].unique()

for c in countries:
    selected_rows = f500[f500["country"] == c]
    sorted_rows = selected_rows.sort_values("employees", ascending = False)
    top_employer = sorted_rows.iloc[0]["company"]
    top_employer_by_country[c] = top_employer

f500["roa"] = f500["profits"]/f500["assets"]
sector = f500["sector"].unique()
top_roa_by_sector = {}
for s in sector:
    selected_sector = f500[f500["sector"] == s]
    sort_values = selected_sector.sort_values("roa", ascending = False)
    company_name = sort_values.iloc[0]["company"]
    top_roa_by_sector[s] = company_name