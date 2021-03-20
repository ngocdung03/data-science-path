##### Data aggregration
- Aggregations: applying a statistical operation to groups of our data
- DataFrame.groupby('col'): performs the "split-apply-combine" process on a dataframe
- GroupBy.get_group(): select data for a certain group.
- GroupBy.groups attribute: get more information about the GroupBy object
- Pandas common aggregration methods: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
- GroupBy.agg() peforms multiple aggregration at once: `GroupBy.agg([func1, func2, func3])  #without ()`
- df.pivot_table(): perform the same aggregration
```python
happiness2015.pivot_table(values='Happiness Score', index='Region', aggfunc=np.mean)
# may omit aggfunc because mean is the default aggregation function of df.pivot_table()
```

##### Combining data with Pandas
- pd.concat([df1, df2]):
    - Stacked: axis=0 (default)
    - Side by side: axis=1
    - ignore_index: reset the index
    - By default, the function will keep ALL of the data, no matter if missing values are created.
- pd.merge(): execute high performance database-style joins: `pd.merge(left=df1, right=df2, on='Col_Name')     #on=col used as the key`
    - much more common to use inner and left joins for database-style joins
    - Inner join by index: `pd.merge(left=four_2015, right=three_2016, left_index=True, right_index=True, suffixes=('_2015','_2016'))`
- [pd.concat-vs-pd.merge.jpg] 

##### Transforming data with pandas
