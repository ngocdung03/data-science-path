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
- Series.map()
- Series.apply()
- Only use the Series.apply() method to apply a function with additional arguments element-wise. Series.map() will return errors
- DataFrame.applymap(): apply same function to multiple columns
- We used the df.apply() method to transform multiple columns. This is only possible because the parameter (eg. pd.value_counts) function operates on a series. If we tried to use the df.apply() method to apply a function that works element-wise to multiple columns, we'd get an error:
```
def label(element):
    if element > 1:
        return 'High'
    else:
        return 'Low'
happiness2015[factors].apply(label)  #Error
```
- In general, we should only use the apply() method when a vectorized function does not exist
- pd.melt() function: 
```
pd.melt(df, id_vars=[col1, col2], value_vars=[col3,col4])
#col1, col2: names of the columns that should remain the same in the result
#col3, col4: names of the columns that should be changed to rows in the result
```
    - Data is in a format that makes it easier to aggregate. We refer to data in this format as *tidy* data.
- df.pivot(): "un-melt" the data
    -  !!Different from df.pivot_table() method 
