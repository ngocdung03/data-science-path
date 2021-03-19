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


