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
- pd.merge(): execute high performance database-style joins: `pd.merge(left=df1, right=df2, on='Col_Name')     #on=col used as the key. left_on or right_on`
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

##### Working with strings in Pandas
- World_dev.csv: https://www.kaggle.com/worldbank/world-development-indicators/version/2
- We should use built-in vectorized methods (if they exist) instead of the Series.apply() method for performance reasons.
     - Instead of string.split(), we could've used the vectorized equivalent method - Series.str.split()
     - [Common equivalent vectorized string method.jpb]: https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html
     - Access these vectorized string methods by adding a str between the Series name and method name: `Series.str.method_name()`
     - `str` attribute indicates that each object in the Series should be treated as a string, without us having to explicitly change the type to a string
     - Chaining those methods: `merged['CurrencyUnit'].str.upper().str.split()`
- Regex aka regular expression: https://docs.python.org/3.4/library/re.html
    - Series.str.contains() search for pattern in the regex.
    - Character could be a range of numbers: `pattern = r"[0-9]" or r"[1-6]"`
    - Character could be a range of letters: `pattern = r"[a-z]" or r"[A-Z]"`
    - curly brackets {} indicate the number of times the pattern repeats: `pattern = r"[1-6][a-z][a-z]" = r"[1-6][a-z]{2}"`
    - parentheses () indicate that only the character pattern matched should be extracted and returned in a series. We call this a *capturing group*: Series.str.extract("(pattern)")
        - If the capturing group doesn't exist in a row (or there is no match) the value in that row is set to NaN instead
        - return the results as a dataframe by changing the expand parameter to True.
    -  Series.str.extract() will only extract the first match of the pattern. If we wanted to extract all of the matches, we can use the Series.str.extractall() method
    - Named capturing group: refer to the group by the specified name instead of just a number `?P<Column_Name>...`:
    ```py
    #name the capturing group Years:
    pattern = r"(?P<Years>[1-2][0-9]{3})"
    merged['SpecialNotes'].str.extractall(pattern)
    ```
    - Deal with "2018/19" pattern:
    ```py
    pattern = r"(?P<First_Year>[1-2][0-9]{3})(/)?(?P<Second_Year>[0-9]{2})?"
    years = merged['IESurvey'].str.extractall(pattern)
    ```
        - We added a question mark, ?, after each of the two new groups to indicate that a match for those groups is optional. This allows us to extract years listed in the yyyy format AND the yyyy/yy format at once.
- If part of the regex isn't grouped using parantheses, (), it won't be extracted.
- When we add a string to a column using the plus sign, +, pandas will add that string to every value in the column. Note that the strings will be added together without any spaces.

##### Missing and duplicate data
- Missing or duplicate data is introduced as: 
    - perform cleaning and transformation tasks such as: Combining data, Reindexing data, Reshaping data
    - User input error
    - Data storage or conversion issues
    - Purposely indicate that data is unavailable.
- pandas will not automatically identify values such as n/a, -, or -- as NaN or None, but they may also indicate data is missing.
- how to use the pd.read_csv() function to read those values in as NaN: https://stackoverflow.com/questions/40011531/in-pandas-when-using-read-csv-how-to-assign-a-nan-to-a-value-thats-not-the#answer-40011736
- Once we ensure that all missing values were read in correctly, we can use the Series.isnull() method to identify rows with missing values:
```py
missing = happiness2015['Happiness Score'].isnull()
happiness2015[missing]
```
- when working with bigger data sets, it's easier to get a summary of the missing values: `happiness2015.isnull().sum()`
- Determine if the data is really missing or if it's the result of some kind of error:
    - Check for errors in data cleaning/transformation.
    - Use data from additional sources to fill missing values.
    - Drop row/column.
    - Fill missing values with reasonable estimates computed from the available data (Imputation): A constant value, mean of the column, median of the column, or mode of the column.
- DataFrame.duplicated(): check for duplicate values: Check if there are rows with both same country and year: `dups = combined.duplicated(['COUNTRY', 'YEAR'])`
- df.drop_duplicates() drop the duplicate rows. It will only keep the first duplicate row. To keep the last duplicate row, set the keep parameter to 'last'. Sometimes, this will mean sorting the dataframe before dropping the duplicate rows.
- When deciding if you should drop a row or column, it's better to also ask the following questions:
    1. Is the missing data needed to accomplish our end goal?
    2. How will removing or replacing the missing values affect our analysis?
        - *What percentage of the data is missing?* the lower the percentage of missing values, the less likely dropping them will significantly impact the analysis.
        - *Will dropping missing values cause us to lose valuable information in other columns?* if the data is missing in almost every column, dropping them won't cause us to lose valuable information in other columns. 
        - Can we identify any patterns in the missing data?
- DataFrame.dropna(): delete rows with missing values, axis=1: delete columns
    - `thresh` parameter to only drop columns if they contain below a certain number of non-null values.
- To confirm the number of values that are NOT missing, we can use the DataFrame.notnull()method: `combined.notnull().sum().sort_values`
-  Data cleaning workflow:
    - Set a goal for the project.
    - Researched and tried to understand the data.
    - Determined what data was needed to complete     - our analysis.
    - Added columns.
    - Cleaned specific data types.
    - Combined data sets.
    - Removed duplicate values.
    - Handled the missing values.






