##### NumPy
- Documentation: https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation
- The NumPy library takes advantage of a processor feature called Single Instruction Multiple Data (SIMD) to process data faster. SIMD allows a processor to perform the same operation, on multiple data points, in a single processor cycle
- The concept of replacing for loops with operations applied to multiple data points at once is called *vectorization* and ndarrays make vectorization possible.
- Tuples are very similar to Python lists, but can't be modified.
- Functions act as stand alone segments of code that usually take an input, perform some processing, and return some output
- In contrast, methods are special functions that belong to a specific type of object.
- In NumPy, sometimes there are operations that are implemented as both methods and functions:
```
np.min(trip_mph)
trip_mph.min()
```
- To remember the right terminology, anything that starts with np (e.g. np.mean()) is a function and anything expressed with an object (or variable) name first (e.g. trip_mph.mean()) is a method. When both exist, it's up to you to decide which to use, but it's much more common to use the method approach.
- Read file into NumPy ndarrays: `np.genfromtxt(filepath, delimiter=None, skip_header=0)`
- NumPy ndarrays can contain only one datatype.
- ndarray.dtype: see the internal datatype that has been used. (eg float64)
- NaN: similar to Python's None constant

##### Boolean indexing with NumPy
- Operation between a ndarray and a single value results in a new ndarray: `print(np.array([2,4,6,8]) + 10)   #[12 14 16 18]`
- To index using our new boolean array, we simply insert it in the square brackets
- When working with 2D ndarrays,  boolean array must have the same length as the dimension you're indexing
- Modify values within an ndarray: 
```
np.genfromtxt(filename, delimiter=None)
ndarray[location_of_values] = new_value`
```
- Shortcut in one line
```
array[array[:, column_for_comparison] == value_for_comparison, column_for_assignment] = new_value
```

##### Introduction to Pandas
- Documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.max.html
- Advantages:
    - Axis values can have string labels, not just numeric ones.
    - Dataframes can contain columns with multiple data types: including integer, float, and string.
- DataFrame.dtypes attribute (similar to NumPy's ndarray.dtype attribute): return information about the types of each column
- object type: used for columns that have data that doesn't fit into any other dtype
- DataFrame.info() method: an overview of all the dtypes used in dataframe, along with its shape and other information
- Select data using labels: `df.loc[row_label, column_label]`
- Series is the pandas type for one-dimensional objects
-  Instead of `df.loc[:,["col1","col2"]]`, you can also use `df[["col1", "col2"]]` to select specific columns.
- Slice columns with labels: `f500_selection.loc[:, "rank":"profits"]  #Inclusive!`
- Similar for selecting row: `
```
single_row = f500_selection.loc["Sinopec Group"]
list_rows = f500_selection.loc[["Toyota Motor", "Walmart"]]
slice_rows = f500_selection["State Grid":"Toyota Motor"]
```
- single_col = df["D"]
single_row = df.loc["v"]
mult_cols = df[["A", "C", "D"]]
mult_rows = df.loc[["v", "w", "x"]]
- Series.value_counts() method: displays each unique non-null value in a column and their counts in order.
- Select items from series:
    - Single item: s.loc["item"] or s["item"]
    - List of items: s.loc[["item1", "item7"]] or s[["item1", "item7"]]
    - Slice of items: s.loc["item2":"item4"] or s["item2":"item4"]
- [Summary of label selection.jpg]

##### Pandas fundamental
- Series.describe(): tells us how many non-null values are contained in the series, along with the mean, minimum, maximum, and other statistics we'll learn about later in this path.
    - Documentation: https://app.dataquest.io/m/381/exploring-data-with-pandas%3A-fundamentals/7/dataframe-describe-method
    - By default return statistics for only numeric columns
    - Getting just the object columns, include=['O'] parameter: `print(f500.describe(include=['O']))`
    - Series.describe() returns a series object; the DataFrame.describe() returns a dataframe object.
- Pandas dataframe methods also accept the strings "index" and "columns" for the axis parameter:
    - Calculates result for each column: DataFrame.method(axis="index")
    - Calculates results for each row: DataFrame.method(axis="column")
- Boolean indexing: didn't use loc[]  because boolean arrays use the same shortcut as slices to select along the index axis.
- Where you try and assign a NaN value to an integer column, pandas will silently convert that column to a float dtype: https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html

##### Pandas Intermediate
- Pandas.read_csv(): https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
-
``` 
f500 = pd.read_csv("f500.csv", index_col=0)  #specify the first column as the row labels
f500.index.name = None  #don't inlcude the name for the index axis
```
- .loc and .iloc:
    - loc: label based selection
    - iloc: integer position based selection
- Positional slicing:
```
second_to_sixth_rows = f500[1:5]  
```
    - With loc[], the ending slice is included.
    - With iloc[], the ending slice is not included.
- [Index slicing iloc.jpg]
- Series.isnull() method and Series.notnull() method: select either rows that contain null (or NaN) values or rows that do not contain null values for a certain column: `rev_is_null = f500["revenue_change"].isnull()`
- Concatenate series/dataframes by mutual index labels: `food["alt_name"] = alt_name`
- Combining boolean indexing: 
```
big_rev_neg_profit = f500[(f500["revenues"] > 100000) & (f500["profits"] < 0)]
# our boolean operation will fail without parentheses
```
- Pandas `~a` (Python equivalent `not a`): True if a is False
- DataFrame.sort_values(): 
    - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
    - `sorted_rows = selected_rows.sort_values("employees", ascending=False)`
- Aggregation: apply a statistical operation to groups of our data. 



