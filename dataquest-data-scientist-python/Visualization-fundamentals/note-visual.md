##### Line graphs and time series
- A quirk of Matplotlib is that we generally import the pyplot submodule instead of the whole module: `import matplotlib.pyplot as plt`
- Add the data to the plt.plot(x_list, y_list) function: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html. X_list has same length with y_list.
- Display the plot using the plt.show() function: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html. Generates a line graph by default.
- Remove scientific notation, format tick labels: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ticklabel_format.html
```
plt.plot(month_number, new_cases)
plt.ticklabel_format(axis='y', style='plain')
plt.show()
```
-  We call a series of data points that is listed in time order a time series
- [Three types of growth.jpg]
- Drawing 2 line graphs with the same axes: 
```python
plt.plot(france['Date_reported'], france['Cumulative_cases'],
         label='France')
plt.plot(uk['Date_reported'], uk['Cumulative_cases'],
         label='The UK')
plt.legend()
plt.show()
#When we use plt.plot() the first time, Matplotlib creates a line graph. When we use plt.plot() again, Matplotlib creates another line graph that shares the same x- and y-axis as the first graph. If we want Matplotlib to draw the second line graph separately, we need to close the first graph with the plt.show() function.
```
##### Scatter plots and correlations
- plt.scatter(x,y)
- Series.corr() method: calculate the Pearson's r between two variables: `bike_sharing['temp'].corr(bike_sharing['cnt'])`
- DataFrame.corr() returns a DataFrame (correlation table): `bike_sharing.corr()[['cnt', 'casual', 'registered']]`

##### Bar plots, histograms, and distributions
- plt.bar(x, y)
- plt.barh(): horizontal bar plot
- Bar plots work well when generating the frequency tables for categorical columns. For numerical columns, we need to use a histogram.
- Group the frequency table into ten equal intervals by using the bins=10 argument: `bike_sharing['cnt'].value_counts(bins=10)`
- plt.hist(x): histogram

##### Pandas visualization and grid charts
- About the dataset: https://archive.ics.uci.edu/ml/datasets/Behavior+of+the+urban+traffic+of+the+city+of+Sao+Paulo+in+Brazil
- Series.plot.hist(): a quicker way to generate histogram: `traffic['Slowness in traffic (%)'].plot.hist()`
- Series.plot.line(): generates a line plot.
- Calculate sums of columnsz: DataFrame.sum()
- A grid chart is a collection of similar graphs that usually share the same x- and y-axis range. The main purpose of a grid chart is to ease comparison.
    - First creating the larger figure where we will plot all the graphs: plt.figure()
    - plt.subplot(nrows, ncols, index)
    - When we want to add another plot, we add another plt.subplot() function
    - it only takes positional arguments. If we use keyword arguments, we'll get an error — plt.subplot(nrows=3, ncols=2, index=1)
    - plt.figure(figsize=(width, height))
    
##### Relational plots and multiple variables
- Seaborn enables us to easily show more than two variables on a graph
- Call the sns.relplot() function to generate the plot.
    - We pass in the housing DataFrame to the data parameter.
    - We pass in the column names as strings to parameters x and y.
    - By default, the sns.relplot() function generates a scatter plot.
    - To switch to Seaborn defaults,  call the sns.set_theme() before generating the plot
- Matplotlib color palette: https://matplotlib.org/stable/tutorials/colors/colormaps.html
- Change marker shape: https://matplotlib.org/3.3.3/api/markers_api.html
- The graph we built is essentially a scatter plot. However, because it shows the relationships between so many variables, we call it a relational plot.
