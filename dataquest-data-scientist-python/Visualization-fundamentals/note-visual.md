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
- 
