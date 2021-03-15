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
