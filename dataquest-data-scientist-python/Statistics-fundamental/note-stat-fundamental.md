##### Sampling
- Using statistical techniques, we can organize, summarize, and visualize large amounts of data to find patterns that otherwise would remain hidden.
- sampling error = parameter - statistic
-  Series.sample() uses a pseudorandom number generator under the hood. A pseudorandom number generator uses an initial value to generate a sequence of numbers that has properties similar to those of a sequence that is truly random
- To ensure we end up with a sample that has observations for all the categories of interest, we can organize our data set into different groups, and then do simple random sampling for every group. This sampling method is called stratified sampling, and each stratified group is also known as a stratum.
- If the number of total points is influenced by the number of games played:
    - Use stratified sampling while being mindful of the proportions in the population. We can stratify our data set by the number of games played, and then sample randomly from each stratum a proportional number of observations.
    - Category of range intead of unique numeric values: `wnba['Games Played'].value_counts(bins = 3, normalize = True) * 100)`
- Few guidelines for choosing good strata:
    1. Minimize the variability within each stratum.
    2. Maximize the variability between strata.
    3. The stratification criterion should be strongly correlated with the property you're trying to measure.
- Sometimes data is scattered across different locations. One way is to list all the data sources you can find, and then randomly pick only a few of them to collect data from. Then you can sample individually each of the sources you've randomly picked. This sampling method is called **cluster sampling**.
    - Eg: Sample 4 clusters randomly: `pd.Series(wnba['Team'].unique()).sample(4, random_state = 0))`

##### Variables in statistics
- The system of rules that define how each variable is measured is called scale of measurement or, less often, level of measurement.
- Quantitative variables: ordinal scale,  interval scale, or on a ratio scale.
- What sets apart ratio scales from interval scales: On a ratio scale, the zero point means no quantity. On an interval scale, however, the zero point doesn't indicate the absence of a quantity. It actually indicates the presence of a quantity.
- In practice, variables measured on an interval scale are relatively rare
- If there's no possible intermediate value between any two adjacent values of a variable, we call that variable *discrete*.
- If there's an infinity of values between any two values of a variable, we call that variable *continuous*.
- boundaries of an interval are sometimes called real limits. The lower boundary of the interval is called lower real limit, and the upper boundary is called upper real limit.

##### Frequency distribution
- In case the variable in concern have direction, sort the table by index using the Series.sort_index(): `wnba['Height'].value_counts().sort_index(ascending=False)`
- Sort by ordinal variable: 
```py
wnba['PTS_ordinal_scale'].value_counts()[['very few points', 'few points', 'many points', 'a lot of points']]
wnba['PTS_ordinal_scale'].value_counts().iloc[[3, 1, 2, 0]]
```
- Seriex.value_count():
    - param normalize=True: return proportion
- A percentile rank of a value x  in a frequency distribution is given by the percentage of values that are equal or less than x.
```py
from scipy.stats import percentileofscore
print(percentileofscore(a = wnba['Age'], score = 23, kind = 'weak'))
# kind = 'weak' to indicate that we want to find the percentage of values thar are equal to or less than the value we specify in the score parameter.
```
- Series.describe(): returns by default the 25th, the 50th, and the 75th percentiles
    - Return others than default percentiles: `wnba['Age'].describe(percentiles = [.1, .15, .33, .5, .592, .85, .9]).iloc[3:])`
- *Grouped frequency distribution table*, class interval: If needing ten equal intervals, specify bins = 10: `wnba['Weight'].value_counts(bins = 10).sort_index()`
- As a rule of thumb, 10 is a good number of class intervals to choose because it offers a good balance between information and comprehensibility.
- Manually define the intervals:
```py
intervals = pd.interval_range(start = 0, end = 600, freq = 100)
gr_freq_table = pd.Series([0,0,0,0,0,0], index = intervals)
for value in wnba['PTS']:
    for interval in intervals:
        if value in interval:
            gr_freq_table.loc[interval] += 1
            break
```

##### Visualizing frequency distribution
- For variables measured on a nominal or an ordinal scale it's common to use a bar plot to visualize their distribution. 
- Pie chart: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html
```py
wnba['Pos'].value_counts().plot.pie(figsize = (6,6), autopct = '%.1f%%')
# the string '%.1f%%' to have percentages displayed with a precision of one decimal place. https://docs.python.org/3/library/string.html#format-specification-mini-language

#[Percentages-display-pie-chart.jpg]
```
- For visualize distribution: histogram
    - arange() function from numpy: generate the intervals (start, end, step)
```py
from numpy import arange
wnba['PTS'].plot.hist(grid = True, xticks = arange(2,585,58.2), rot = 30)
```
```py
wnba['PTS'].plot.hist(range = (1,600), bins = 3)
```