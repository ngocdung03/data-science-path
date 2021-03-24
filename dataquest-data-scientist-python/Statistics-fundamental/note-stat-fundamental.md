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

