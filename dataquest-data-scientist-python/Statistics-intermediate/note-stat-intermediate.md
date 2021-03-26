##### The mean 
- Numpy randint() function: Return random integers from low (inclusive) to high (exclusive). https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randint.html
- When a statistic is on average equal to the parameter it estimates, we call that statistic an unbiased estimator. This also holds true if we sample with replacement.

##### The Weighted Mean and the Median
- Weighted mean by numpy.average(): `numpy.average(a=array_like, weights=array_like)`
- Median for *open-ended distribution*
- Because the median is so resistant to changes in the data, it's classified as a resistant or robust statistic -  ideal for finding reasonable averages for distributions containing outliers
- Computing the mean involves meaningful arithmetical operations, so it's not theoretically sound to use the mean for ordinal variables 
    - Median is the better alternative
    - This doesn't fully apply, however, to even-numbered distributions, where we need to take the mean of the middle two values to find the median. This poses some theoretical problems, and we'll see in the next mission that the mode might be a better choice in this case as a measure of average.
    - Although it can be argued that it's theoretically unsound to compute the mean for ordinal variables, in the last exercise we found the mean more informative and representative than the median. The truth is that in practice many people get past the theoretical hurdles and use the mean nonetheless because in many cases it's much richer in information than the median.
    ```
    mean = houses['Overall Cond'].mean()
    median = houses['Overall Cond'].median()
    houses['Overall Cond'].plot.hist()
    more_representative = 'mean'
    ```
- Unlike the median, the mean is sensitive to small changes in the data, and this property is what makes it more useful in cases such as evaluate satisfaction based on changes meant to improve internet speed.
- It should be clear by now that whether we should use the mean for ordinal data is contentious. In practice, you should be flexible and make your choice on a case by case basis. Also, you are not constrained to choose one metric or the other — you can choose both the mean and median to describe a distribution.