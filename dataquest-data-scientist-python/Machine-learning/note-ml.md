##### Introduction to k-nearest neighbor

##### Evaluating model performance
- Error metric quantifies how inaccurate our predictions were compared to the actual values
- Mean absolute error
- Mean squared error: For many prediction tasks, we want to penalize predicted values that are further away from the actual value far more than those closer to the actual value.
- While comparing MSE values helps us identify which model performs better on a relative basis, it doesn't help us understand if the performance is good enough in general. This is because the units of the MSE metric are squared.
- *Root mean squared error*
- To better understand a specific model, we can compare multiple error metrics for the same model (MAE and RMSE?)
```py
# Compare mae, rmse of one model
errors_one = pd.Series([5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10])
errors_two = pd.Series([5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 1000])

#mae_one = np.mean(errors_one.apply(lambda x: abs(x - errors_one.mean())))   # why wrong if chain .mean()?
mae_one = errors_one.mean()   #7.5
rmse_one = np.sqrt((errors_one**2).mean()) #7.9

mae_two = errors_two.mean()  #62.5
rmse_two = np.sqrt((errors_two**2).mean())  #235.8
```

While the MAE (7.5) to RMSE (7.9056941504209481) ratio was about 1:1 for the first list of errors, the MAE (62.5) to RMSE (235.82302686548658) ratio was closer to 1:4 for the second list of errors. In general, we should expect that the MAE value be much less than the RMSE value. When we're working with larger data sets, we can't inspect each value to understand if there's one or some outliers or if all of the errors are systematically higher. Looking at the ratio of MAE to RMSE can help us understand if there are large but infrequent errors.
- MAE and RMSE — Which Metric is Better? https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d#.lyc8od1ix

##### Multivariate k-nearest neighbors
- When selecting more attributes to use in the model, we need to watch out for columns that don't work well with the distance equation:
    - non-numerical values
    - missing values
    - non-ordinal values (e.g. latitude or longitude)
- To prevent any single column from having too much of an impact on the distance, we can normalize all of the columns to have a mean of 0 and a standard deviation of 1.
- distance.euclidean() function:
```py
from scipy.spatial import distance
first_listing = [-0.596544, -0.439151]
second_listing = [-0.596544, 0.412923]
dist = distance.euclidean(first_listing, second_listing)
```
- Calculate Euclidean distance: distance.euclidean() function from scipy.spatial, which takes in 2 vectors as the parameters and calculates the Euclidean distance between them.  
- KNeighborsRegressor(): takes a matrix-like object and a list-like object
- Intead of calculating MSE and RMSE using pandas arithmetic operators, we can instead use the sklearn.metrics.mean_squared_error function(). Once you become familiar with the different machine learning concepts, unifying your workflow using scikit-learn helps save you a lot of time and avoid mistakes. 

##### Hyperparameter Optimization
- In other words, we're impacting how the model performs without trying to change the data that's used.
- Values that affect the behavior and performance of a model that are unrelated to the data that's used are referred to as hyperparameters. 