#### Machine learning fundementals 
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
- Workflow:
    - select relevant features to use for predicting the target column.
    - use grid search to find the optimal hyperparameter value for the selected features.
    - evaluate the model's accuracy and repeat the process.

##### Cross-validation
- Holdout validation technique
- When splitting the data set, don't forget to set a copy of it using .copy() to ensure you don't get any unexpected results later on:
    - SettingWithCopy warning: https://www.dataquest.io/blog/settingwithcopywarning/
- Holdout validation is better than train/test validation because the model isn't repeatedly biased towards a specific subset of the data, both models that are trained only use half the available data. 
- K-fold cross validation, on the other hand, takes advantage of a larger proportion of the data during training while still rotating through different subsets of the data to avoid the issues of train/test validation.
- While the average RMSE value was approximately 129, the RMSE values ranged from 102 to 164+. This large amount of variability between the RMSE values means that we're either using a poor model or a poor evaluation criteria (or a bit of both!).
- KFold():
```py
from sklearn.model_selection import KFold
kf = KFold(n_splits, shuffle=False, random_state=None)
```
- cross_val_score():
    - estimator is a sklearn model that implements the fit method (e.g. instance of KNeighborsRegressor)
    - scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
    - cv can be: an instance of the KFold class or an integer representing the number of folds.
```py
from sklearn.model_selection import cross_val_score
cross_val_score(estimator, X, Y, scoring=None, cv=None)
```
- Through lots of trial and error, data scientists have converged on 10 as the standard k value.
- Through CV: the standard deviation of the RMSE values can be a proxy for a model's **variance** while the average RMSE is a proxy for a model's **bias**.
- A mathematical model is usually an equation that can exist without the original data, which isn't true with k-nearest neighbors. Bias-variance tradeoff is importanct when working with mathematical models in particular.

#### Calculus for Machine Learning
##### Understanding linear and nonlinear functions
-  A model, represented by mathematical function, approximates the underlying function that describes how the features are related to the target attribute -> making predictions is computationally cheap.
- Instantaneous rate of change: slope at a particular point

##### Understanding limits
- Calculatating limits in SymPy library:
```py
import sympy
x,y = sympy.symbols('x y') #declare the variables we want to be treated as symbols
limit_one = sympy.limit((-x2**2+3*x2-1+1)/(x2-3), x2, 2.9)
```
##### Find extreme points
- The process of finding a function's derivative is known as differentiation.
- [Differentiation.pdf]
- Critical points: can be extreme values or not, or relative minimum/maximum
- Sign chart for identifying direction of slope: http://www.rasmus.is/uk/t/F/Su53k02.htm
- Eg: f(x) = x^3 - x^2
    - Derivative: 0, 2/3
    - Through sign chart: relative minimum: 2/3, relative maximum: 0

##### Linear algebra for machine learning
- Systems of linear equations can be solved using linear algebra using a variation of arithmetic elimination called Gaussian elimination.
- Matrix can represent a linear system compactly. *Algebra* is a set of rules for manipulating that representation. We need to rerrange each functions into the *general form*.
```py
#Because we'll be performing operations using the values in this matrix, we need to set the type to float32 to preserve precision.
matrix_one = np.asarray([
    [30, -1, -1000],
    [50, -1, -100]  
], dtype=np.float32)
```
- Augmented matrix
- To preserve the relationships in the linear system, we can only use row operations: swap rows, multiply by a nonzero constant, add rows. Note that you can't multiply or divide by other rows.
```py
# Swap the second row (at index value 1) with the first row (at index value 0).
matrix = matrix[[1,0]]
matrix[1] = 0.5*matrix[2] + matrix[1] + matrix[3]
```
- To find the solutions of a matrix:
    - First, rearrange the matrix into *echelon form* - the values on the diagonal locations are all equal to 1 and the values below the diagonal are all equal to 0. [Echelon rearrangement.jpeg]
    - Second, rearrange the matrix into *row reduced echelon form*
- Generally, the word vector refers to the column vector
- Can visualize vectors in matplotlib using the pyplot.quiver()
    - We also need to set the angles and scale_units parameters to xy and the scale parameter to 1. Setting angles to 'xy' lets matplotlib know we want the angle of the vector to be between the points we specified. The scale_units and scale parameters lets us specify custom scaling parameters for the vectors. 
    ```py
    plt.quiver(0, 0, 1, 2, angles='xy', scale_units='xy', scale=1)
    ```
- Vector in numpy
```py
vector_one = np.asarray([
    [1],
    [2],
    [1]
], dtype=np.float32)
```
- Dot product: one of the two vectors need to be represented as a row vector while the other a column vector
```py
dot_product = np.dot(vector_one[:,0], vector_two) 
```
- Being able to scale vectors using scalar multiplication then adding or subtracting these scaled vectors is known as *linear combination*. 
- To multiply a matrix by a vector, the number of columns in the matrix needs to match the number of rows in the vector.
- Transpose matrix:
    - Distribute for sum operation: (A+B)^T = A^T + B^T
    - Distribute for multiplication: (AB)^T = B^T*A^T  #note the order
    - np.transpose()
