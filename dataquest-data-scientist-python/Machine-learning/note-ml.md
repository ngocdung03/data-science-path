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
- To solve for vector x: Ax = b   (b is also vector)
    - Remember Ix = x (I is identity matrix)
    - Transform and convert A into the identity matrix  AA^-1 = I
    - Multiply A^-1 on both sides
    - [Inverse matrix.jpeg]
    - np.identity(): creating identity matrix
    - np.linalg.inv(): find inverse matrix
    - Because IA = A, similar to solve for matrix (?)
- [Determinant of higher dim square matrix.jpeg]
- [Inverse 3x3 matrix.jpeg]
- 2 ways to solve for Ax = b
    - Gaussian elimination
    - Multiplying both sides to inverse of A
- Not all systems of equations have a solution and these systems are inconsistent. If the determinant of the matrix is equal to 0, the matrix is *singular*, or contains no inverse.
    - the solution set for a linear system doesn't exist
    - the solution set for a linear system isn't just a single vector
    - b is equal to 0
- linear systems where the constants vector (b) doesn't contain all zeroes: *nonhomogenous systems*.
- On the other hand, when the constant vector is equal to the zero vector: *homogenous system* ->  *always have a trivial solution* - the zero vector.
    - We're interested in determining if infinitely many solutions exist or not using Gaussian elimination. 
    - If after echelon reduction, the last row contain 0 = a, there is no solution.
    - On the other hand, when the solution is a solution space, it's common to rewrite it into *parametric vector form*. Eg: x = x3.[4/3, 0, 1]
- For a *nonsquare*, nonhomogenous systems, there are 2 possible solutions: no solution, infinitely many solutions (solution space). 

##### Linear regression for machine learning
- K-nearest neighbors is known as an *instance-based learning* algorithm because it relies completely on previous instances to make predictions. It doesn't try to understand or capture the relationship between the feature columns and the target column. -> doesn't scale well to medium and larger datasets.
- Parametric machine learning approaches: work by making **assumptions** about the relationship between the features and the target column
- To find the optimal parameters for a linear regression model, we want to optimize the model's residual sum of squares (or RSS).
- RSS seems very similar to the calculation for MSE (mean squared error).
- Code for select integer and float columns in dataset:
```py
train.select_dtypes(include=['int', 'float'])
```
- AmesHousing.txt data documentation: https://s3.amazonaws.com/dq-content/307/data_description.txt
- Workflow...
- The problem of choosing a set of values that minimize or maximize another function is known as an *optimization problem*.
- Gradient descent: iteratively trying different parameter values until the model with the lowest mean squared error is found.
    - select initial values for the parameter: a1
    - repeat until convergence (usually implemented with a max number of iterations):
        - calculate the error (MSE) of model that uses current parameter value: MSE(a1)=1n∑(^yi−yi)^2
        - calculate the derivative of the error (MSE) at the current parameter value: (d/da1)MSE(a1)
        - update the parameter value by subtracting the derivative times a constant (α, called the learning rate): a1:=a1−α(d/da1)MSE(a1)
    - Selecting an appropriate initial parameter and learning rate will reduce the number of iterations required to converge, and is part of hyperparameter optimization.
- Cost function/ loss function
- [Gradient descent of MSE.jpeg]
- [Gradient descent of multiple parameter MSE.jpeg]
- Ordinary least squares OLS: rovides a clear formula (*closed form solution*) to directly calculate the optimal parameter values that minimize the cost function.
    - Optimal vector a: a = (X^T.X)^-1.X^T.y
    - scikit-learn uses OLS under the hood when you call fit() on a LinearRegression instance.
    - [Matrix form of LR model.jpeg]  - Note the first column with all 1
    - [Cost function OLS.jpeg]
    - Derivatives of the cost function: https://eli.thegreenplace.net/2015/the-normal-equation-and-matrix-calculus/
    - [From derivative of cost function to OLS.jpeg]
- OLS is commonly used when the number of elements in the dataset (and therefore the matrix that's inverted) is less than a few million elements. On larger datasets, gradient descent is used because it's much more flexible. For many practical problems, we can set a threshold accuracy value (or a set number of iterations) and use a "good enough" solution.
- Processing and transforming features
- Feature engineering: the process of processing and creating new features. 
- Some issues of non-missing features:
    - the column is not numerical (e.g. a zoning code represented using text)
    - the column is numerical but not ordinal (e.g. zip code values)
    - the column is numerical but isn't representative of the type of relationship with the target column (e.g. year values)
- Convert string column (or any other type) that contains no missing using pd.Series.astype().
    - We need to use the .cat accessor followed by the .codes property to actually access the underlying numerical representation of a column (instead of the string label): `train['Utilities'].cat.codes`
```py
train['Utilities'] = train['Utilities'].astype('category')
```
- Transform categorical vars into dummies: pandas.get_dummies()
- Approaches for missing values:
    - Remove rows containing missing values for specific columns
        - Pro: Rows containing missing values are removed, leaving only clean data for modeling
        - Con: Entire observations from the training set are removed, which can reduce overall prediction accuracy
    - Impute (or replace) missing values using a descriptive statistic from the column
        - Pro: Missing values are replaced with potentially similar estimates, preserving the rest of the observation in the model.
        - Con: Depending on the approach, we may be adding noisy data for the model to learn
        - We'll focus on columns that contain at least 1 missing value but less than 365 missing values (or 25% of the number of rows in the training set). Many people instead use a 50% cutoff 

 
