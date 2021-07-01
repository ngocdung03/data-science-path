## Multivariate k-nearest neighbor
#Data wrangling
import pandas as pd
dc_listings = pd.read_csv('dc_airbnb.csv')
print(dc_listings.head(1))

dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

# number of non-null values in each column.
dc_listings.info()

# Remove non-numerical and numerical but non-ordinal columns
# Since a host could have many living spaces and we don't have enough information to uniquely group living spaces to the hosts themselves, let's avoid using any columns that don't directly describe the living space or the listing itself.
dc_listings = dc_listings.drop(['room_type', 'city', 'state', 'latitude', 'longitude','zipcode', 'host_response_rate', 'host_acceptance_rate', 'host_listings_count'], axis=1)

# These columns have large number of missing values
dc_listings = dc_listings.drop(['cleaning_fee', 'security_deposit'], axis=1)

# Since the number of rows containing missing values for these columns is low, we can select and remove those rows without losing much information. 
dc_listings = dc_listings.dropna(axis=0)

dc_listings[dc_listings.isna()]

# Normalize all of the feature columns 
# ?why wrong: normalized_listings = dc_listings.apply(lambda x: (x - x.mean())/x.std())
normalized_listings = (dc_listings - dc_listings.mean())/(dc_listings.std())
normalized_listings['price'] = dc_listings['price']
print(normalized_listings.head(3))

# Calculate euclidean distance using function
from scipy.spatial import distance
# wrong first_fifth_distance = distance.euclidean(normalized_listings['accommodates'][[0,4]], normalized_listings['bathrooms'][[0,4]])
first_fifth_distance = distance.euclidean(normalized_listings.iloc[0][['accommodates', 'bathrooms']], normalized_listings.iloc[4][['accommodates', 'bathrooms']])
print(first_fifth_distance)

#Knn analysis
from sklearn.neighbors import KNeighborsRegressor

train_df = normalized_listings.iloc[0:2792]
test_df = normalized_listings.iloc[2792:]

knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')
knn.fit(train_df[['accommodates', 'bathrooms']], train_df['price'])
predictions = knn.predict(test_df[['accommodates', 'bathrooms']])

# Calculating MSE and RMSE by sklearn
from sklearn.metrics import mean_squared_error
two_features_mse = mean_squared_error(test_df['price'], predictions)
two_features_rmse = np.sqrt(two_features_mse)
print(two_features_mse)
print(two_features_rmse)

# Feed all the other vars as predictors
all_features = train_df.columns.tolist()
all_features.remove('price')

knn.fit(train_df[all_features], train_df['price'])
all_features_predictions = knn.predict(test_df[all_features])
all_features_mse = mean_squared_error(test_df['price'], all_features_predictions)
all_features_rmse = all_features_mse ** (1/2)
print(all_features_mse)
print(all_features_rmse)

## Hyperparameter optimization
import pandas as pd
train_df = pd.read_csv('dc_airbnb_train.csv')
test_df = pd.read_csv('dc_airbnb_test.csv')

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

hyper_params = list(range(1, 6))

mse_values = []
for k in hyper_params:
    knn = KNeighborsRegressor(n_neighbors = k, algorithm = 'brute')
    knn.fit(train_df[['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']], train_df['price'])
    predictions = knn.predict(test_df[['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']])
    mse_value = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse_value)
    
print(mse_values)

plt.scatter(hyper_params, mse_values)
plt.show()

# Find k such as mse is the lowest
two_features = ['accommodates', 'bathrooms']
three_features = ['accommodates', 'bathrooms', 'bedrooms']
hyper_params = [x for x in range(1,21)]
# Append the first model's MSE values to this list.
two_mse_values = list()
for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors = hp, algorithm ='brute')
    knn.fit(train_df[two_features], train_df['price'])
    predictions = knn.predict(test_df[two_features])
    mse_value = mean_squared_error(test_df['price'], predictions)
    two_mse_values.append(mse_value)
    
# Append the second model's MSE values to this list.
three_mse_values = list()
for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors = hp, algorithm ='brute')
    knn.fit(train_df[three_features], train_df['price'])
    predictions = knn.predict(test_df[three_features])
    mse_value = mean_squared_error(test_df['price'], predictions)
    three_mse_values.append(mse_value)
    
two_hyp_mse = dict()
two_lowest_mse = two_mse_values[0]
two_lowest_k = 1
for k, mse in enumerate(two_mse_values):
    if mse < two_lowest_mse:
        two_lowest_mse = mse 
        two_lowest_k = k+1   #error when += 1 and mse doesn't consistenly decrease
two_hyp_mse[two_lowest_k] = two_lowest_mse

three_hyp_mse = dict()
three_lowest_mse = three_mse_values[0]
three_lowest_k = 1
for k, mse in enumerate(three_mse_values):
    if mse < three_lowest_mse:
        three_lowest_mse = mse 
        three_lowest_k = k+1  #error when += 1 and mse doesn't consistenly decrease
three_hyp_mse[three_lowest_k] = three_lowest_mse

## Cross-validation
import numpy as np
import pandas as pd

dc_listings = pd.read_csv("dc_airbnb.csv")
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

# Shuffle the ordering of the rows in dc_listings
re_index = np.random.permutation(dc_listings.index)
dc_listings = dc_listings.reindex(re_index)

# To avoid SettingWithCopy warning, make sure to include .copy() whenever you perform operations on a dataframe.
split_one = dc_listings.iloc[0:1862].copy()
split_two = dc_listings.iloc[1862:].copy()

# Find the avg error when switch the training and test sets 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_one = split_one
test_one = split_two
train_two = split_two
test_two = split_one

knn_one = KNeighborsRegressor() #default algorithm auto, n_neighbors 5
knn_one.fit(train_one[['accommodates']], train_one['price'])   #df[col] return 1xn pd.Series while df[[col]] returns pd.DataFrame; ?error when dim of the 2nd argument is nx1? 
predicted_one = knn_one.predict(test_one[['accommodates']])
iteration_one_rmse = np.sqrt(mean_squared_error(test_one['price'], predicted_one))

knn_two = KNeighborsRegressor()
knn_two.fit(train_two[['accommodates']], train_two['price'])
predicted_two = knn_two.predict(test_two[['accommodates']])
iteration_two_rmse = np.sqrt(mean_squared_error(test_two['price'], predicted_two))

avg_rmse = np.mean([iteration_one_rmse, iteration_two_rmse])

#k-fold crosss-validation
# add a new column 'fold'
dc_listings.loc[dc_listings.index[0:745], 'fold'] = 1 # Select a subset of a dataframe's index by position (not label) like so: df.index[0:100]
dc_listings.loc[dc_listings.index[745:1490], 'fold'] = 2
dc_listings.loc[dc_listings.index[1490:2234], 'fold'] = 3
dc_listings.loc[dc_listings.index[2234:2978], 'fold'] = 4
dc_listings.loc[dc_listings.index[2978:3723], 'fold'] = 5
    
print(dc_listings['fold'].value_counts())
print("\n Num of missing values: ", dc_listings['fold'].isnull().sum())

# Train and evaluate on each fold
import numpy as np
fold_ids = [1,2,3,4,5]

def train_and_validate(df, folds):
    knn = KNeighborsRegressor()
    rmses = []
    for fold in range(1, folds+1):
        train = df[df['fold'] != fold]
        test = df[df['fold'] == fold]
        knn.fit(train[['accommodates']], train['price'])
        prediction = knn.predict(test[['accommodates']])
        mse = mean_squared_error(test['price'], prediction)
        rmse = mse**(1/2)
        rmses.append(rmse)
    avg_rmse = np.mean(rmses)
    return rmses, avg_rmse

rmses = train_and_validate(dc_listings, 5)[0]
avg_rmse = train_and_validate(dc_listings, 5)[1]

# Applying KFold and cross_val_score in sklearn
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=1)
knn = KNeighborsRegressor()
mses = cross_val_score(knn, 
                dc_listings[['accommodates']], 
                dc_listings['price'],
                scoring='neg_mean_squared_error',
                cv = kf)
avg_rmse = np.mean(np.sqrt(np.absolute(mses)))

## Calculus for Machine Learning
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 3, num=100)        # generate a sequence
y = -(x**2) + 3*x - 1

plt.plot(x, y)
plt.show()

## Linear algebra for machine learning
# Calculate inverse 2x2 matrix
matrix_a = np.asarray([
    [1.5, 3],
    [1, 4]
])

def matrix_inverse_two(mat):
    det = mat[0, 0]*mat[1, 1] - mat[0, 1]*mat[1, 0]   
    if det ==0:
        raise ValueError("The matrix isn't invertible")
    else:
        inv_mat =  np.asarray([
    [mat[1,1], -mat[0,1]],
    [-mat[1,0], mat[0,0]]  
], dtype=np.float32)
        return np.dot(1/det,inv_mat)         #error (if 1/det)*inv_mat?

inverse_a = matrix_inverse_two(matrix_a)
i_2 = np.dot(inverse_a, matrix_a)

## Linear regression for machine learning
import pandas as pd
data = pd.read_csv('AmesHousing.txt', delimiter='\t')
train = data[0:1460]   #selectinf first 1460 rows
test = data[1460:]
train.info()
target = 'SalePrice'

# Plot some vars vs. target var
import matplotlib.pyplot as plt
fig = plt.figure(figsize=[7, 15])
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

train.plot('Garage Area', 'SalePrice', ax=ax1, kind='scatter')
train.plot('Gr Liv Area', 'SalePrice', ax=ax2, kind='scatter')
train.plot('Overall Cond', 'SalePrice', ax=ax3, kind='scatter')

print(train[['Garage Area', 'Gr Liv Area', 'Overall Cond', 'SalePrice']].corr())

# Training model
from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
model = LinearRegression().fit(train[['Gr Liv Area']], train['SalePrice'])  #error if train['Gr Liv Area'] - ValueError: Found input variables with inconsistent numbers of samples: 
a1 = model.coef_
a0 = model.intercept_

# Predict and evaluate
import numpy as np
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])
prediction_train = lr.predict(train[['Gr Liv Area']])
prediction_test = lr.predict(test[['Gr Liv Area']])
train_rmse = np.sqrt(mean_squared_error(train['SalePrice'], prediction_train))
test_rmse = np.sqrt(mean_squared_error(test['SalePrice'], prediction_test))

## Feature selection
import pandas as pd
data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]
#train.info()
# Select integer and float columns
numerical_train = train.select_dtypes(include=['int', 'float'])
# Drop meaningless columns
numerical_train = numerical_train.drop(columns = ['PID', 'Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Mo Sold', 'Yr Sold'])  # or ([], axis=1)
# Calculate number of missing values into a series object
null_series = numerical_train.isnull().sum()
# Columns with no missing values
full_cols_series = null_series[null_series==0] 

# Calculate correlation values
train_subset = train[full_cols_series.index]
sorted_corrs = abs(train_subset.corr()['SalePrice']).sort_values()

# Diagnosing collinearity
# To avoid the risk of information overload, we can generate a correlation matrix heatmap
import seaborn as sns
import matplotlib.pyplot as plt

strong_corrs = sorted_corrs[sorted_corrs>0.3]
corrmat = train_subset[strong_corrs.index].corr()
sns.heatmap(corrmat)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# There are 2 pairs that are strongly correlated. Because Gr Liv Area and Garage Area are continuous variables that capture more nuance, let's drop the TotRms AbvGrd and Garage Cars
final_corr_cols = strong_corrs.drop(['Garage Cars', 'TotRms AbvGrd'])

# Confirm and filter that the test set contains no missing values
print(test[final_corr_cols.index].info())
clean_test = test[final_corr_cols.index].dropna()

features = final_corr_cols.drop(['SalePrice']).index
target = 'SalePrice'

model = LinearRegression().fit(train[features], train[target])
train_predictions = model.predict(train[features])
train_rmse = np.sqrt(mean_squared_error(train[target], train_predictions))

test_predictions = model.predict(clean_test[features])
test_rmse = np.sqrt(mean_squared_error(clean_test[target], test_predictions))  

## Training and testing after removing features with low variance
# Rescaling
unit_train = (train[features] - train[features].min())/(train[features].max() - train[features].min())
print(unit_train.max())   # should be 1
print(unit_train.min())   # should be 0

print(unit_train.var())
features = features.drop('Open Porch SF')

model = LinearRegression().fit(train[features], train[target])
train_predictions = model.predict(train[features])
train_rmse_2 = np.sqrt(mean_squared_error(train[target], train_predictions))

test_predictions = model.predict(clean_test[features])
test_rmse_2 = np.sqrt(mean_squared_error(clean_test[target], test_predictions))

# Gradient descent
def derivative(a1, xi_list, yi_list):
    deriv = 0
    for i in range(len(xi_list)):
        deriv += (2/len(xi_list))*xi_list[i]*(a1*xi_list[i] - yi_list[i])
    return deriv

def gradient_descent(xi_list, yi_list, max_iterations, alpha, a1_initial):
    a1_list = [a1_initial]

    for i in range(0, max_iterations):
        a1 = a1_list[i]
        deriv = derivative(a1, xi_list, yi_list)
        a1_new = a1 - alpha*deriv
        a1_list.append(a1_new)
    return(a1_list)

param_iterations = gradient_descent(train['Gr Liv Area'], train['SalePrice'], 20, .0000003, 150)
final_param = param_iterations[-1]

# Multiple paramter gradient descent
def a1_derivative(a0, a1, xi_list, yi_list):
    len_data = len(xi_list)
    error = 0
    for i in range(0, len_data):
        error += xi_list[i]*(a0 + a1*xi_list[i] - yi_list[i])
    deriv = 2*error/len_data
    return deriv

def a0_derivative(a0, a1, xi_list, yi_list):
    len_data = len(xi_list)
    error = 0
    for i in range(len_data):
        error += a0 + a1*xi_list[i] - yi_list[i]
    deriv = (2/len_data)*error
    return deriv

def gradient_descent(xi_list, yi_list, max_iterations, alpha, a1_initial, a0_initial):
    a1_list = [a1_initial]
    a0_list = [a0_initial]

    for i in range(0, max_iterations):
        a1 = a1_list[i]
        a0 = a0_list[i]
        
        a1_deriv = a1_derivative(a0, a1, xi_list, yi_list)
        a0_deriv = a0_derivative(a0, a1, xi_list, yi_list)
        
        a1_new = a1 - alpha*a1_deriv
        a0_new = a0 - alpha*a0_deriv
        
        a1_list.append(a1_new)
        a0_list.append(a0_new)
    return(a0_list, a1_list)

a0_params, a1_params = gradient_descent(train['Gr Liv Area'], train['SalePrice'], 20, .0000003, 150, 1000)

## Original least squares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

features = ['Wood Deck SF', 'Fireplaces', 'Full Bath', '1st Flr SF', 'Garage Area',
       'Gr Liv Area', 'Overall Qual']
X = train[features]
X['bias'] = 1
X = X[['bias']+features]  #rearrange so that 'bias' stands first
y = train['SalePrice']

first_term = np.linalg.inv(
        np.dot(
            np.transpose(X),
            X
        )
    )
second_term = np.dot(
    np.transpose(X),
    y
)

ols_estimation = np.dot(first_term, second_term)

## Processing and transforming features
import pandas as pd

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

train_null_counts = train.isnull().sum()
#print(train_null_counts)

no_mv = train_null_counts[train_null_counts==0]  #columns with no missing
df_no_mv = train[no_mv.index] 

# Convert all text columns to categorical data type
text_cols = df_no_mv.select_dtypes(include=['object']).columns

print('Number of categories')
for col in text_cols:
    train[col] = train[col].astype('category')
    print(col+":", len(train[col].unique()))
    
train['Utilities'].cat.codes.value_counts()

# Dummy coding
dummy_cols = pd.DataFrame()
for col in text_cols:
    col_dummies = pd.get_dummies(train[col])
    train = pd.concat([train, col_dummies], axis=1)
    del train[col]

# Create a new feature that are math operated result of other features
train['years_until_remod'] = train['Year Remod/Add'] - train['Year Built']

# Treating missing values
import pandas as pd

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

train_null_counts = train.isnull().sum()
df_missing_values = train[train_null_counts[(train_null_counts>0) &        # error without parentheses
                                            (train_null_counts<584)].index]
print(df_missing_values.isnull().sum())
print(df_missing_values.info())

# Missing value imputation (for numerical features)
float_cols = df_missing_values.select_dtypes(include=['float'])
float_cols = float_cols.fillna(float_cols.mean())
float_cols.isnull().sum()


## Logistic regression
import pandas as pd
import matplotlib.pyplot as plt

admissions = pd.read_csv('admissions.csv')
plt.scatter(admissions['gpa'], admissions['admit'])
plt.show()

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(admissions[["gpa"]], admissions["admit"])

# Return probability for outcome=1
pred_probs = logistic_model.predict_proba(admissions[['gpa']])
plt.scatter(admissions['gpa'], pred_probs[:,1])  #linear relationship expected

# Label prediction
fitted_labels = logistic_model.predict(admissions[['gpa']])
plt.scatter(admissions['gpa'], fitted_labels)

# Evaluation
labels = model.predict(admissions[['gpa']])
admissions['predicted_label'] = labels
print(admissions['predicted_label'].value_counts())
print(admissions.head(5))

admissions['actual_label'] = admissions['admit']
matches = admissions['predicted_label'] == admissions['actual_label']
correct_predictions = admissions[matches == True]
correct_predictions.head(5)
accuracy = len(correct_predictions)/len(admissions)

true_positives = len(admissions[(admissions['predicted_label']==1) & 
                                (admissions['actual_label']==1)])
true_negatives = len(admissions[(admissions['predicted_label']==0) & 
                                (admissions['actual_label']==0)])
false_negatives = len(admissions[(admissions['predicted_label']==0) &
                                 (admissions['actual_label']==1)])
sensitivity = true_positives/(true_positives+false_negatives)
false_positives = len(admissions[(admissions['predicted_label']==1) &
                                 (admissions['actual_label']==0)])
specificity = true_negatives/(false_positives+true_negatives)

## Multiclass classification
import pandas as pd
cars = pd.read_csv("auto.csv")
unique_regions = cars['origin'].unique()

# Get dummies
dummy_cylinders = pd.get_dummies(cars["cylinders"], prefix="cyl")
cars = pd.concat([cars, dummy_cylinders], axis=1)
                 
dummy_years = pd.get_dummies(cars['year'], prefix='year')
cars = pd.concat([cars, dummy_years], axis=1)
cars = cars.drop(['cylinders', 'year'], axis=1)
print(cars.head(5))

# Make train - test sets
shuffled_rows = np.random.permutation(cars.index)
shuffled_cars = cars.iloc[shuffled_rows]

train = shuffled_cars.iloc[:round(len(cars)*0.7)]
test = shuffled_cars.iloc[round(len(cars)*0.7):]

# Training
from sklearn.linear_model import LogisticRegression

unique_origins = cars["origin"].unique()
unique_origins.sort()

models = {}
features = [c for c in train.columns if c.startswith('cyl') or c.startswith('year')]
X = cars[features]
for value in unique_origins:
    y = cars['origin']==value
    model = LogisticRegression().fit(X, y)
    models[value] = model

# Evaluate
testing_probs = pd.DataFrame(columns=unique_origins)

for model in models:   #can be for origin in unique_origins
    prob = models[model].predict_proba(test[features])[:,1]
    testing_probs[model] = prob

predicted_origins = testing_probs.idxmax(axis=1) #return a series where each value corresponds to the column or where the maximum value occurs for that observation

## Overfitting
import pandas as pd
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
cars = pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
filtered_cars = cars[cars['horsepower'] != '?'].copy() #  If you run the code locally in Jupyter Notebook or Jupyter Lab, you'll notice a SettingWithCopy Warning. It's considered good practice to include .copy() whenever you perform operations on a dataframe.
filtered_cars['horsepower'] = filtered_cars['horsepower'].astype('float')

# Function for computing bias and variance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def train_and_test(cols):
    model = LinearRegression().fit(filtered_cars[cols], filtered_cars['mpg'])
    predict = model.predict(filtered_cars[cols])
    variance = np.var(predict)
    mse = mean_squared_error(predict, filtered_cars['mpg'])
    return(mse, variance)

cyl_mse, cyl_var = train_and_test(['cylinders'])
weight_mse, weight_var = train_and_test(['weight'])
three_mse, three_var = train_and_test(['cylinders', 'displacement', 'horsepower'])
four_mse, four_var = train_and_test(['cylinders', 'displacement', 'horsepower', 'weight'])

# Function for computing the cross validation error
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

def train_and_cross_val(cols):
    model = LinearRegression().fit(filtered_cars[cols], filtered_cars['mpg'])
    kf = KFold(n_splits=10, shuffle=True, random_state=3)
    mses = -1*cross_val_score(model,   #can integrate mse in the below loop
                filtered_cars[cols], 
                filtered_cars['mpg'],
                scoring='neg_mean_squared_error',  #this return negative values
                cv = kf)
    variances = []
    for train_index, test_index in kf.split(filtered_cars):
        train = filtered_cars.iloc[train_index]
        test = filtered_cars.iloc[test_index]
        lr = LinearRegression()
        lr.fit(train[cols], train['mpg'])
        predictions = lr.predict(test[cols])
        var = np.var(predictions)
        variances.append(var)
    return(np.mean(mses), np.mean(variances))
two_mse, two_var = train_and_cross_val(["cylinders", "displacement"])
three_mse, three_var = train_and_cross_val(["cylinders", "displacement", "horsepower"])
four_mse, four_var = train_and_cross_val(["cylinders", "displacement", "horsepower", "weight"])
five_mse, five_var = train_and_cross_val(["cylinders", "displacement", "horsepower", "weight", "acceleration"])
six_mse, six_var = train_and_cross_val(["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"])
seven_mse, seven_var = train_and_cross_val(["cylinders", "displacement", "horsepower", "weight", "acceleration","model year", "origin"])
# Sample solution
# def train_and_cross_val(cols):
#     features = filtered_cars[cols]
#     target = filtered_cars["mpg"]
    
#     variance_values = []
#     mse_values = []
    
#     # KFold instance.
#     kf = KFold(n_splits=10, shuffle=True, random_state=3)
    
#     # Iterate through over each fold.
#     for train_index, test_index in kf.split(features):
#         # Training and test sets.
#         X_train, X_test = features.iloc[train_index], features.iloc[test_index]
#         y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
#         # Fit the model and make predictions.
#         lr = LinearRegression()
#         lr.fit(X_train, y_train)
#         predictions = lr.predict(X_test)
        
#         # Calculate mse and variance values for this fold.
#         mse = mean_squared_error(y_test, predictions)
#         var = np.var(predictions)

#         # Append to arrays to do calculate overall average mse and variance values.
#         variance_values.append(var)
#         mse_values.append(mse)
   
#     # Compute average mse and variance values.
#     avg_mse = np.mean(mse_values)
#     avg_var = np.mean(variance_values)
#     return(avg_mse, avg_var)

# Plot the error and variance 
import matplotlib.pyplot as plt
        
mses = [two_mse, three_mse, four_mse, five_mse, six_mse, seven_mse]
variances = [two_var, three_var, four_var, five_var, six_var, seven_var]
plt.scatter(range(2,8), mses, color ='red')
plt.scatter(range(2,8), variances, color ='blue')
plt.show()

## Clustering basics
import pandas as pd
votes = pd.read_csv('114_congress.csv')

# Find how many Senators are in each party
votes['party'].value_counts()

# Find out the 'average' vote for each bill was
print(votes.mean())

# Euclidean distance
from sklearn.metrics.pairwise import euclidean_distances
print(euclidean_distances(votes.iloc[0,3:].values.reshape(1, -1), votes.iloc[1,3:].values.reshape(1, -1)))
distance = euclidean_distances(votes.iloc[0,3:], votes.iloc[2,3:])

# Train
import pandas as pd
from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=2, random_state=1)
senator_distances = kmeans_model.fit_transform(votes.iloc[:,3:])

# Crosstab to shows whether the clusters tend to break down along party lines or not.
labels = kmeans_model.labels_
pd.crosstab(votes['party'], labels)

# Explore the 'outliers'
democratic_outliers = votes[(labels==1) &(votes['party']=='D')]

# Visualization
plt.scatter(senator_distances[:,0], senator_distances[:,1],  # np array has no attribute iloc
            c = labels,
            linewidths =0)
plt.show()
# The most extreme Senators are those who are the furthest away from one cluster. For example, a radical Republican would be as far from the Democratic cluster as possible. 
# Senators who are in between both clusters are more moderate, as they fall between the views of the two parties.

# Create a formula to find extremists -- we'll cube the distances in both columns of senator_distances, then add them together.
# If we left the distances as is, the moderate, who is between both parties, seem extreme.
extremism = (senator_distances**3).sum(axis=1)
votes['extremism'] = extremism
votes.sort_values('extremism', inplace = True, ascending = False)  #inplace: overwrite the existing dataframe
print(votes.head(10))

## K-means clustering
import pandas as pd
import numpy as np

nba = pd.read_csv("nba_2013.csv")
nba.head(3)

point_guards = nba[nba['pos']=='PG']
point_guards['ppg'] = point_guards['pts'] / point_guards['g']  #points per game column

# Make sure ppg = pts/g
point_guards[['pts', 'g', 'ppg']].head(5)

# Create a column of Assist Turnover Ratio
point_guards = point_guards[point_guards['tov'] != 0]
point_guards['atr'] = point_guards['ast']/point_guards['tov']

# Visualize ppg and atr
plt.scatter(point_guards['ppg'], point_guards['atr'], c='y')
plt.title("Point Guards")
plt.xlabel('Points Per Game', fontsize=13)
plt.ylabel('Assist Turnover Ratio', fontsize=13)
plt.show()

# K-means is an iterative algorithm
num_clusters = 5
# Use numpy's random function to generate a list, length: num_clusters, of indices
random_initial_points = np.random.choice(point_guards.index, size=num_clusters)

# Use the random indices to create the centroids
centroids = point_guards.loc[random_initial_points]

# Visualize where the randomly chosen centroids started out
plt.scatter(point_guards['ppg'], point_guards['atr'], c='yellow')
plt.scatter(centroids['ppg'], centroids['atr'], c='red')
plt.title("Centroids")
plt.xlabel('Points Per Game', fontsize=13)
plt.ylabel('Assist Turnover Ratio', fontsize=13)
plt.show()

# Extract centroid ID and coordinates
def centroids_to_dict(centroids):
    dictionary = dict()
    # iterating counter we use to generate a cluster_id
    counter = 0

    # iterate a pandas data frame row-wise using .iterrows()
    for index, row in centroids.iterrows():
        coordinates = [row['ppg'], row['atr']]
        dictionary[counter] = coordinates
        counter += 1

    return dictionary

centroids_dict = centroids_to_dict(centroids)

# Create a function that calculate distance from points to centroid
import math

def calculate_distance(centroid, player_values):
    root_distance = 0
    
    for x in range(0, len(centroid)):
        difference = centroid[x] - player_values[x]
        squared_difference = difference**2
        root_distance += squared_difference

    euclid_distance = math.sqrt(root_distance)
    return euclid_distance

q = [5, 2]
p = [3,1]

# Sqrt(5) = ~2.24
print(calculate_distance(q, p))

# Step 1: Assign points to the closest centroid
from sklearn.metrics.pairwise import euclidean_distances
def assign_to_cluster(row):
    lowest_distance = euclidean_distances(row[['ppg', 'atr']], centroids_dict[0])  # solution: [row['ppg'], row['atr']]
    closest_cluster = 0 #solution: lowest_distance and closest_cluster = -1
    for cluster_id, centroid in centroids_dict.items():
        distance = euclidean_distances(row[['ppg', 'atr']], centroid) #solution: calculate_distance()
        if distance < lowest_distance:
            lowest_distance = distance
            closest_cluster = cluster_id
    return closest_cluster                                      
  
# applying assign_to_cluster row-by-row
point_guards['cluster'] = point_guards.apply(lambda row: assign_to_cluster(row), axis=1)

# Visualizing clusters
def visualize_clusters(df, num_clusters):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for n in range(num_clusters):
        clustered_df = df[df['cluster'] == n]
        plt.scatter(clustered_df['ppg'], clustered_df['atr'], c=colors[n-1])
        plt.xlabel('Points Per Game', fontsize=13)
        plt.ylabel('Assist Turnover Ratio', fontsize=13)
    plt.show()

visualize_clusters(point_guards, 5)

# Step 2: recalculate the centroids for each cluster
def recalculate_centroids(df):
    new_centroids_dict = dict()
    num_clusters = df['cluster'].max() + 1
    
    for cluster_id in range(0, num_clusters):
        players = df[df['cluster']==cluster_id]
        new_centroid = players[['ppg', 'atr']].mean()   #solution: np.average()
        new_centroids_dict[cluster_id] = new_centroid
        return new_centroids_dict

centroids_dict = recalculate_centroids(point_guards)

# Re-run step 1
point_guards['cluster'] = point_guards.apply(lambda row: assign_to_cluster(row), axis=1)
visualize_clusters(point_guards, num_clusters)

# Re run step 2
centroids_dict = recalculate_centroids(point_guards)
point_guards['cluster'] = point_guards.apply(lambda row: assign_to_cluster(row), axis=1)
visualize_clusters(point_guards, num_clusters)

# sklearn implementation of K-means re-run the process many times with random initial centroids
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(point_guards[['ppg', 'atr']])
point_guards['cluster'] = kmeans.labels_

visualize_clusters(point_guards, num_clusters)
i
## Decition trees
import pandas

# Set index_col to False to avoid pandas thinking that the first column is row indexes (it's age)
income = pandas.read_csv("income.csv", index_col=False)
print(income.head(5))

# Convert columns from text categories to numbers
# If convert the columns to categorical types, pandas displays the labels as strings, but internally store them as numbers so we can do computations with them. The numbers aren't always compatible with other librarie
raw_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'high_income']
income[raw_cols] = income[raw_cols].apply(lambda col: pandas.Categorical(col).codes, axis=0)   #retain numbers only
print(income[raw_cols].head(5))

# We are predicting on high_income
# Compute the information gain for splitting on the age column of income.
import numpy

def calc_entropy(column):
    """
    Calculate entropy given a pandas series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column
    counts = numpy.bincount(column)
    # Divide by the total column length to get a probability
    probabilities = counts / len(column)
    
    # Initialize the entropy to 0
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)
    
    return -entropy

# Verify that our function matches our answer from earlier
entropy = calc_entropy([1,1,0,0,1])
print(entropy)

information_gain = entropy - ((.8 * calc_entropy([1,1,0,0])) + (.2 * calc_entropy([1])))
print(information_gain)

median_age = income["age"].median()
cut_off = income['age']>median_age
# print(cut_off.head(10))
probabilities = (numpy.bincount(cut_off))/len(income['age'])   #? how are values in .bincount() ordered
age_information_gain = calc_entropy(income['high_income']) - (probabilities[0]*calc_entropy(income['high_income'][-cut_off]) + probabilities[1]*calc_entropy(income['high_income'][cut_off]))

# We'll find the initial variable to split on by calculating which split would have the highest information gain.
def calc_information_gain(data, split_name, target_name):
    """
    Calculate information gain given a data set, column to split on, and target
    """
    # Calculate the original entropy
    original_entropy = calc_entropy(data[target_name])
    
    # Find the median of the column we're splitting
    column = data[split_name]
    median = column.median()
    
    # Make two subsets of the data, based on the median
    left_split = data[column <= median]
    right_split = data[column > median]
    
    # Loop through the splits and calculate the subset entropies
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0]) 
        to_subtract += prob * calc_entropy(subset[target_name])
    
    # Return information gain
    return original_entropy - to_subtract

# Verify that our answer is the same as on the last screen
print(calc_information_gain(income, "age", "high_income"))

columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]
information_gains = []
for col in columns:
    information_gain = calc_information_gain(income, col, 'high_income')
    information_gains.append(information_gain)
highest_gain_index = information_gains.index(max(information_gains)) # equivalent to which() in R
highest_gain = columns[highest_gain_index]

# Returns the name of the column we should use to split a data set.
def find_best_column(data, target_name, columns):
    # Fill in the logic here to automatically find the column in columns to split on
    # data is a dataframe
    # target_name is the name of the target variable
    # columns is a list of potential columns to split on
    information_gains = []
    for col in columns:
        information_gain = calc_information_gain(data, col, target_name)
        information_gains.append(information_gain)
    highest_gain_index = information_gains.index(max(information_gains))
    return columns[highest_gain_index]

# A list of columns to potentially split income with
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

income_split = find_best_column(income, 'high_income', columns)

# Simple version of id3 algorithm: store the label of the node
# We'll use lists to store our labels for nodes (when we find them)
# Lists can be accessed inside our recursive function, whereas integers can't.  
# Look at the python missions on scoping for more information on this topic
label_1s = []
label_0s = []

def id3(data, target, columns):
    # The pandas.unique method will return a list of all the unique values in a series
    unique_targets = pandas.unique(data[target])
    
    if len(unique_targets) == 1:
        # Insert code here to append 1 to label_1s or 0 to label_0s, based on what we should label the node
        # See lines 2 and 3 in the algorithm
        
        # Returning here is critical -- if we don't, the recursive tree will never finish, and run forever
        # See our example above for when we returned
        if 0 in unique_targets:
            label_0s.append(0)
        elif 1 in unique_targets:
            label_1s.append(1)
        return 
    
    # Find the best column to split on in our data
    best_column = find_best_column(data, target, columns)
    # Find the median of the column
    column_median = data[best_column].median()
    
    # Create the two splits
    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    
    # Loop through the splits and call id3 recursively
    for split in [left_split, right_split]:
        # Call id3 recursively to process each branch
        id3(split, target, columns)
    
# Create the data set that we used in the example on the last screen
data = pandas.DataFrame([
    [0,20,0],
    [0,60,2],
    [0,40,1],
    [1,25,1],
    [1,35,2],
    [1,55,1]
    ])
# Assign column names to the data
data.columns = ["high_income", "age", "marital_status"]

# Call the function on our data to set the counters properly
id3(data, "high_income", ["age", "marital_status"])

# Create a dictionary to hold the tree  
# It has to be outside of the function so we can access it later
tree = {}

# This list will let us number the nodes  
# It has to be a list so we can access it inside the function
nodes = []

def id3(data, target, columns, tree):
    unique_targets = pandas.unique(data[target])
    
    # Assign the number key to the nodes list
    nodes.append(len(nodes) + 1)
    tree["number"] = nodes[-1]

    if len(unique_targets) == 1:
        # Insert code here that assigns the "label" field to the nodes list
        if 0 in unique_targets:
            tree['label'] = 0
        elif 1 in unique_targets:
            tree['label'] = 1
        return
    
    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()
    
    # Insert code here that assigns the "column" and "median" fields to the nodes list
    tree['column'] = best_column
    tree['median'] = column_median
    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    split_dict = [["left", left_split], ["right", right_split]]
    
    for name, split in split_dict:
        tree[name] = {}
        id3(split, target, columns, tree[name])

# Call the function on our data to set the counters properly
id3(data, "high_income", ["age", "marital_status"], tree)

# Fix tree appearance by printing it out in a better format.
# def print_node(tree, depth):
#     1 Check for the presence of the "label" key in the tree
#     2     If found, print the label and return
#     3 Print out the tree's "column" and "median" keys
#     4 Iterate through the tree's "left" and "right" keys
#     5     Recursively call print_node(tree[key], depth+1)
def print_with_depth(string, depth):
    # Add space before a string
    prefix = "    " * depth
    # Print a string, and indent it appropriately
    print("{0}{1}".format(prefix, string))
    
def print_node(tree, depth):
    # Check for the presence of "label" in the tree
    if "label" in tree:
        # If found, then this is a leaf, so print it and return
        print_with_depth("Leaf: Label {0}".format(tree["label"]), depth)
        # This is critical -- without it, you'll get infinite recursion
        return
    # Print information about what the node is splitting on
    print_with_depth("{0} > {1}".format(tree["column"], tree["median"]), depth)
    
    # Create a list of tree branches
    branches = [tree["left"], tree["right"]]
        
    # Insert code here to recursively call print_node on each branch
    for branch in [tree['left'], tree['right']]:
        print_node(branch, depth+1)
    # Don't forget to increment depth when you pass it in

print_node(tree, 0)  #The left branch prints out first

# Write a function that makes predictions automatically
# def predict(tree, row):
#     1 Check for the presence of "label" in the tree dictionary
#     2    If found, return tree["label"]
#     3 Extract tree["column"] and tree["median"]
#     4 Check whether row[tree["column"]] is less than or equal to tree["median"]
#     5    If it's less than or equal, call predict(tree["left"], row) and return the result
#     6    If it's greater, call predict(tree["right"], row) and return the result
def predict(tree, row):
    if "label" in tree:
        return tree["label"]
    
    column = tree["column"]
    median = tree["median"]
    
    # Insert code here to check whether row[column] is less than or equal to median
    # If it's less than or equal, return the result of predicting on the left branch of the tree
    # If it's greater, return the result of predicting on the right branch of the tree
    if row[column] <= median:
        return predict(tree['left'], row)
    return predict(tree['right'], row) 

    # Remember to use the return statement to return the result!

# Print the prediction for the first row in our data
print(predict(tree, data.iloc[0]))

# Predict for multiple rows
new_data = pandas.DataFrame([
    [40,0],
    [20,2],
    [80,1],
    [15,1],
    [27,2],
    [38,1]
    ])
# Assign column names to the data
new_data.columns = ["age", "marital_status"]

def batch_predict(tree, df):
    predictions = df.apply(lambda row: predict(tree, row), axis=1)
    return predictions

predictions = batch_predict(tree, new_data)

## Applying decision tree
from sklearn.tree import DecisionTreeClassifier

# A list of columns to train with
# We've already converted all columns to numeric
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

# Instantiate the classifier
# Set random_state to 1 to make sure the results are consistent
clf = DecisionTreeClassifier(random_state=1)
clf.fit(income[columns], income['high_income'])

# Evaluating
import numpy
import math

# Set a random seed so the shuffle is the same every time
numpy.random.seed(1)

# Shuffle the rows  
# This permutes the index randomly using numpy.random.permutation
# Then, it reindexes the dataframe with the result
# The net effect is to put the rows into random order
income = income.reindex(numpy.random.permutation(income.index))

train_max_row = math.floor(income.shape[0] * .8)
train = income.iloc[:train_max_row, :]
test = income.iloc[train_max_row:, :]

from sklearn.metrics import roc_auc_score

clf = DecisionTreeClassifier(random_state=1, min_samples_split = 13)
clf.fit(train[columns], train["high_income"])

predictions = clf.predict(test[columns])
error = roc_auc_score(test['high_income'], predictions)

predictions_train = clf.predict(train[columns]) 
print(roc_auc_score(train['high_income'], predictions_train)) # if the AUC between training set predictions and actual values is significantly higher than the AUC between test set predictions and actual values, it's a sign that the model may be overfitting.


