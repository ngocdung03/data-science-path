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
df_missing_values = train[train_null_counts[(train_null_counts>0) &
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

