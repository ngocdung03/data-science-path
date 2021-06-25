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