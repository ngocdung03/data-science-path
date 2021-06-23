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
    mse_values = mean_squared_error(test_df['price'], predictions)
    print(mse_values)