import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pd.options.mode.chained_assignment = None
import numpy as np
import sklearn
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/5067 ML Code/DTC Final Project/'

data_final = pd.read_csv(path+'data_final.csv')


#################################
#TEST TRAIN SPLIT: CLASSIFICATION
#################################
X = data_final[['Order Day of Week', 'Order Day of Month', 'Order Month of Year', 'Order Hour of Day',
                'Product Num', 'State Num', 'City Num']] # removed 'Quantity Ordered'

y = data_final['Large Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 21)
#ready for training model!

##################
# try decision Tree First:
# Classify Large purchase
clf = DecisionTreeClassifier()
clf1 = clf.fit(X_train, y_train)
prediction2 = clf1.predict(X_test)

print('r2', r2_score(y_test, prediction2))
print('MSE', mean_squared_error(y_test, prediction2))
#r2 0.9776858900495985
#MSE 0.0038774301254779473


#############################
# try K Nearest Neighbor next as a baseline model:
# Classify Large purchase

scaler1 = StandardScaler()
scaler1.fit(X_train.values)
X_train = scaler1.transform(X_train.values)
X_test = scaler1.transform(X_test.values)

knn = sklearn.neighbors.KNeighborsRegressor(
    n_neighbors = 3)
knn1 = knn.fit(X_train, y_train)
prediction3 = knn1.predict(X_test)


print('r2', r2_score(y_test, prediction3))
print('MSE', mean_squared_error(y_test, prediction3))
#Mixed performance:
#r2 0.06819651511899949
#MSE 0.16191561802526316

################################################
###Neural Network MLP Classify the Purchase Type:
# NEW MODEL AND GOAL TO PREDICT IF IT WILL BE A LARGE OR NORMAL PURCHASE
#normalize first:
data_final_le = data_final[['Order Day of Week', 'Order Day of Month', 'Order Month of Year', 'Order Hour of Day',
                'Product Num', 'State Num', 'City Num', 'Quantity Ordered', 'Large Purchase',
                            'Full Purchase']]

normalizer = preprocessing.MinMaxScaler()
col_names = data_final_le.columns

norm = normalizer.fit_transform(data_final_le)
norm_df = pd.DataFrame(norm, columns=col_names)
print(norm_df.head(5))

X = norm_df[['Order Day of Week', 'Order Day of Month', 'Order Month of Year', 'Order Hour of Day',
                'Product Num', 'State Num', 'City Num', 'Quantity Ordered']]

y = norm_df['Large Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 20)

#Then MLP:
clf = MLPClassifier(hidden_layer_sizes=(15,15,15), random_state=21, max_iter=2000)
clf.fit(X_train, y_train)
TrainEstimate = clf.predict(X_test)

# score:
print('r2', r2_score(y_test, TrainEstimate))
print('MSE', mean_squared_error(y_test, TrainEstimate))# I could use Grid Search for the number of layers
# r2 0.9347231066312558
# MSE 0.011336097797404276

#GRID SEARCH ON MLP PURCHASE TYPE:
param_search = {
    'hidden_layer_sizes': [(15,20,20), (30,30,30), (50,50,50), (100,)],
    'activation': ['tanh', 'relu', 'selu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05]
}

clf_grid = MLPClassifier(random_state=21, max_iter=2000)
searcher = GridSearchCV(clf_grid, param_search, n_jobs=-1, cv=3)
searcher.fit(X_train, y_train)

print('Optimum Parameters', searcher.best_params_)
#Optimum Parameters {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'adam'}

### Do with optimum parameters:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 20)
clf_op = MLPClassifier(hidden_layer_sizes=(50,50,50), activation='tanh', solver='adam',
                    random_state=20,
                    max_iter=2000)
clf_op.fit(X_train, y_train)
TrainEstimate_op = clf_op.predict(X_test)

# score:
print('r2', r2_score(y_test, TrainEstimate_op))
print('MSE', mean_squared_error(y_test, TrainEstimate_op))
#r2 1.0
#MSE 0.0
#Is this over fit?

#######################################################
###Neural Network MLP Classify the Purchase Day of Week:
# NEW MODEL GOAL TO PREDICT WHICH DAY OF THE WEEK THE ORDER IS FROM
data_final_nm = data_final

cols_to_norm = ['Order Day of Month', 'Order Month of Year', 'Order Hour of Day','Product Num',
                'State Num', 'City Num', 'Quantity Ordered', 'Large Purchase','Full Purchase']

normalizer = preprocessing.MinMaxScaler()
col_names = data_final_nm[cols_to_norm].columns

norm = normalizer.fit_transform(data_final_nm[cols_to_norm])
norm_df = pd.DataFrame(norm, columns=col_names)
print(norm_df.head(5))

X = norm_df[cols_to_norm]

Y = data_final[['Order Day of Week']]  # I need to encode this as factors? so it can be categorized

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state= 20)

#Then MLP:
clf = MLPClassifier(hidden_layer_sizes=(15,15,15), random_state=21, max_iter=2000)
clf.fit(X_train, y_train.values.ravel()) # ravel to change shape
TrainEstimate = clf.predict(X_test)

# score:
print('r2', r2_score(y_test, TrainEstimate))
print('MSE', mean_squared_error(y_test, TrainEstimate))
# r2 -0.45622217767096007
# MSE 5.85349237977274

#GRID SEARCH ON MLP: PURCHASE DAY OF WEEK
param_search = {
    'hidden_layer_sizes': [(15,20,20), (30,30,30), (50,50,50), (100,)],
    'activation': ['tanh', 'relu', 'selu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05]
}

clf_grid = MLPClassifier(random_state=22, max_iter=2000)
searcher = GridSearchCV(clf_grid, param_search, n_jobs=-1, cv=3)
searcher.fit(X_train, y_train)

print('Optimum Parameters', searcher.best_params_)
#Optimum Parameters {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (30, 30, 30), 'solver': 'adam'}

### Do with optimum parameters:
clf = MLPClassifier(hidden_layer_sizes=(30,30,30), activation='tanh', solver='adam', alpha=0.001,
                    random_state=21,
                    max_iter=2000)
clf.fit(X_train, y_train)
TrainEstimate = clf.predict(X_test)

# score:
print('r2', r2_score(y_test, TrainEstimate))
print('MSE', mean_squared_error(y_test, TrainEstimate))
#r2 1.0
#MSE 0.0
#Over fitting?

########################################################
### Neural Network MLP Classify the Purchase Day of YEAR:
# NEW MODEL AND GOAL: Predict the purchase year
data_final_nm = data_final

cols_to_norm = ['Order Day of Week', 'Order Month of Year', 'Order Hour of Day','Product Num',
                'State Num', 'City Num', 'Quantity Ordered', 'Large Purchase','Full Purchase']

normalizer = preprocessing.MinMaxScaler()
col_names = data_final_nm[cols_to_norm].columns

norm = normalizer.fit_transform(data_final_nm[cols_to_norm])
norm_df = pd.DataFrame(norm, columns=col_names)
print(norm_df.head(5))

X = norm_df[cols_to_norm]

Y = data_final[['Order Day of Year']]  # I need to encode this as factors? so it can be categorized

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state= 20)

#Then MLP:
clf = MLPClassifier(hidden_layer_sizes=(15,15,15), random_state=21, max_iter=2000)
clf.fit(X_train, y_train.values.ravel()) # ravel to change shape
TrainEstimate = clf.predict(X_test)

# score:
print('r2', r2_score(y_test, TrainEstimate))
print('MSE', mean_squared_error(y_test, TrainEstimate))
#r2 0.9858209130537351
#MSE 163.24037374118154
#Large error, large explainability?

#GRID SEARCH ON MLP: PURCHASE DAY OF YEAR
param_search = {
    'hidden_layer_sizes': [(15, 20, 20), (30, 30, 30), (50, 50, 50), (100,)],
    'activation': ['tanh', 'relu', 'selu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05]
}


clf_grid = MLPClassifier(random_state=22, max_iter=2000)
searcher = GridSearchCV(clf_grid, param_search, n_jobs=-1, cv=3)
searcher.fit(X_train, y_train.values.ravel())

print('Optimum Parameters', searcher.best_params_)

### Do MLP with optimum parameters:
clf = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation='tanh', solver='adam', alpha=0.001,
                    random_state=21,
                    max_iter=2000)
clf.fit(X_train, y_train.values.ravel())
TrainEstimate = clf.predict(X_test)

# score:
print('r2', r2_score(y_test, TrainEstimate))
print('MSE', mean_squared_error(y_test, TrainEstimate))

#r2 0.9872085404372152
#MSE 147.264957725241
