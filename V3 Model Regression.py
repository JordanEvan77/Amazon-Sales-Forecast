import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pd.options.mode.chained_assignment = None
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/5067 ML Code/DTC Final Project/'

data_final = pd.read_csv(path+'data_final.csv')
###############################
# TEST TRAIN SPLIT: REGRESSION
###############################

print(data_final.columns)
# ['Order ID', 'Product', 'Quantity Ordered', 'Price Each', 'Order Date','Purchase Address',
# 'Purchase City',
# 'Purchase State', 'Order Date New','Order Time', 'Full Purchase', 'Large Purchase', 'Order Day of Week',
# 'Order Day of Month', 'Order Month of Year', 'Order Hour of Day', 'Product Num', 'State Num',
# 'City Num']

# Split up data for modeling fit:
X = data_final[['Order Day of Week', 'Order Day of Month', 'Order Month of Year', 'Order Hour of Day',
                'Product Num', 'State Num', 'City Num', 'Quantity Ordered']]

y = data_final['Full Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 21)
# ready for training model!

##################
# Linear Regression:
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_pred = lin_reg.predict(X_test)

print('r2', r2_score(y_test, lin_pred))
print('MSE', mean_squared_error(y_test, lin_pred))
#r2 0.032870708913033786
#MSE 106236.16762002453 # as expected, very poor performance

##################
# Random Forest
regressor_f = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_f.fit(X_train, y_train)
# the regressor works!

prediction = regressor_f.predict(X_test)

print('r2', r2_score(y_test, prediction))
print('MSE', mean_squared_error(y_test, prediction))
#r2 0.9999855176120136
#MSE 1.5908456209892834 # Very nice performance!!!

# CROSS VALIDATION ON RF:
scores = cross_val_score(regressor_f, X, y, scoring="neg_mean_squared_error",
                         cv=10)
rf_rmse_scores = np.sqrt(-scores)

print('Scores', rf_rmse_scores)
print('Mean', rf_rmse_scores.mean()) #4.777
print('SD', rf_rmse_scores.std())
# Very acceptable results:
# Mean 1.5594298927330892
# SD 1.3709647822894626


##################
#simple RNN
# I want to predict the next order full purchase size, based purely on the timing factor:
# with major help from: https://www.datatechnotes.com/2018/12/rnn-example-with-keras-simplernn-in
# .html

# Limit the data set to the time and important variable of purchase amount.
data_fs = data_final[["Order Day of Year", "Full Purchase"]]
data_ts = data_fs.groupby("Order Day of Year").sum()
scaler = MinMaxScaler(feature_range=(0, 1))
data_ts = scaler.fit_transform(data_ts)

plt.plot(data_ts)
plt.show()


train = data_ts[0:290, :]
test = data_ts[290:, :]

step_size = 4

# Create set of matrices, so that the step wise data is formatted
def Matrix(data, step=step_size):
 X = []
 Y = []
 for i in range(len(data)-step):
  d=i+step
  X.append(data[i:d,])
  Y.append(data[d,])
 return np.array(X), np.array(Y)

train_X, train_y =Matrix(train)

test_X, test_y =Matrix(test)

# reshape to 3D for keras:
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# layer a sequential simple RNN model. Approach with multiple layers, to a single regression output.
rnn_model = keras.models.Sequential([
    keras.layers.SimpleRNN(30, return_sequences=True, input_shape=(1, step_size)),
    keras.layers.SimpleRNN(30, return_sequences=True),
    keras.layers.SimpleRNN(1)
])

rnn_model.compile(loss='mean_squared_error', optimizer='adam') # If I had a validation set I
# could also use metrics = ["accuracy"]
rnn_model.summary()

back_end = rnn_model.fit(train_X, train_y, epochs=100, batch_size=16, verbose=2, )
trainPredict = rnn_model.predict(train_X)
testPredict= rnn_model.predict(test_X)
predicted=np.concatenate((trainPredict,testPredict),axis=0)

trainScore = rnn_model.evaluate(train_X, train_y, verbose=0)
print(trainScore) # MSE
#0.008200788870453835  # really good score!


#graph overlapping prediction and historical.
plt.plot(data_ts)
plt.plot(predicted, c="g")
plt.axvline(x=290, c="r")
plt.show()

#Performance from history:
pd.DataFrame(back_end.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.gca().set_xlim(0, 20)
plt.show()



