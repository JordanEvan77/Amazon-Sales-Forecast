#Data source; https://www.kaggle.com/datasets/knightbearr/sales-product-data?select=Sales_November_2019.csv
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import datetime
from datetime import datetime
import sklearn
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/5067 ML Code/DTC Final Project/'
#raw_data = pd.read_csv(path + 'Raw Data/Sales_September_2019.csv')

data_lister= [path + 'Raw Data/Sales_September_2019.csv', path + 'Raw Data/Sales_October_2019.csv',
              path + 'Raw Data/Sales_November_2019.csv', path + 'Raw Data/Sales_December_2019.csv',
              path + 'Raw Data/Sales_January_2019.csv', path + 'Raw Data/Sales_February_2019.csv',
              path + 'Raw Data/Sales_March_2019.csv', path + 'Raw Data/Sales_April_2019.csv',
              path + 'Raw Data/Sales_May_2019.csv', path + 'Raw Data/Sales_June_2019.csv',
              path + 'Raw Data/Sales_July_2019.csv', path + 'Raw Data/Sales_August_2019.csv']



raw_data = pd.concat(map(pd.read_csv, data_lister), ignore_index=True)

print(raw_data.head(5))
print(raw_data.describe())
print(raw_data.columns)
#['Order ID', 'Product', 'Quantity Ordered', 'Price Each', 'Order Date', 'Purchase Address']

data_prog = raw_data

#remove repetition of column titles throughout data
data_prog = data_prog[data_prog["Product"].str.contains("Product")==False]



#feature extract address
data_prog[['Purchase Address', 'Purchase City', 'Purchase State']] = data_prog[
    'Purchase Address'].str.split(',', expand=True)

data_prog['Purchase State'] = data_prog['Purchase State'].apply(lambda x: x[:3])

#convert to date time
#data_prog['Order Date'] = datetime.strptime(data_prog['Order Date'], '%m/%d/%y ') #series error
# I want the time to be in a separate column.
data_prog['Order Date'] = pd.to_datetime(data_prog['Order Date'], infer_datetime_format=True)
data_prog['Order Date New'] = pd.to_datetime(data_prog['Order Date']).dt.date
print(data_prog['Order Date New'].head(5)) #looks good!

data_prog['Order Time'] = pd.to_datetime(data_prog['Order Date']).dt.time
print(data_prog['Order Time'].head(5))

data_prog['Order Day of Week'] = pd.to_datetime(data_prog['Order Date']).dt.dayofweek
data_prog['Order Day of Month'] = pd.to_datetime(data_prog['Order Date']).dt.day
data_prog['Order Month of Year'] = pd.to_datetime(data_prog['Order Date']).dt.month
data_prog['Order Hour of Day'] = pd.to_datetime(data_prog['Order Date']).dt.hour


#data_prog['Order Time'] = pd.to_datetime(data_prog['Order Time'], format='%H:%M').dt.time
#print(data_prog['Order Time'].head(5)) #should I leave this out?

data_prog['Price Each'] = pd.to_numeric(data_prog['Price Each'])
data_prog['Quantity Ordered'] = pd.to_numeric(data_prog['Quantity Ordered'])

#create new variable "Full Purchase" that gauges amount fully paid
data_prog["Full Purchase"] = data_prog['Price Each'] * data_prog['Quantity Ordered']
print(statistics.median(data_prog["Full Purchase"]))
print(statistics.mean(data_prog["Full Purchase"]))
#anything over 185 could be flagged as a large purchase:

data_prog['Large Purchase'] = (data_prog["Full Purchase"] > 185).astype(int)
print(data_prog['Large Purchase'])

#keep looking at visuals, begin looking for missing values, outliers, etc.

#NULL VALUES:
#do the counts per first:
data_prog.isna().sum()
#it appears we have no missing data!


#OUTLIERS PER ATTRIBUTE:
#show visuals, then look at outside 2.5 SD.
data_prog['Price Each'].hist(bins=50) # as expected, most items are cheaperz
plt.show()# far right price may be an outlier

data_prog['Full Purchase'].hist(bins=50) # as expected, most items are cheaper
plt.show()# far right price may be an outlier

data_prog['Quantity Ordered'].hist(bins=50) # as expected, most items are cheaper
plt.show()# mostly orders of single value


print(data_prog.describe())
#there are definitely outliers. Using Interquartile Range to gather them:
def get_outliers(df):
    # we want this to look at numeric data only, return dictionary
    df= df.select_dtypes(include='number')
    outliers = dict()
    for i in df.columns:
        temp_df = df[[i]]
        q1 = temp_df[i].quantile(.25)
        q3 = temp_df[i].quantile(.75)
        iqr = q3-q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr
        print(upper, lower)
        namer = '"%s"' % i
    #df.drop(df[(df[namer] > upper) | (df[namer] < lower)].index, inplace = True) # this should
        # work to drop
        newer = temp_df[(temp_df[i] > upper) | (temp_df[i] < lower)]

        outliers[namer] = newer[i].unique
    return outliers

df_outliers = get_outliers(data_prog)
print(df_outliers) # looks like it is finally working!
#now I can decide what to do with these values
#df.drop(df[df['Fee'] >= 24000].index, inplace = True)

# Create Dummies!
labelencoder = LabelEncoder()
#'Product'
data_prog['Product Num'] = labelencoder.fit_transform(data_prog['Product'])
#'Purchase State'
data_prog['State Num'] = labelencoder.fit_transform(data_prog['Purchase State'])
#'Purchase City'
data_prog['City Num'] = labelencoder.fit_transform(data_prog['Purchase City'])

#Duplicates
print(data_prog[data_prog.duplicated()]) # does this show duplication across all attributes?
#interesting, look at this and see if I want to remove duplicates
#if so, use drop_duplicates()

data_final = data_prog.drop_duplicates(keep="last")

#Further Visuals:
# looking at plain count of orders per hour of the day
hours_count = [j.hour for j in data_final['Order Time']]
numbers = [x for x in range(0, 24)]
labels = map(lambda x: str(x), numbers)
plt.xticks(numbers, labels)
plt.xlim(0, 24)
plt.hist(hours_count)# majority of purchases are made in the afternoon!


#Orders over time:
data_final['Order Date New'].hist(bins=50)
plt.show() # order spike in may and end of year

avg_viz = data_final
avg_viz['rolling 7'] = avg_viz['Large Purchase'].rolling(7).mean()
plt.figure(figsize=(15,6))
sns.lineplot(x='Order Day of Week',y='rolling 7' ,data=avg_viz)
plt.show()# it appears that the highest average of large purchases is on Wednesdays

data_final['Purchase State'].hist(bins=50)
plt.show()# california has the most orders

data_final['Large Purchase'].hist(bins=50)
plt.show() # we have about 40k large orders out of 180k

#To csv for observable:
data_final.to_csv(path+'data_final.csv')

data_final = pd.read_csv(path+'data_final.csv')
# TEST TRAIN SPLIT: REGRESSION
print(data_final.columns)
#['Order ID', 'Product', 'Quantity Ordered', 'Price Each', 'Order Date','Purchase Address', 'Purchase City',
# 'Purchase State', 'Order Date New','Order Time', 'Full Purchase', 'Large Purchase', 'Order Day of Week',
#'Order Day of Month', 'Order Month of Year', 'Order Hour of Day', 'Product Num', 'State Num', 'City Num']


X = data_final[['Order Day of Week', 'Order Day of Month', 'Order Month of Year', 'Order Hour of Day',
                'Product Num', 'State Num', 'City Num', 'Quantity Ordered']]

y = data_final['Full Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 21)
#ready for training model!

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)
# the regressor works!

prediction = regressor.predict(X_test)
mse = mean_squared_error(y_test, prediction)
rmse = mse**.5
print(rmse)


#TEST TRAIN SPLIT: CLASSIFICATION
X = data_final[['Order Day of Week', 'Order Day of Month', 'Order Month of Year', 'Order Hour of Day',
                'Product Num', 'State Num', 'City Num', 'Quantity Ordered']]

y = data_final['Large Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 21)
#ready for training model!

# try decision Tree First:
clf = DecisionTreeClassifier()
clf1 = clf.fit(X_train, y_train)
prediction2 = clf1.predict(X_test)

mse = mean_squared_error(y_test, prediction2)
rmse = mse**.5
print(rmse) # performs perfectly for some reason? something wrong here


# try K Nearest Neighbor next:
scaler1 = StandardScaler()
scaler1.fit(X_train.values)
X_train = scaler1.transform(X_train.values)
X_test = scaler1.transform(X_test.values)

knn = sklearn.neighbors.KNeighborsRegressor(
    n_neighbors = 3)
knn1 = knn.fit(X_train, y_train)
prediction3 = knn1.predict(X_test)

mse = mean_squared_error(y_test, prediction3)
rmse = mse**.5
print(rmse)
# ran when scaled! 0.40122955333439453


#NOTES:
# run with the now 12 months loaded DONE
#should I make the product name into a factor? There might be helpful ones to learn for the model?
# What am I trying to do here? Forecast with a time series right?
# How many purchases are estimated in a time?
# Cluster analysis, maybe what tends to get ordered around what time?
# Maybe look at what is forecasted to be ordered in a set of states in a period of time?
# The proximity, am I able to forecast what will be ordered next?
#-for outliers, should we remove outside those bounds?