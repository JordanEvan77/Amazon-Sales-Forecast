#Data source; https://www.kaggle.com/datasets/knightbearr/sales-product-data?select=Sales_November_2019.csv
import pandas as pd
pd.options.mode.chained_assignment = None
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder




path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/5067 ML Code/DTC Final Project/'
# raw_data = pd.read_csv(path + 'Raw Data/Sales_September_2019.csv')

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

# remove repetition of column titles throughout data
data_prog = data_prog[data_prog["Product"].str.contains("Product")==False]



# feature extract address, to make it easier for categorical model use
data_prog[['Purchase Address', 'Purchase City', 'Purchase State']] = data_prog[
    'Purchase Address'].str.split(',', expand=True)

data_prog['Purchase State'] = data_prog['Purchase State'].apply(lambda x: str(x[:3]))

data_prog['Purchase State'] = data_prog['Purchase State'].astype('string')
data_prog['Purchase City'] = data_prog['Purchase City'].astype('string')
data_prog['Purchase Address'] = data_prog['Purchase Address'].astype('string')


# convert to date time
# data_prog['Order Date'] = datetime.strptime(data_prog['Order Date'], '%m/%d/%y ') #series error
# I want the time to be in a separate column.
data_prog['Order Date'] = pd.to_datetime(data_prog['Order Date'], infer_datetime_format=True)
data_prog['Order Date New'] = pd.to_datetime(data_prog['Order Date']).dt.date
print(data_prog['Order Date New'].head(5)) #looks good!

# Separate and Isolate
data_prog['Order Time'] = pd.to_datetime(data_prog['Order Date']).dt.time
print(data_prog['Order Time'].head(5))# looks good

# Feature extraction for Time Series Modeling:
data_prog['Order Day of Week'] = pd.to_datetime(data_prog['Order Date']).dt.dayofweek
data_prog['Order Day of Month'] = pd.to_datetime(data_prog['Order Date']).dt.day
data_prog['Order Day of Year'] = pd.to_datetime(data_prog['Order Date']).dt.dayofyear
data_prog['Order Month of Year'] = pd.to_datetime(data_prog['Order Date']).dt.month
data_prog['Order Hour of Day'] = pd.to_datetime(data_prog['Order Date']).dt.hour


# data_prog['Order Time'] = pd.to_datetime(data_prog['Order Time'], format='%H:%M').dt.time
# print(data_prog['Order Time'].head(5)) #should I leave this out?

# Numeric Conversions
data_prog['Price Each'] = pd.to_numeric(data_prog['Price Each'])
data_prog['Quantity Ordered'] = pd.to_numeric(data_prog['Quantity Ordered'])

# create new variable "Full Purchase" that gauges amount fully paid
data_prog["Full Purchase"] = data_prog['Price Each'] * data_prog['Quantity Ordered']
print(statistics.median(data_prog["Full Purchase"]))
print(statistics.mean(data_prog["Full Purchase"]))
# anything over 185 could be flagged as a large purchase:

data_prog['Large Purchase'] = (data_prog["Full Purchase"] > 185).astype(int)
print(data_prog['Large Purchase'])

# keep looking at visuals, begin looking for missing values, outliers, etc.

# NULL VALUES:
# do the counts per first:
data_prog.isna().sum()
# it appears we have no missing data!


# OUTLIERS PER ATTRIBUTE:
# show visuals, then look at outside 2.5 SD.
data_prog['Price Each'].hist(bins=50) # as expected, most items are cheaper
plt.show()# far right price may be an outlier

data_prog['Full Purchase'].hist(bins=50) # as expected, most items are cheaper
plt.show()# far right price may be an outlier

data_prog['Quantity Ordered'].hist(bins=50) # as expected, most items are cheaper
plt.show()# mostly orders of single value


print(data_prog.describe())
# there are definitely outliers. Using Interquartile Range to gather them:
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
# now I can decide what to do with these values
# Decided to keep outliers as they are large orders, which are important and logical occurences for
# The time series modeling.

################
# Create Dummies!
#################

labelencoder = LabelEncoder()
#'Product'
data_prog['Product Num'] = labelencoder.fit_transform(data_prog['Product'])
#'Purchase State'
data_prog['State Num'] = labelencoder.fit_transform(data_prog['Purchase State'])
#'Purchase City'
data_prog['City Num'] = labelencoder.fit_transform(data_prog['Purchase City'])

#Duplicates
print(data_prog[data_prog.duplicated()])
#interesting, look at this and see if I want to remove duplicates
#if so, use drop_duplicates()

data_final = data_prog.drop_duplicates(keep="last")


#######################
#To csv for observable:
#######################
data_final.to_csv(path+'data_final.csv')