import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
import matplotlib.pyplot as plt
path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/5067 ML Code/DTC Final Project/'

data_prog = pd.read_csv(path+'data_final.csv')
data_prog['Order Time'] = pd.to_datetime(data_prog['Order Date']).dt.time
print(data_prog.head(5))

##################
# Further Visuals:
##################

# looking at plain count of orders per hour of the day
hours_count = [j.hour for j in data_prog['Order Time']]
numbers = [x for x in range(0, 24)]
labels = map(lambda x: str(x), numbers)
plt.xticks(numbers, labels)
plt.xlim(0, 24)
plt.hist(hours_count) # majority of purchases are made in the afternoon!


# Orders over time:
data_prog['Order Date New'].hist(bins=50)
plt.show() # order spike in may and end of year

# rolling average visual of largest purchases
avg_viz = data_prog
avg_viz['rolling 7'] = avg_viz['Large Purchase'].rolling(7).mean()
plt.figure(figsize=(15,6))
sns.lineplot(x='Order Day of Week',y='rolling 7' ,data=avg_viz)
plt.show() # it appears that the highest average of large purchases is on Wednesdays

# histogram of orders per state
data_prog['Purchase State'].hist(bins=50)
plt.show() # california has the most orders

# histogram of count of large or normal purchases
data_prog['Large Purchase'].hist(bins=50)
plt.show() # I have about 40k large orders out of 180k

# Grouping of large purchases in Top states over time:
state_group = pd.DataFrame(
    data_prog.groupby(["Order Month of Year", "Purchase State"], as_index=False)["Large "
                                                                                "Purchase"].count())
state_group['Purchase State'] = state_group['Purchase State'].astype('string')

top_states = state_group.sort_values("Large Purchase", ascending=False).groupby(
    ["Order Month of Year"]).head(2)

sns.barplot(x="Order Month of Year", y = "Large Purchase", hue="Purchase State",  data=top_states)
# looks good!


# Group by Time of day, full purchase amount, large purchases only
day_time = pd.DataFrame(
    data_prog.groupby(["Order Hour of Day", "Large Purchase"], as_index=False)[
        "Full Purchase"].sum())

sns.barplot(x="Order Hour of Day", y = "Full Purchase", hue="Large Purchase",  data=day_time)
# this looks extremely interesting, this is in millions
# large purchases are a very important of our daily cycle of purchases

# Scatter plot of cities, size is purchase amount, day and month of year.
# need to find Top purchasing city for every day of month and month of year.
city_day = pd.DataFrame(
    data_prog.groupby(["Order Month of Year", "Order Day of Month", "Purchase City"],
                      as_index=False)["Full Purchase"].sum())

# now I want to find for every city, what their largest purchase month and day is.

city_day_top = city_day.sort_values("Full Purchase", ascending=False).groupby('Order Day of '
                                                                              'Month').head(3)

sns.scatterplot(x="Order Day of Month", y="Order Month of Year",data=city_day_top)
# extremely useful visual with hover labels

print('Done')
