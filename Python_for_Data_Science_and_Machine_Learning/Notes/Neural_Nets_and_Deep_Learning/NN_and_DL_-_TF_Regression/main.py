import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from tensorflow.keras.models import load_model

# Read the data to DataFrame
df = pd.read_csv('data/kc_house_data.csv')

# Check for NaN values
print(f'Number of NaN values in columns:\n{df.isnull().sum()}')

# Check on some information
description = df.describe().transpose()
print(description)

# Visualization

# Check the distribution of the label ( what we want to predict )
# As you can see on the plot there are some houses that are way expensive
# and those houses might lower our prediction accuracy, so we might think about
# dropping them.
plt.figure(figsize=(10, 6))
sns.displot(df['price'])
# plt.show()

# As you can see on the count plot some houses have > 7 bedrooms
# ( some have 33 ), which is exactly the same scenario as with the very
# expensive houses.
plt.figure(figsize=(10, 6))
sns.countplot(df['bedrooms'])
# plt.show()

# We can look for correlation in the data. Since we are looking for factors that
# are correlated with our label 'price', we can call the following command.
# We can see that the most correlation is with price ( obviously ), but the
# second most correlated feature is 'sqft_living'.
print(df.corr()['price'].sort_values())

# We can inspect highly correlated features through a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='sqft_living', data=df)
plt.show()

# This boxplot shows us the large variety of prices based on the number of
# bedrooms.
plt.figure(figsize=(10, 6))
sns.boxplot(x='bedrooms', y='price', data=df)
plt.show()

# We can see that on Longitude -122.2 there is some kind of expensive housing
# area.
plt.figure(figsize=(12, 8))
sns.scatterplot(x='price', y='long', data=df)
plt.show()

# We can also see that at some areas are expensive housing areas as well.
plt.figure(figsize=(12, 8))
sns.scatterplot(x='price', y='lat', data=df)
plt.show()

# We can make a combination of Longitude and Latitude to discover expensive area
plt.figure(figsize=(16, 9))
sns.scatterplot(x='long', y='lat', data=df, hue='price')
plt.show()

# The following command drops the top 1% ( the most expensive houses )
non_top_1_percent = df.sort_values('price', ascending=False).iloc[216:]

# As you can see on the following plot properties next to the water seem to have
# higher price
plt.figure(figsize=(16, 9))
sns.scatterplot(
    x='long',
    y='lat',
    data=non_top_1_percent,
    hue='price',
    edgecolor=None,
    alpha=0.2,
    palette='RdYlGn'
)
plt.show()

# As you can see on this box plot if you have a waterfront property it tends to
# have higher value
sns.boxplot(data=df, x='waterfront', y='price')
plt.show()