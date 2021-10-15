import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Read the data
df = pd.read_csv('data/cancer_classification.csv')

# Display some basic information and value distribution
print(df.info())
print(df.describe().transpose())

# Exploratory Data Analysis

# We can see that there are more malignant
sns.countplot(x='benign_0__mal_1', data=df)
plt.show()

# Sometimes it is nice to check the correlation of features
print(df.corr()['benign_0__mal_1'].sort_values())

# As you can see on the plot we've got highly negatively correlated features
# so basically we will be able to make strong predictions
plt.figure(figsize=(16, 9))
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
plt.show()

# Display a Heatmap of correlation
plt.figure(figsize=(16, 9))
sns.heatmap(df.corr())
plt.show()

# Separate the Label and the features
X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

# Split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101
)

# Next we scale our data
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
