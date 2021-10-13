import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Read the (fake) data to DataFrame
df = pd.read_csv('data/fake_reg.csv')

# We create a Pair plot to see some information about the data
# Based on the plot we can see that 'feature 2' has very high correlation
# with the actual 'price'.
sns.pairplot(df)
# plt.show()

# Grab the features ( based on which we will predict )

# We need to send in NumPy Array instead of Pandas DataFrame so we add .values
# By convention we mark it as capital X, because the Feature array is
# 2 dimensional.
X = df[['feature1', 'feature2']].values

# Grab the label ( which we will predict )

# By convention we mark it as lowercase y, because it is only 1 dimensional
# array.
y = df['price'].values

# We split our data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Data Scaling and Normalization

# Since we are working with weights and biases inside of a Neural Network
# really large values in the feature set that can cause error in the weights.
# We are going to use the MinMaxScaler from Scikit Learn.
# It's going to transform the data based on the Standard Deviation of the data
# as well as the Min and Max values.
# We only need to scale the features, since that will passed through
# the Network.

# To use a scaler we first create an instance of it.
scaler = MinMaxScaler()

# Once the instance is created we need to fit it to our training data.
# Calculates the Standard Deviation, Min and Max values for this set.

# The reason we only run it on the training set is because we want to prevent
# Data Leakage from the test set. We don't want to have prior knowledge what
# we have in the test set.
scaler.fit(X_train)

# Once the fit process is done, we can transform the actual transformation of
# our train set and our test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# To build a very simple model in Keras we need a base Sequential model and
# add layers to it.
# In this example we are using a Sequential model with Dense layer.
# Dense layers are used to create simple Feed forward densely connected
# ( every neuron is connected with every neuron in the next layer ).

# There are two methods of defining a Network

# Method 1
# model = Sequential([
#     Dense(units=4, activation='relu'),
#     Dense(units=4, activation='relu'),
#     Dense(units=4, activation='relu'),
#     Dense(units=1)
# ])

# Method 2
# This one is better if we want to edit the model in the future and turn off
# some layers.
model = Sequential()
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=1))

# We define the optimizer, which function we want to use for the gradient
# descent ( when we are looking for the weights, which produce the minimal
# loss ). The loss parameter will represent the loss function and it will
# depend on the given task.
model.compile(optimizer='rmsprop', loss='mse')

# Once the model is created we can fit it to the training set.
model.fit(x=X_train, y=y_train, epochs=250)

# The model.history.history returns the historical loss values. We can plot that
# out.
loss_df = pd.DataFrame(data=model.history.history)

loss_df.plot()
plt.show()
