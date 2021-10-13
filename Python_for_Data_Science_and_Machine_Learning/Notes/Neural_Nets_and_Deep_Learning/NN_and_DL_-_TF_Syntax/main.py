import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

# Read the (fake) data to DataFrame
df = pd.read_csv('data/fake_reg.csv')

# We create a Pair plot to see some information about the data
# Based on the plot we can see that 'feature 2' has very high correlation
# with the actual 'price'.
sns.pairplot(df)
plt.show()

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

# We can evaluate how well our Network does on data based on the metrics we've
# chosen ( in our case Mean Squared Error - MSE )
model.evaluate(x=X_train, y=y_train)
model.evaluate(x=X_test, y=y_test)

# To truly evaluate our models performance is through prediction.
test_predictions = model.predict(x=X_test)

# We can compare our prediction and the true values
test_predictions = pd.Series(test_predictions.reshape(300, ))
pred_df = pd.DataFrame(data=y_test, columns=['True Test Y'])
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['True Test Y', 'Model Predictions']

# If the prediction was 100% accurate we would have a perfect straight line
sns.scatterplot(data=pred_df, x='True Test Y', y='Model Predictions')
plt.show()

# Display some error metrics
print('Mean absolute error:')
print(
    mean_absolute_error(
        y_true=pred_df['True Test Y'],
        y_pred=pred_df['Model Predictions']
    )
)

print('Mean squared error:')
print(
    mean_squared_error(
        y_true=pred_df['True Test Y'],
        y_pred=pred_df['Model Predictions']
    )
)

print('Root mean squared error:')
print(
    mean_squared_error(
        y_true=pred_df['True Test Y'],
        y_pred=pred_df['Model Predictions']
    ) ** 0.5
)

# Predict on brand new data

# Pick new values
new_gem = [[998, 1000]]

# Scale those values
new_gem = scaler.transform(new_gem)

# Predict based on new values
model.predict(new_gem)

print(model.predict(new_gem))

# You can also save the model for later
# Commented out so it does not overwrite our good model
# model.save('my_gem_model.h5')

# Load model from save file
later_model = load_model('my_gem_model.h5')

# Test our save model once again

brand_new_gem = [[420, 690]]

# Scale those values
brand_new_gem = scaler.transform(brand_new_gem)

# Predict based on new values
print(later_model.predict(brand_new_gem))
