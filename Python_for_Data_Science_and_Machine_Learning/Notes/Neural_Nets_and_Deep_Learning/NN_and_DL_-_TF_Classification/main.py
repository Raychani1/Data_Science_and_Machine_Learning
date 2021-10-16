import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

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

# Overfitted Model

# Create the model
overfitted_model = Sequential()

# Most of the time we want to have the number of neurons based on the number of
# features we have.
print(X_train.shape)  # We have 19 features

# Hidden layers

# For a binary classification task you can have 1 Neuron in the output layer,
# but then you'll need a sigmoid activation function.
overfitted_model.add(Dense(30, activation='relu'))
overfitted_model.add(Dense(15, activation='relu'))
overfitted_model.add(Dense(1, activation='sigmoid'))

overfitted_model.compile(optimizer='adam', loss='binary_crossentropy')

overfitted_model.fit(x=X_train, y=y_train, epochs=600,
                     validation_data=(X_test, y_test))

plt.figure(figsize=(16, 9))
losses = pd.DataFrame(overfitted_model.history.history)
losses.plot()
plt.show()

# Early Stopping Model

# Create the model
early_stopping_model = Sequential()

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=25
)

early_stopping_model.add(Dense(30, activation='relu'))
early_stopping_model.add(Dense(15, activation='relu'))
early_stopping_model.add(Dense(1, activation='sigmoid'))

early_stopping_model.compile(optimizer='adam', loss='binary_crossentropy')

early_stopping_model.fit(
    x=X_train,
    y=y_train,
    epochs=600,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

plt.figure(figsize=(16, 9))
losses = pd.DataFrame(early_stopping_model.history.history)
losses.plot()
plt.show()

# Drop Out Layer + Early Stopping Model

# Create the model
drop_out_layer_model = Sequential()

drop_out_layer_model.add(Dense(30, activation='relu'))
drop_out_layer_model.add(Dropout(0.5))

drop_out_layer_model.add(Dense(15, activation='relu'))
drop_out_layer_model.add(Dropout(0.5))

drop_out_layer_model.add(Dense(1, activation='sigmoid'))

drop_out_layer_model.compile(optimizer='adam', loss='binary_crossentropy')

drop_out_layer_model.fit(
    x=X_train,
    y=y_train,
    epochs=600,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

plt.figure(figsize=(16, 9))
losses = pd.DataFrame(drop_out_layer_model.history.history)
losses.plot()
plt.show()

# Once our model(s) is/are trained we can predict with it/them
overfitted_prediction = overfitted_model.predict(X_test)
overfitted_prediction = \
    np.argmax(
        overfitted_prediction,
        axis=1
    ).reshape(-1, overfitted_prediction.shape[1])
print(overfitted_prediction)

early_stopping_prediction = early_stopping_model.predict(X_test)
early_stopping_prediction = \
    np.argmax(
        early_stopping_prediction, axis=1
    ).reshape(-1, overfitted_prediction.shape[1])
print(early_stopping_prediction)

drop_out_layer_prediction = drop_out_layer_model.predict(X_test)
drop_out_layer_prediction = \
    np.argmax(
        drop_out_layer_prediction,
        axis=1).reshape(-1, drop_out_layer_prediction.shape[1])
print(drop_out_layer_prediction)

print('Overfitted: '
      + classification_report(y_true=y_test,
                              y_pred=overfitted_prediction)
      )

print(confusion_matrix(y_true=y_test, y_pred=overfitted_prediction))

print('Early Stopping: '
      + classification_report(y_true=y_test,
                              y_pred=early_stopping_prediction)
      )

print(confusion_matrix(y_true=y_test, y_pred=early_stopping_prediction))

print('Dropout Layer: '
      + classification_report(y_true=y_test,
                              y_pred=drop_out_layer_prediction)
      )

print(confusion_matrix(y_true=y_test, y_pred=drop_out_layer_prediction))
