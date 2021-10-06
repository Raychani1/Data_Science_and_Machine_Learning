import numpy as np
import pandas as pd
from numpy.random import randn

# Set the same seed to have the same output as in the tutorial
np.random.seed(101)

# DataFrame is basically a bunch of Series ( the columns )
# sharing the same index
df = pd.DataFrame(
    data=randn(5, 4),
    index=['A', 'B', 'C', 'D', 'E'],
    columns=['W', 'X', 'Y', 'Z']
)

# Select a column - returns Series
print(df['W'])

# The proof that each column is a Pandas Series
print(type(df['W']))

# Select multiple columns - returns DataFrame
print(df[['X', 'Y', 'Z']])

# New column creation
df['new'] = df['W'] + df['Y']

# Drop column
# Axis 0 = ROW
# Axis 1 = COLUMN
# This process by default is not happening inline
df.drop('new', axis=1, inplace=True)

# Drop row
df.drop('E', axis=0)

# Why are Rows marked as 0 and Columns as 1?
# If we call

print(df.shape)

# It returns us a tuple, where the first one ( on the index 0 ) represents rows
# while the second one ( on the index 1 ) represents columns.

# Selecting rows - 2 way

# Select based on the label of the index - returns Series
# This is the proof that not only the columns but even the rows are Series
print(df.loc['A'])

# Select based on the index number - returns Series
print(df.iloc[0])

# Selecting subset of rows and columns
print(df.loc['B', 'Y'])
print(df.loc[['A', 'B'], ['W', 'Y']])

# Conditional selection
booldf = df > 0

# Contains NaN where the condition is False
print(df[booldf])

# If you specify the column no NaN values will be present
# Returns DataFrame with rows where the condition is True
print(df[df['W'] > 0])

print(df[df['Z'] < 0])

# Select the specific column(s) from the sub set of a DataFrame
print(df[df['W'] > 0]['X'])

print(df[df['W'] > 0][['X', 'Y']])

# Multiple Conditional Select

# You have to use the & / | symbols instead of and / or operators
# Because you get a Series of those values instead of two values
print(df[(df['W'] > 0) & (df['Y'] > 1)])

# Index

# Resetting index

# The reset_index() method will reset the index to numerical index, and the
# previous index will be saved as a separate column called index.
# This is by default not in place.

df.reset_index()


# Setting to something else

# The set_index() method will set the index to a specified column values.
# This is by default not in place.

new_ind = 'CA NY WY OR CO'.split()

df['State'] = new_ind

df.set_index('State')
