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
