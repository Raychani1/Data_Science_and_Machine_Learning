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

# Multi-Level Index

# Index levels
outside = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']
inside = [1, 2, 3, 1, 2, 3]

# list(zip(outside, inside)) command makes list of tuple pairs
# from the two lists
hier_index = list(zip(outside, inside))

# Create a Multi Index from a list of tuples
hier_index = pd.MultiIndex.from_tuples(hier_index)

df = pd.DataFrame(
    data=randn(6, 2),
    index=hier_index,
    columns=['A', 'B']
)

# Calling data from Multi Index DataFrame

print(df.loc['G1'].loc[1])

df.index.names = ['Groups', 'Num']

print(df)

print(df.loc['G2'].loc[2]['B'])
print(df.loc['G1'].loc[3]['A'])

# Cross Section can be used when we have a Multi-Level Index

# Specify which value you need, and which lever to search in.

print(df.xs(1, level='Num'))

# Missing Data

# Dropping Missing Data

# We can inport DataFrame from a dictionary where keys will be column names
d = {
    'A': [1, 2, np.nan],
    'B': [5, np.nan, np.nan],
    'C': [1, 2, 3]
}

df = pd.DataFrame(d)

# Every row/column with a single or more NaN value will be dropped.
# The row or column can be specified by the axis variable.
# This is by default not in place.
# With the thresh ( stands for threshold ) you can specify how many NaN values
# you want to keep.

df.dropna(axis=0)

# Replacing Missing Data

# The missing values can be filled using the fillna method.

df.fillna(value='Fill Value')

# Common strategy is to fill the missing value with the mean value of the column
df['A'].fillna(value=df['A'].mean())

# Group by

data = {
    'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
    'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
    'Sale': [200, 120, 340, 124, 243, 350]
}

df = pd.DataFrame(data)

# The groupby method creates a DataFrameGroupBy Object, and we can call
# aggregate functions on this object
by_comp = df.groupby(by='Company')

# The mean function returns the average sale ( since that is the only numerical
# column for which pandas is capable to calculate the mean value, and ignores
# all string data )
print(by_comp.mean())

# The sum function returns the sum of sales values ( since that is the only
# numerical column )
print(by_comp.sum())

# The std function returns the standard deviation of sales values ( since that
# is the only numerical column )
print(by_comp.std())

# Since we get back a DataFrame we can use indexing on it
print(by_comp.sum().loc['FB'])

# We can also use it as a one liner
print(df.groupby(by='Company').sum().loc['FB'])

# The count method is also useful, because it returns the number of instances
# of each column per Company ( The column we called group by on )
print(df.groupby(by='Company').count())

# The min ( max ) method returns the min ( max ) value for each column
# ( Even for string values)
print(df.groupby(by='Company').min())
print(df.groupby(by='Company').max())

# Also supports the describe method
print(df.groupby(by='Company').describe())

# Merging, Joining and Concatenating

df1 = pd.DataFrame(
    {
        'A': ['A0', 'A1', 'A2', 'A3'],
        'B': ['B0', 'B1', 'B2', 'B3'],
        'C': ['C0', 'C1', 'C2', 'C3'],
        'D': ['D0', 'D1', 'D2', 'D3']
    },
    index=[0, 1, 2, 3]
)

df2 = pd.DataFrame(
    {
        'A': ['A4', 'A5', 'A6', 'A7'],
        'B': ['B4', 'B5', 'B6', 'B7'],
        'C': ['C4', 'C5', 'C6', 'C7'],
        'D': ['D4', 'D5', 'D6', 'D7']
    },
    index=[4, 5, 6, 7]
)

df3 = pd.DataFrame(
    {'A': ['A8', 'A9', 'A10', 'A11'],
     'B': ['B8', 'B9', 'B10', 'B11'],
     'C': ['C8', 'C9', 'C10', 'C11'],
     'D': ['D8', 'D9', 'D10', 'D11']
     },
    index=[8, 9, 10, 11]
)

# Concatenating

# Basically glues together DataFrames, the dimensions should match along the
# axis we are concatenating on. We are using the concat method which takes a
# list of DataFrames. By default it concatenates on axis = 0.

print(pd.concat([df1, df2, df3]))

# When the dimensions don't match they are usually filled with NaN Values
print(pd.concat([df1, df2, df3], axis=1))

# Merging

# This operation is similar to SQL Join Functionality. Allows us to merge
# two tables, with given way and on specific column(s)

left = pd.DataFrame(
    {
        'key': ['K0', 'K1', 'K2', 'K3'],
        'A': ['A0', 'A1', 'A2', 'A3'],
        'B': ['B0', 'B1', 'B2', 'B3']
    }
)

right = pd.DataFrame(
    {
        'key': ['K0', 'K1', 'K2', 'K3'],
        'C': ['C0', 'C1', 'C2', 'C3'],
        'D': ['D0', 'D1', 'D2', 'D3']
    }
)

print(pd.merge(left, right, how='inner', on='key'))

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

print(pd.merge(left, right, how='inner', on=['key1', 'key2']))

# Joining

# Joins 2 dataframes based on the index ( row ) values.

left = pd.DataFrame(
    {
        'A': ['A0', 'A1', 'A2'],
        'B': ['B0', 'B1', 'B2']
    },
    index=['K0', 'K1', 'K2']
)

right = pd.DataFrame(
    {
        'C': ['C0', 'C2', 'C3'],
        'D': ['D0', 'D2', 'D3']
    },
    index=['K0', 'K2', 'K3']
)

print(left.join(right))

# We can also specify the way we want to join DataFrames

print(left.join(right, how='outer'))
