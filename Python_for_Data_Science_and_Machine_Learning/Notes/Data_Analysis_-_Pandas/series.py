import numpy as np
import pandas as pd

# The Pandas Series is a similar object like a NumPy Array
# It differentiates from NumPy Array in Axis labels, it can be indexed by label

labels = ['a', 'b', 'c']

my_data = [10, 20, 30]

arr = np.array(my_data)

d = {'a': 10, 'b': 20, 'c': 30}

# Creating a Series from list
series = pd.Series(data=my_data, index=labels)

# Creating a Series from NumPy Array
series2 = pd.Series(data=arr, index=labels)

# Creating a Series from dictionary
series3 = pd.Series(data=d)

# You can store even built-in functions
series4 = pd.Series(data=[sum, print, len])

# Indexing

ser1 = pd.Series(data=[1, 2, 3, 4], index=['USA', 'Germany', 'USSR', 'Japan'])
ser2 = pd.Series(data=[1, 2, 5, 4], index=['USA', 'Germany', 'Italy', 'Japan'])

# Grabbing information from a specific index works like in a dictionary
print(ser1['USA'])


ser3 = pd.Series(data=labels)

print(ser3[0])

# Operations

# The +/- operators will try to match values based on the index, if no match
# found NaN object is added. After any operation integers will be converted to
# floats.

print(ser1 + ser2)
