import numpy as np

if __name__ == '__main__':
    # Cast a Python list to NumPy Array ( One dimensional NumPy Array - Vector )
    my_list = [1, 2, 3]

    arr = np.array(my_list)

    # Cast a Python list of lists to NumPy Array
    # ( Two dimensional NumPy Array - Matrix )
    my_mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    mat = np.array(my_mat)

    # Easy way to create a NumPy Array with the range of values
    # ( Up to but not including )

    my_range = np.arange(start=0, stop=11, step=2)

    # Generate zeros / ones
    # ( Single digit - Vector / Tuple of numbers ( Rows, Columns ) - Matrix )

    ones = np.ones(shape=(6, 5))
    zeros = np.zeros(shape=(6, 5))

    # np.linspace returns evenly spaced numbers over a specified interval
    # in a Vector ( 1D array )

    lin = np.linspace(start=0, stop=5, num=100)

    # Using the np.eye method we can create an Identity Matrix
    # Identity Matrix is a Matrix which has only ones on the main diagonal
    # and zeros anywhere else.

    eye = np.eye(N=9)

    # Random Numbers

    # Using the np.random.rand() we can create an array of a given shape
    # and populate it with random numbers from the Uniform distribution
    # over 0 to 1

    rand1 = np.random.rand(9)  # 1D
    rand2 = np.random.rand(9, 9)  # 2D

    # If we want to return a sample ( or many samples ) from the Standard
    # Normal ( Gaussian ) distribution we can use np.random.randn()

    randn = np.random.randn(9, 9)

    # To return random integers from a low to a high number
    # ( up to but not including ) we can use the np.random.randint() method

    randint = np.random.randint(low=0, high=100, size=(5, 5))

    arr2 = np.arange(25)
    ranarr = np.random.randint(0, 50, 10)

    # The .reshape() method will return an array with the same data but in a
    # different shape.

    # Quick check Number of rows x Number of Columns = Number of Elements

    arr2 = arr2.reshape(5, 5)

    # Finding the min/max value of an array using the min/max method
    ranarr.max()

    # Find the index of min/max value of an array using the argmin/argmax method
    print(f'Randint Array: {ranarr}\n'
          f'Min: {ranarr.min()}\n'
          f'Max: {ranarr.max()}\n'
          f'Index of Min: {ranarr.argmin()}\n'
          f'Index of Max: {ranarr.argmax()}')

    # The shape attribute will return the shape of an array
    print(ranarr.shape)

    # The dtype attribute returns the datatype stored in the array
    print(ranarr.dtype)

    # Indexing and Selection

    arr3 = np.arange(0, 11)

    # From index 0 up to but not including 5
    print(arr3[:5])

    # From index 5 to the end
    print(arr3[5:])

    # Broadcasting - Assign a value to a subset of an array
    arr3[:5] = 100

    print(arr3)

    # Changes the original array
    slice_of_arr3 = arr3[:6]
    slice_of_arr3[:] = 99

    print(slice_of_arr3)

    # The copy method creates a copy of an array so it is not changed in the
    # process
    array_copy = arr3.copy()
    array_copy[:] = 50

    print(f'\nOriginal array {arr3}\nCopy of array {array_copy}\n')

    arr_2d = np.array(
        [
            [5, 10, 15],
            [20, 25, 30],
            [35, 40, 45]
        ]
    )

    # Double bracket notation
    print(arr_2d[0][0])

    # Comma notation
    print(arr_2d[0, 2])

    # 2D array slicing
    print(arr_2d[1:, :])

    # Conditional selection

    new_array = np.arange(1, 11)

    bool_array = new_array[new_array > 5]

    print(bool_array)

    # Operations

    arr4 = np.arange(1, 11)

    print(arr4 + arr4)

    print(arr4 - arr4)

    print(arr4 * arr4)

    print(arr4 ** arr4)

    # Scalers are the numbers after the operation sign which should be
    # broadcasted on a given array

    print(arr4 + 100)

    print(arr4 - 100)

    print(arr4 * 100)

    print(arr4 ** 3)

    # If you have an operation which results in division by zero
    # NumPy won't throw an error it will put show a warning and put:
    # NaN if the operation was 0 / 0
    # Inf if the operation was any number / 0

    arr5 = np.arange(10)

    print(arr5/arr5)

    print(arr4/arr5)

    # Universal Array Functions

    # Get the square root of every element in the NumPy Array

    print(np.sqrt(arr5))

    # Calculate the exponential value of every element in the NumPy Array

    print(np.exp(arr4))

    # Get the min/max value from NumPy Array
    # ( The same as .min()/.max() )

    print(np.min(arr4))
    print(np.max(arr4))

    # Pass every element to Sine

    print(np.sin(arr))

    # Pass every element to Cosine

    print(np.cos(arr))

    # Pass every element to Logarithm
    # If you pass an Array with 0 in it, the method will return -Inf and
    # RuntimeWarning
    
    print(np.log(arr))

