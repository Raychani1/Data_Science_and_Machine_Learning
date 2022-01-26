import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Generate some data
    x = np.linspace(0, 5, 11)
    y = x ** 2

    # Functional Plotting Method
    plt.plot(x, y)  # Plot the data
    plt.xlabel('X label')  # Set the X label
    plt.ylabel('Y label')  # Set the Y label
    plt.title('Title')  # Set the Title of the Plot

    # If we are not using the Jupyter Notebooks we need this method to render
    # our plots
    plt.show()

    # Creating subplots

    # plt.subplot(number_of_rows, number_of_columns, referring_plot_number)

    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'r')

    plt.subplot(1, 2, 2)
    plt.plot(y, x, 'b')

    plt.show()

    # Object-Oriented Plotting Method

    # Create a figure object (like a blank canvas)
    fig = plt.figure()

    # Add axes to be able to plot on it

    # fig.add_axes(
    #     [
    #         left_of_the_axes,
    #         bottom_of_the_axes,
    #         width_of_the_axes,
    #         height_of_the_axes
    #     ]
    # )

    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Plot the data
    axes.plot(x, y)

    # Set the X Label
    axes.set_xlabel('X Label')

    # Set the Y Label
    axes.set_ylabel('Y Label')

    # Set the Title
    axes.set_title('Title')

    plt.show()

    # Creating subplots the Object-Oriented way

    fig = plt.figure()

    axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])

    axes1.plot(x, y)
    axes1.set_title('Larger Plot')

    axes2.plot(y, x)
    axes2.set_title('Smaller Plot')

    plt.show()
