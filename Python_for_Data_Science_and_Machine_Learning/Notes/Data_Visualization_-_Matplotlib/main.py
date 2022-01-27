import os
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

    # The subplots call adds the number of subplots to the canvas
    fig, axes = plt.subplots(nrows=1, ncols=2)

    # We can iterate through axes
    # for current_ax in axes:
    #     current_ax.plot(x, y)

    # We can also index axes
    axes[0].plot(x, y)
    axes[0].set_title('First plot')

    axes[1].plot(y, x)
    axes[1].set_title('Second plot')

    # Fix overlapping
    plt.tight_layout()

    plt.show()

    # Figure Size
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), dpi=300)

    axes[0].plot(x, y)
    axes[1].plot(y, x)

    plt.tight_layout()

    plt.show()

    # Save figure
    fig.savefig(os.path.join(os.getcwd(), 'output', 'my_figure.jpg'))

    # Legends
    fig = plt.figure(figsize=(16, 10), dpi=300)

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Add legends for each plot
    ax.plot(x, x ** 2, label='X Squared')
    ax.plot(x, x ** 3, label='X Cubed')

    # Display the legends
    ax.legend(loc='best')

    plt.show()

    fig.savefig(
        os.path.join(os.getcwd(), 'output', 'my_figure_with_legend.jpg')
    )

    # Plot Appearance

    # Color
    fig = plt.figure(figsize=(16, 10), dpi=300)

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # We can specify the line color by simple color names like 'green' and
    # 'blue' or by RGB HEX Code #FF8C00
    ax.plot(x, y, color='#FF8C00')

    plt.show()

    # Line Width
    fig = plt.figure(figsize=(16, 10), dpi=300)

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # We can also specify the line width
    ax.plot(x, y, color='purple', linewidth=2)

    plt.show()

    # Visibility
    fig = plt.figure(figsize=(16, 10), dpi=300)

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # We can also specify the Line Width
    ax.plot(x, y, color='purple', linewidth=3, alpha=0.5)

    plt.show()

    # Line Style
    fig = plt.figure(figsize=(16, 10), dpi=300)

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # We can also specify the Line Style
    ax.plot(x, y, color='purple', linewidth=3, linestyle='-.')

    plt.show()

    # Markers
    fig = plt.figure(figsize=(16, 10), dpi=300)

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # We can also add markers to our plot
    ax.plot(
        x,
        y,
        color='purple',
        linewidth=2,
        linestyle='-',
        marker='o',
        markersize=10,
        markeredgewidth=2,
        markerfacecolor='yellow',
        markeredgecolor='green'
    )

    plt.show()

    # Axis appearance

    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.plot(
        x,
        y,
        color='purple',
        linewidth=2,
        linestyle='--',
    )

    # We can set limitations for our axis ranges
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 2])

    plt.show()
