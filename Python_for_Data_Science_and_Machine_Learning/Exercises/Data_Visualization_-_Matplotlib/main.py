import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.arange(0, 100)
    y = x * 2
    z = x ** 2

    # Quick disclaimer the max() + some value is only added to show the exact
    # upper bound, because otherwise we would have there some empty space

    # Exercise 1

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.plot(x, y, color='blue')
    ax.set_xlim([0, x.max() + 1])
    ax.set_ylim([0, y.max() + 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')

    plt.show()

    # Exercise 2

    fig = plt.figure(figsize=(16, 9))

    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax2 = fig.add_axes([0.2, 0.5, 0.2, 0.2])

    for ax in [ax1, ax2]:
        ax.plot(x, y, color='red', alpha=0.5)
        ax.set_xlim([0, x.max() + 1])
        ax.set_ylim([0, y.max() + 2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.show()

    # Exercise 3

    fig = plt.figure(figsize=(16, 9))

    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax2 = fig.add_axes([0.18, 0.47, 0.4, 0.4])

    ax1.plot(x, z, color='blue', alpha=0.5)
    ax1.set_xlim([0, x.max() + 1])
    ax1.set_ylim([0, z.max() + 200])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')

    ax2.plot(x, y, color='blue', alpha=0.5)
    ax2.set_xlim([20, 22])
    ax2.set_ylim([30, 50])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('zoom')

    plt.show()

    # Exercise 4

    fig, axes = plt.subplots(figsize=(8, 2), nrows=1, ncols=2)

    axes[0].plot(x, y, color='blue', linewidth=3, linestyle='-', alpha=0.5)
    axes[0].set_xlim([0, x.max() + 1])
    axes[0].set_ylim([0, y.max() + 2])

    axes[1].plot(x, z, color='red', linewidth=3, linestyle='--', alpha=0.5)
    axes[1].set_xlim([0, x.max() + 1])
    axes[1].set_ylim([0, z.max() + 200])

    plt.tight_layout()

    plt.show()
