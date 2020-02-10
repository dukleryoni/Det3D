import numpy as np
import matplotlib.pyplot as plt
import fire

# run from command line


def plot_method(A=2):
    if A == 2:
        plt.plot([1], [1])
        plt.show()
    else:
        plt.plot(np.arange(10))
        plt.show()


if __name__ == "__main__":
    fire.Fire()

# works from docker, possibly from jupyter too
