"""
It estimates π given a number of iterations and random numbers in each iteration (size), taking the mean as the best estimative at the end.
"""

import numpy as np

iterations = 20
size = 10**8

accumulator = 0

for i in range(iterations):

    x = np.random.rand(size).astype(np.float32)
    y = np.random.rand(size).astype(np.float32)

    within_circle = ((x**2 + y**2) <= 1)
    circle = np.count_nonzero(within_circle)

    accumulator += circle

print(f'π estimative: {4*(accumulator/iterations)/size}')
