import numpy as np
import matplotlib.pyplot as plt
import time

neighbors_offset = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
i = 13
j = 10
for di, dj in neighbors_offset:
    ni, nj = i + di, j + dj
    print(ni)
    print(nj)
    print('/n')

