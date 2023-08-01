import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


a = np.array([[1,2,3], [4,5,6], [7,8,9]])
b = a
b[1,1] = 0
print(a)
print(b)
mask = np.array([[True, False, True], [True, False, True], [True, False, True]])
print(a)
print(a[mask])

random = np.random.choice(np.arange(100), size=(3,3))
print(random)
a[mask] = random[mask]
print(a)