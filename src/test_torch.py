import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


a = np.array([[1,2,3], [4,5,6], [7,8,9]])

print(f'a normal:\n{a}')

mask = np.array([[True, False, True], [True, False, True], [True, False, True]])
print(f'mask:\n{mask}')

random = np.random.choice(np.arange(100), size=(3,3))
print(f'random:\n{random}')
a[mask] = random[mask]
print(f'a modified:\n{a}')

a[mask] = 0
print(f'a zero:\n{a}')