import torch
import numpy as np

rand = np.random.choice(np.arange(100), size=(3,10))
print(torch.tensor(rand))
print(np.arange(100))

a = np.arange(10)
b = np.zeros(shape=10, dtype=bool)

b[2] = True
b[5] = True
b[7] = True
print(b)

