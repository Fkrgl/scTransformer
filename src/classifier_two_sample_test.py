from metrics import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# load tensors
path_original = '../data/pancreas_binned_original.npy'
path_generated = '../data/pancreas_binned_generated.npy'
original = np.load(path_original)
generated = np.load(path_generated)
original = torch.tensor(original).type(torch.DoubleTensor)
generated = torch.tensor(generated).type(torch.DoubleTensor)

# n = 2500
# x, y = torch.randn(n, 5), torch.randn(n, 5)

accuracy = c2st(original, generated)
print(f'accuracy of classifier two sample test: {accuracy}')

