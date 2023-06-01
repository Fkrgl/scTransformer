from metrics import *
import torch
import numpy as np
import matplotlib.pyplot as plt

# load tensors
path_input = '../data/val_input_50_epochs.pt'
path_reconstruction = '../data/reconstructed_profiles_50_epochs.pt'
path_mask = '../data/masks_50_epochs.pt'
X = torch.load(path_input).type(torch.DoubleTensor)
Y = torch.load(path_reconstruction).type(torch.DoubleTensor)
mask = torch.load(path_mask)
zeros = torch.where(X == 0)
#X[zeros[0], zeros[1]] += 10
# for i,j in zip(zeros[0], zeros[1]):
#     X[i.item(), j.item()] = np.random.rand(1)[0] / 100
# accuracy = c2st(X, Y)
# print(f'accuracy of classifier two sample test: {accuracy}')

# idea: are the values at the masked positions more similar compared to the unmasked ones?
print(mask.shape)
print(Y.shape)
mask_diff = Y[mask] - X[mask]
not_mask_diff = Y[~mask] - X[~mask]
print(Y[mask].shape)
print(mask_diff)
plt.hist(mask_diff.detach(), color='blue', density=True, edgecolor='black', alpha=0.5, label='masked', bins=20)
plt.hist(not_mask_diff.detach(), color='orange', density=True, edgecolor='black', alpha=0.5, label='unmasked', bins=20)
plt.legend()
plt.show()