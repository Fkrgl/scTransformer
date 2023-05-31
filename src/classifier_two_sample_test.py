from metrics import *
import torch

# load tensors
path_input = '../data/val_input_50_epochs.pt'
path_reconstruction = '../data/reconstructed_profiles_50_epochs.pt'
X = torch.load(path_input).type(torch.DoubleTensor)
Y = torch.load(path_reconstruction).type(torch.DoubleTensor)
print(X[0])
print(Y[0])
print(f'input shape: {X.shape}')
print(torch.isnan(X))
print(f'reconstruction shape: {Y.shape}')

accuracy = c2st(X, Y)
print(f'accuracy of classifier two sample test: {accuracy}')