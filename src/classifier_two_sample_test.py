from metrics import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
print(Y[zeros])
#plt.hist(Y[zeros].detach(), bins=30)
# plt.hist(mask_diff.detach(), color='blue', density=True, edgecolor='black', alpha=0.5, label='masked', bins=25)
# plt.hist(not_mask_diff.detach(), color='orange', density=True, edgecolor='black', alpha=0.5, label='unmasked', bins=25)
# plt.legend()
print(X[:,1].shape)
X = X.detach().numpy()
Y = Y.detach().numpy()
mask = mask.detach().numpy()
# get difference
D = Y - X
var = np.var(D, axis=0)
sorted_var = sorted(var.tolist(), reverse=True)
print(sorted_var)
print(np.where(sorted_var[0] == var)[0][0])
gene_80_mask = mask[:,80]
X_80_masked = X[gene_80_mask, 80]
Y_80_masked = Y[gene_80_mask, 80]
print(X_80_masked.shape, Y_80_masked.shape)

mean = np.mean(D, axis=0)
sorted_mean = sorted(mean.tolist(), reverse=True)

print(mask.shape)
n_masked = mask.sum(axis=0)
masked_mean_difference = []
for i in range(mask.shape[1]):
    curr_mask = mask[:,i]
    print(i)
    diff_gene = ((X[curr_mask, i] - Y[curr_mask, i])**2).sum() / n_masked[i]
    print(diff_gene)
    masked_mean_difference.append(diff_gene)

print(masked_mean_difference)
print(mask.shape[1])
sorted_mean = sorted(masked_mean_difference, reverse=True)
print(sorted_mean)
# plt.hist(X_80_masked, color='blue', edgecolor='black', alpha=0.5, label='true expression', bins=20)
# plt.hist(Y_80_masked, color='red', edgecolor='black', alpha=0.5, label='predicted expression', bins=20)
# plt.title('Expression of gene 80 over cells where gene is masked in validation set')
# plt.xlabel('expression value')
# plt.ylabel('frequency')
# plt.legend()
#plt.show()

def plot_ten_most_variant(X, Y, mask, var, sorted_var):
    n_row = 2
    n_col = 5
    k = 0
    fig, axs = plt.subplots(n_row, n_col, figsize=(15,5), sharex=True, sharey=True)
    for i in range(n_row):
        for j in range(n_col):
            gene_var = sorted_var[k]
            gene_idx = np.where(gene_var == var)[0][0]
            gene_mask = mask[:, gene_idx]
            X_masked = X[gene_mask, gene_idx]
            Y_masked = Y[gene_mask, gene_idx]
            bins = np.histogram(np.hstack((X_masked, Y_masked)), bins=10)[1]
            axs[i,j].hist(X_masked, color='blue', edgecolor='black', alpha=0.5, label='true expression', bins=bins,
                          density=True)
            axs[i,j].hist(Y_masked, color='red', edgecolor='black', alpha=0.5, label='predicted expression', bins=bins,
                          density=True)
            k += 1
    blue_patch = mpatches.Patch(color='blue', label='true')
    red_patch = mpatches.Patch(color='red', label='predicted')
    fig.legend(handles=[blue_patch, red_patch])
    fig.suptitle('predicted and true expression values of top 10 worst predicted genes')
    fig.supylabel('relative frequency')
    fig.supxlabel('expression value')
    plt.show()


def plot_ten_most_different(X, Y, mask, mean,sorted_mean):
    n_row = 2
    n_col = 5
    k = 0
    fig, axs = plt.subplots(n_row, n_col, figsize=(15,5), sharex=True, sharey=True)
    for i in range(n_row):
        for j in range(n_col):
            gene_mean = sorted_mean[k]
            gene_idx = np.where(gene_mean == mean)[0][0]
            gene_mask = mask[:, gene_idx]
            X_masked = X[gene_mask, gene_idx]
            Y_masked = Y[gene_mask, gene_idx]
            bins = np.histogram(np.hstack((X_masked, Y_masked)), bins=10)[1]
            axs[i,j].hist(X_masked, color='blue', edgecolor='black', alpha=0.5, label='true expression', bins=bins,
                          density=True)
            axs[i,j].hist(Y_masked, color='red', edgecolor='black', alpha=0.5, label='predicted expression', bins=bins,
                          density=True)
            k += 1
    blue_patch = mpatches.Patch(color='blue', label='true')
    red_patch = mpatches.Patch(color='red', label='predicted')
    fig.legend(handles=[blue_patch, red_patch])
    fig.suptitle('predicted and true expression values of top 10 worst predicted genes')
    fig.supylabel('relative frequency')
    fig.supxlabel('expression value')
    plt.show()


plot_ten_most_different(X, Y, mask, masked_mean_difference,sorted_mean)