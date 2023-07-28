import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scvelo as scv
from preprocessor import Preprocessor

class scDataSet(Dataset):
    def __init__(self,
                 data,
                 n_non_zero_bins: int,
                 n_tokens: int
                 ):
        self.n_non_zero_bins = n_non_zero_bins
        self.n_tokens = n_tokens
        # load data
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get sample
        sample = torch.tensor(self.data[idx])
        # generate mask for sample
        mask = self.get_balanced_mask(sample)
        return sample, mask

    # def get_prob_mask(self, expressions: torch.Tensor) -> torch.Tensor:
    #     """
    #     generates a mask for a proportion of genes in the input data. The masks genes are predicted later in the training
    #     process. More information here:
    #     https://github.com/pytorch/pytorch/blob/11f1014c05b902d3eef0fe01a7c432f818c2bdfe/torch/nn/functional.py#L3892
    #     Args:
    #         expressions: expression matrix (batch, seq_length)
    #         mlm_probability: probability fo a gene to get masked
    #
    #     Returns:
    #         Boolean Tensor of shape (batch, n_token) with True where genes should be masked
    #
    #     """
    #     shape = expressions.shape
    #     probability_matrix = torch.full(shape, self.mlm_probability)
    #     mask = torch.bernoulli(probability_matrix).bool()
    #     return mask

    def get_balanced_mask(self, sample):
        if self.n_non_zero_bins == 1:
            mask = np.zeros(self.n_tokens, dtype=bool)
            idx = np.arange(self.n_tokens)
            rand_index = np.random.choice(idx, 1, replace=False)
            mask[rand_index] = True

        else:
            mask = np.zeros(self.n_tokens, dtype=bool)
            n_non_zeros = np.count_nonzero(sample)
            # trim n_non_zeros if its exceeds the maximal number of masked genes (=2*self.n_non_zero_bins)
            if n_non_zeros > 2*self.n_non_zero_bins:
                # in this case all masked genes have a non zero bin value
                n_non_zeros = 2*self.n_non_zero_bins
            #print(f'number of non zero bins: {n_non_zeros}')
            n_zeros = self.n_non_zero_bins
            # less non zero bins than average
            if n_non_zeros < self.n_non_zero_bins:
                diff = self.n_non_zero_bins - n_non_zeros
                # print(f'less non zero genes. diff={diff}')
                # print(f'n_zeros = {self.n_non_zero_bins} + {diff}')
                n_zeros = self.n_non_zero_bins + diff
            # more non-zero bins than average
            elif n_non_zeros > self.n_non_zero_bins:
                diff = n_non_zeros - self.n_non_zero_bins
                # print(f'more non zero genes. diff={diff}')
                # print(f'n_zeros = {self.n_non_zero_bins} - {diff}')
                n_zeros = self.n_non_zero_bins - diff
            # sample indeces
            # print(f'n_zeros={n_zeros}')
            idx = np.arange(self.n_tokens)
            idx_zero = idx[sample == 0]
            idx_non_zero = idx[sample != 0]
            # randomly select genes to be masked
            idx_masked_zero = np.random.choice(idx_zero, n_zeros, replace=False)
            idx_masked_non_zero = np.random.choice(idx_non_zero, n_non_zeros, replace=False)
            # mask
            mask[idx_masked_zero] = True
            mask[idx_masked_non_zero] = True
        return mask



# path = '../data/Pancreas/endocrinogenesis_day15.h5ad'
# data = scv.datasets.pancreas(path)
# p = Preprocessor(data, 100, 10, 200)
# p.preprocess()
# tokens = p.get_gene_tokens()
# data = p.binned_data
# p.get_mean_number_of_nonZero_bins()
# dataset = scDataSet(data, p.mean_non_zero_bins, 200)
# for i in range(len(data)):
#     sample, mask = dataset.__getitem__(i)
#     print(f'size mask={mask.sum()}')
# trainset, testset = random_split(dataset, [0.9, 0.1])
# train_loader = DataLoader(trainset, batch_size=10, shuffle=True)
# test_loader = DataLoader(testset, batch_size=10, shuffle=True)
# print(trainset.__len__())
# # for epoch in range(3):
# #     for i, (x_val, mask) in enumerate(train_loader):
# #         #print(i)
# #         if (i+1) % 50 == 0:
# #             print(i)
# #             print(x_val)
# #             print(mask)
#
# print(dataset.gene_tokens)