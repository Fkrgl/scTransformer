import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional
import scvelo as scv
from preprocessor import Preprocessor
import scanpy as scp
from collections import Counter

class scDataSet(Dataset):
    def __init__(self,
                 data,
                 n_bins: int,
                 n_non_zero_bins: int,
                 n_tokens: int,
                 finetune: bool = False,  # for finetuning, the src_key padding mask has to be adapted to include the padded
                                        # positions and no padded positions should be selected for masking
                 hvg_finetune: Optional[str] = None
                 ):
        self.n_non_zero_bins = n_non_zero_bins
        print(f'masking prob real: {self.n_non_zero_bins}')
        self.n_tokens = n_tokens
        self.mask_value = n_bins  # value used for masking gene expression values
        self.finetune = finetune
        self.hvg_finetune = hvg_finetune
        # load data
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get sample
        sample = torch.tensor(self.data[idx])
        # generate mask for sample
        #mask = self.get_prob_mask(sample)
        mask = self.get_balanced_mask(sample)
        #attn_mask = self.create_attention_mask(mask)
        attn_mask = torch.tensor([0])
        masked_sample = sample.detach().clone()
        masked_sample[mask] = self.mask_value
        # exclude padding position from masking
        if self.finetune:
            masked_sample[self.hvg_finetune:] = 0
        # masked_sample = masked_sample.detach().numpy()
        # c = Counter(masked_sample)
        # print(c)
        return sample, masked_sample, attn_mask, mask

    def get_prob_mask(self, expressions: torch.Tensor) -> torch.Tensor:
        """
        generates a mask for a proportion of genes in the input data. The masks genes are predicted later in the training
        process. More information here:
        https://github.com/pytorch/pytorch/blob/11f1014c05b902d3eef0fe01a7c432f818c2bdfe/torch/nn/functional.py#L3892
        Args:
            expressions: expression matrix (batch, seq_length)
            mlm_probability: probability fo a gene to get masked

        Returns:
            Boolean Tensor of shape (batch, n_token) with True where genes should be masked

        """
        shape = expressions.shape
        probability_matrix = torch.full(shape, self.n_non_zero_bins)
        mask = torch.bernoulli(probability_matrix).bool()
        return mask

    def get_balanced_mask(self, sample):
        mask = np.zeros(self.n_tokens, dtype=bool)
        if self.finetune:
            # mask all padding positions
            mask[self.hvg_finetune:] = True
            idx = np.arange(self.hvg_finetune)
            sample = sample[:self.hvg_finetune]
        else:
            idx = np.arange(self.n_tokens)

        n_non_zeros = np.count_nonzero(sample)
        n_zeros = len(sample) - n_non_zeros
        # less non zero bins than average
        if n_non_zeros < self.n_non_zero_bins:
            diff = self.n_non_zero_bins - n_non_zeros
            # print(f'less non zero genes. diff={diff}')
            # print(f'n_zeros = {self.n_non_zero_bins} + {diff}')
            n_zeros = self.n_non_zero_bins + diff
        elif n_non_zeros > self.n_non_zero_bins:
            n_non_zeros = self.n_non_zero_bins
            n_zeros = self.n_non_zero_bins
        # more non-zero bins than average
        elif n_zeros < self.n_non_zero_bins:
            diff = self.n_non_zero_bins - n_zeros
            # print(f'more non zero genes. diff={diff}')
            # print(f'n_zeros = {self.n_non_zero_bins} - {diff}')
            n_non_zeros = self.n_non_zero_bins + diff
        elif n_non_zeros == self.n_non_zero_bins:
            n_non_zeros = self.n_non_zero_bins
            n_zeros = self.n_non_zero_bins

        idx_zero = idx[sample == 0]
        idx_non_zero = idx[sample != 0]
        # randomly select genes to be masked
        idx_masked_zero = np.random.choice(idx_zero, n_zeros, replace=False)
        idx_masked_non_zero = np.random.choice(idx_non_zero, n_non_zeros, replace=False)
        # mask
        mask[idx_masked_zero] = True
        mask[idx_masked_non_zero] = True
        return mask

    def create_attention_mask(self, mask):
        '''
        this function takes as input a padding mask (seq_len) and outputs a attention mask (seq_len x seq_len)
        '''
        attn_mask = torch.zeros(size=(len(mask), len(mask)))
        attn_mask[mask, :] = 1
        attn_mask[:, mask] = 1
        return attn_mask


if __name__ == '__main__':
    path = '../data/Pancreas/endocrinogenesis_day15.h5ad'
    data = scp.read_h5ad(path)
    p = Preprocessor(data, 100, None, n_hvg=1628)
    p.preprocess()
    p.get_mean_number_of_nonZero_bins()
    print(data)
    dataset = scDataSet(p.binned_data, 100, 0, 1628)
    sample, masked_sample, _ , mask = dataset.__getitem__(3)
    print(mask.sum())
    print(mask)