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
                 mlm_prob: float,
                 n_tokens: int,
                 finetune: bool = False,  # for finetuning, the src_key padding mask has to be adapted to include the padded
                                        # positions and no padded positions should be selected for masking
                 hvg_finetune: Optional[int] = None
                 ):
        self.mlm_prob = mlm_prob
        print(f'masking prob real: {self.mlm_prob}')
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
        sample = self.data[idx]
        # generate mask for sample
        mask = self.get_prob_mask(sample)
        attn_mask = torch.tensor([0])
        masked_sample = sample.copy()
        masked_sample[mask] = self.mask_value
        # exclude padding position from masking
        if self.finetune:
            masked_sample[self.hvg_finetune:] = 0

        return sample, masked_sample, attn_mask, mask

    def get_prob_mask(self, sample) -> torch.Tensor:
        """
        only take mlmProb % of the non zero gene for masking
        """
        #mask = np.zeros(self.n_tokens, dtype=bool)
        mask = np.ones(self.n_tokens, dtype=bool)
        non_zero_idx = sample.nonzero()[0]
        n_non_zero_gene = len(non_zero_idx)
        # select maksing at random
        n_masked = int(np.round(self.mlm_prob*n_non_zero_gene))
        # if no non zero gene is selected, take zero genes into account
        if n_masked == 0:
            idx = np.arange(self.n_tokens)
            idx_masked = np.random.choice(idx, n_masked, replace=False)
        else:
            idx_masked = np.random.choice(non_zero_idx, n_masked, replace=False)

        #mask[idx_masked] = True
        mask[idx_masked] = False
        return mask

if __name__ == '__main__':
    path = '../data/Pancreas/endocrinogenesis_day15.h5ad'
    data = scp.read_h5ad(path)
    p = Preprocessor(data, 100, None, n_hvg=1628)
    p.preprocess()
    p.get_mean_number_of_nonZero_bins()
    print(data)
    dataset = scDataSet(p.binned_data, 100, 0.15, 1628, finetune=True, hvg_finetune=200)
    sample, masked_sample, _ , mask = dataset.__getitem__(3)
    print(mask.sum())
    print(mask)