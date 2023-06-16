import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scvelo as scv
from preprocessor import Preprocessor

class scDataSet(Dataset):
    def __init__(self,
                 path: str,
                 bins: int,
                 min_counts_genes: int,
                 n_hvg: int,
                 mlm_probability: float
                 ):
        self.mlm_probability = mlm_probability
        # load data
        self.data = scv.datasets.pancreas(path)
        # preprocess data
        self.data, self.gene_tokens = self.preprocess_data(self.data, bins, min_counts_genes, n_hvg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get sample
        sample = torch.tensor(self.data[idx])
        # generate mask for sample
        mask = self.get_mask(sample)
        return sample, mask

    def preprocess_data(self, data, bins, min_counts_genes, n_hvg):
        """
        performs all preprocessing steps for scRNA data
        """
        p = Preprocessor(data, bins, min_counts_genes, n_hvg)
        p.preprocess()
        tokens = p.get_gene_tokens()
        return p.binned_data, tokens

    def get_mask(self, expressions: torch.Tensor) -> torch.Tensor:
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
        probability_matrix = torch.full(shape, self.mlm_probability)
        mask = torch.bernoulli(probability_matrix).bool()
        return mask





# dataset = scDataSet('../data/Pancreas/endocrinogenesis_day15.h5ad', 10, 10, 200)
# # print(dataset.__len__())
# # print(dataset.__getitem__(1))
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