import scvelo as scv
import scanpy as sc
import numpy as np
from anndata import AnnData
import torch
from typing import Optional
from collections import Counter
import matplotlib.pyplot as plt


class Preprocessor:
    """
    prepare row gene expression counts for transformer. Expression counts are 1) filtered 2) normalized 3) log1p
    transformed 4) highly variable genes (hvg) selected 5) value binned
    """

    def __init__(self,
                anndata: AnnData,
                n_bins: int,
                min_counts_genes: int = 10,
                n_hvg: int = 200,
                padding_idx: Optional[int] = None
                ):
        """

        Parameters
        ----------
        anndata:
            AnnData object that contains row scRNA seq counts for each cell (n_cell x n_genes)
        min_counts_genes:
            minimal number of raw counts for a gene to be considered
        n_hvg:
            number of hvgs to select. 2400 is suggestion from Luecken et al. (2019)
        n_bins:
            number of bins for value binning of expression counts
        padding_idx:
            number to fill padding positions (genes not present in new data). The padding index can be set to n_hvg
            because the bins are 0 indexed (range = 0 : (n_hvg-1))

        Returns
        -------

        """

        self.data = anndata
        self.min_counts_genes = min_counts_genes
        self.n_hvg = n_hvg
        self.n_bins = n_bins
        self.binned_data = None
        self.mean_non_zero_bins = None
        self.hvg = None
        self.padding_idx = padding_idx

    def preprocess(self):
        # filter by counts of genes
        sc.pp.filter_genes(self.data,
                           min_counts=self.min_counts_genes
                           )

        # filter by counts of cell
        # min_counts_cell = None
        # sc.pp.filter_cells(data,
        #                    min_counts=min_counts_cell
        # )

        # normalize counts
        sc.pp.normalize_total(self.data)

        # log1p transformation
        # without log1p row sums are all equal, with log1p they slightly differ
        sc.pp.log1p(self.data)

        # get highly varaible genes (hvg)
        sc.pp.highly_variable_genes(self.data, n_top_genes=self.n_hvg, subset=True)
        # save hvg gene names
        self.hvg = self.data.var.index.values
        np.save('../data/hvg_pancreas.npy', self.hvg)
        # value binning
        # get full data matrix (includes zeros)
        data = self.data.X.toarray()
        binned_rows = []
        # perform value binning for whole data set
        idx_non_zero_i, idx_non_zero_j = data.nonzero()
        values_non_zero = data[idx_non_zero_i, idx_non_zero_j]
        # get borders of equally distributed bins
        bins = np.quantile(values_non_zero, np.linspace(0, 1, self.n_bins))

        for row in data:
            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            # spread all values equally across the bins
            #non_zero_digits = self._digitize(non_zero_row, bins)
            non_zero_digits = np.digitize(non_zero_row, bins, right=True)
            binned_row = np.zeros_like(row, dtype=np.int64)
            # assign genes to bins
            binned_row[non_zero_ids] = non_zero_digits
            binned_rows.append(binned_row)
        # construct matrix from binned rows
        self.binned_data = np.stack(binned_rows)

    def preprocess_new_data(self):
        # filter by counts of genes
        sc.pp.filter_genes(self.data,
                           min_counts=self.min_counts_genes
                           )

        # filter by counts of cell
        # min_counts_cell = None
        # sc.pp.filter_cells(data,
        #                    min_counts=min_counts_cell
        # )

        # normalize counts
        sc.pp.normalize_total(self.data)

        # log1p transformation
        # without log1p row sums are all equal, with log1p they slightly differ
        sc.pp.log1p(self.data)

        # select highly varaible genes in new dataset
        hvg = np.load('../data/hvg_pancreas.npy', allow_pickle=True)
        print(f'hvg: {hvg}')
        self.data.var['id'] = np.arange(len(self.data.var))
        hvg_in_data = self.data.var[self.data.var.index.isin(hvg)]
        print(f'hvg_in_data: {hvg_in_data}')
        data = self.data.X.toarray()[:,hvg_in_data.id.values]
        print(f'shape of data = {data.shape}')

        # value binning
        # get full data matrix (includes zeros)

        binned_rows = []
        # perform value binning for whole data set
        idx_non_zero_i, idx_non_zero_j = data.nonzero()
        values_non_zero = data[idx_non_zero_i, idx_non_zero_j]
        # get borders of equally distributed bins
        bins = np.quantile(values_non_zero, np.linspace(0, 1, self.n_bins))

        for row in data:
            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            # spread all values equally across the bins
            non_zero_digits = np.digitize(non_zero_row, bins, right=True)
            binned_row = np.zeros_like(row, dtype=np.int64)
            # assign genes to bins
            binned_row[non_zero_ids] = non_zero_digits
            binned_rows.append(binned_row)
        # construct matrix from binned rows
        self.binned_data = np.stack(binned_rows)
        #self.create_bin_mapping(bins)
        if self.binned_data.shape[1] < self.n_hvg:
            # perform padding
            self._pad()
            print(type(self.binned_data))

    def _pad(self):
        self.binned_data = torch.cat(
            [
                torch.tensor(self.binned_data),
                torch.full(
                    (self.binned_data.shape[0], self.n_hvg-self.binned_data.shape[1]),
                    self.padding_idx,
                ),
            ], dim=1
        )
        # convert tensor back to numpy array
        self.binned_data = self.binned_data.detach().numpy()

    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        right_digits = np.digitize(x, bins, right=True)

        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_digits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    def create_bin_mapping(self, bins):
        bins_to_value = {}
        for i in range(len(bins)-1):
            bins_to_value[i] = (bins[i] + bins[i+1]) / 2
        bins_to_value[0] = 0
        print(bins_to_value)


    def permute(self):
        """
        randomly shuffles the dataset
        """
        n = self.data.shape[0]
        permute = np.random.permutation(n)
        self.data = self.data[permute]

    def get_gene_tokens(self):
        return self.data.var.index.values

    def get_mean_number_of_nonZero_bins(self):
        non_zeros = np.count_nonzero(self.binned_data, axis=1)
        mean_non_zeros = int(np.round(np.mean(non_zeros),decimals=0))
        self.mean_non_zero_bins = mean_non_zeros

if __name__ == '__main__':
    # load dataset
    anndata = scv.datasets.pancreas()
    p = Preprocessor(anndata, 100)
    p.preprocess()
    denta = scv.datasets.dentategyrus()
    p2 = Preprocessor(denta, 100, padding_idx=100)
    p2.preprocess_new_data()
    # print(p2.binned_data)
    # print(f'output shape: {p2.binned_data.shape}')
    # print(f' one single example has size {p.binned_data[0].shape}: \n{p.binned_data[0]}')
    # print(f'number of non zeros: {np.count_nonzero(p.binned_data[0])}')
    # non_zeros = np.count_nonzero(p.binned_data, axis=1)
    # print(f'number of non zeros: {non_zeros}')
    # print(f'{non_zeros.shape}')
    # print(f'mean of non zeros: {np.round(np.mean(non_zeros),decimals=0)}')
    #torch.save(p.binned_data, f'../data/test_values_wholeDataset.pt')
    token_names = p.get_gene_tokens()
    vocab_size = len(token_names)
    # map each gene to a unique id
    token_to_id = {token: i + 1 for i, token in enumerate(token_names)}
    id_to_token = {i + 1: token for i, token in enumerate(token_names)}
    # add padding token with id 0
    token_to_id["pad"] = 0
    id_to_token[0] = "pad"
    # create mapping functions
    encode = lambda token_list: [token_to_id[t] for t in
                                 token_list]  # encoder: take a list of gene names, output list of integers
    decode = lambda index_list: [id_to_token[idx] for idx in
                                 index_list]  # decoder: take a list of integers, output a string
    # print(token_to_id)
    # print(token_to_id['pad'])
    # print(encode(['pad'])[0])
