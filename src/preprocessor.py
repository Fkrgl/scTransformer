import scvelo as scv
import scanpy as sc
import numpy as np
from anndata import AnnData
import scanpy as scp
import sys
import argparse
import math
import torch
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

        Returns
        -------

        """

        self.data = anndata
        self.min_counts_genes = min_counts_genes
        self.n_hvg = n_hvg
        self.n_bins = n_bins
        self.binned_data = None
        self.mean_non_zero_bins = None

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

        # value binning
        # get full data matrix (includes zeros)
        data = self.data.X.toarray()
        binned_rows = []
        # perform value binning for whole data set
        idx_non_zero_i, idx_non_zero_j = data.nonzero()
        values_non_zero = data[idx_non_zero_i, idx_non_zero_j]
        # get borders of equally distributed bins
        bins = np.quantile(values_non_zero, np.linspace(0, 1, self.n_bins))
        # the 0 bin is from 0 to bin_1
        non_zero_ids = data[0].nonzero()
        non_zero_row = data[0][non_zero_ids]
        non_zero_digits = np.digitize(non_zero_row, bins)
        all_binned = []
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
            all_binned += list(binned_row)
        # construct matrix from binned rows
        self.binned_data = np.stack(binned_rows)
        #self.create_bin_mapping(bins)

        # all_binned = np.array(all_binned)
        # c = Counter(list(all_binned[all_binned > 0]))
        # c = dict(sorted(c.items()))
        # for k, v in c.items():
        #     print(k, v)
        # plt.hist(all_binned[all_binned > 0], bins=99)
        # plt.show()
        #
        # # with 0
        # c = Counter(all_binned)
        # c = dict(sorted(c.items()))
        # for k, v in c.items():
        #     print(k, v)
        # plt.hist(all_binned, bins=100)
        # plt.show()

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

    def save_processed_data(self, path_out: str) -> None:
        # get size in byte of array
        print("Size of the array: ",
              self.binned_data.size)
        print("Memory size of one array element in bytes: ",
              self.binned_data.itemsize)
        # memory size of numpy array in bytes
        print("Memory size of numpy array in GB:",
              np.round(self.binned_data.size * self.binned_data.itemsize / math.pow(1024, 3), decimals=4))
        # save array
        np.save(path_out, self.binned_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_in',
                        type=str,
                        help='path to file of data set in h5ad format')
    parser.add_argument('n_hvg',
                        type=int,
                        help='number of highly variable genes')
    parser.add_argument('-path_out',
                        type=str,
                        help='path to file for preprocessed data')
    args = parser.parse_args()
    # load dataset
    anndata = scp.read_h5ad(args.path_in)
    p = Preprocessor(anndata, 100, n_hvg=args.n_hvg)
    p.preprocess()
    p.get_mean_number_of_nonZero_bins()
    print(f'mean number of non zero bins: {p.mean_non_zero_bins}')
    if args.path_out:
        p.save_processed_data(args.path_out)
    # print(p.binned_data)
    # print(f'output shape: {p.binned_data.shape}')
    # print(f' one single example has size {p.binned_data[0].shape}: \n{p.binned_data[0]}')
    # print(f'number of non zeros: {np.count_nonzero(p.binned_data[0])}')
    # non_zeros = np.count_nonzero(p.binned_data, axis=1)
    # print(f'number of non zeros: {non_zeros}')
    # print(f'{non_zeros.shape}')
    # print(f'mean of non zeros: {np.round(np.mean(non_zeros),decimals=0)}')
    #torch.save(p.binned_data, f'../data/test_values_wholeDataset.pt')
