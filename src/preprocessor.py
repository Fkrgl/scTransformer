import scvelo as scv
import scanpy as sc
import numpy as np
from anndata import AnnData


class Preprocessor():
    """
    prepare row gene expression counts for transformer. Expression counts are 1) filtered 2) normalized 3) log1p
    transformed 4) highly variable genes (hvg) selected 5) value binned
    """

    def __init__(self,
                anndata: AnnData,
                n_bins: int,
                min_counts_genes: int = 10,
                n_hvg: int = 2400,
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
        bin_edges = []
        # perform value binning for each cell
        for row in data:
            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            # get borders of equally distributed bins
            bins = np.quantile(non_zero_row, np.linspace(0, 1, self.n_bins - 1))
            # spread all values equally across the bins
            non_zero_digits = self._digitize(non_zero_row, bins)
            binned_row = np.zeros_like(row, dtype=np.int64)
            # assign genes to bins
            binned_row[non_zero_ids] = non_zero_digits
            binned_rows.append(binned_row)
            bin_edges.append(np.concatenate([[0], bins]))
        # construct matrix from binned rows
        self.binned_data = np.stack(binned_rows)

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

    def permute(self):
        """
        randomly shuffles the dataset
        """
        n = self.data.shape[0]
        permute = np.random.permutation(n)
        self.data = self.data[permute]

    def get_gene_tokens(self):
        return self.data.var.index.values

if __name__ == '__main__':
    # load dataset
    anndata = scv.datasets.pancreas()
    p = Preprocessor(anndata, 10)
    p.preprocess()
    print(p.binned_data)
    print(f'output shape: {p.binned_data.shape}')
    print(f' one single example has size {p.binned_data[0].shape}: \n{p.binned_data[0]}')
