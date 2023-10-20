'''
Testing pyDEseq2 on our binned dataset
'''

import os
import pickle as pkl
import pandas as pd
import numpy as np
import json
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data



def deseq2(data, gene_names, celltypes):
    binned_data = pd.DataFrame(data, columns=gene_names)
    group = np.repeat('other', len(data))
    group[celltypes == 'gamma-delta T cell'] = 'celltype'
    metadata = pd.DataFrame({'group' : group})

    dds = DeseqDataSet(
        counts=binned_data,
        metadata=metadata,
        design_factors="group",
        refit_cooks=True,
        n_cpus=6,
    )

    dds.deseq2()
    print(dds)

    stat_res = DeseqStats(dds, n_cpus=6)
    stat_res.summary()
    print(stat_res.results_df)
    stat_res.results_df.to_csv('/mnt/qb/work/claassen/cxb257/data/pbmc/dge.csv')

def wilcoxon(data, gene_names, celltypes):
    # construct anndata object
    counts = csr_matrix(data)
    adata = ad.AnnData(counts)
    adata.var_names = gene_names
    adata.obs["cell_type"] = celltypes
    adata.layers["counts"] = adata.X

    # normalize "count data"
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # one vs all testing
    sc.tl.rank_genes_groups(
        adata,
        method="wilcoxon",
        groupby="cell_type",
        key_added="dea_celltypes"
    )

    # filter for cluster specific hvgs
    sc.tl.filter_rank_genes_groups(
        adata,
        min_in_group_fraction=0.2,
        max_out_group_fraction=0.2,
        key="dea_celltypes",
        key_added="dea_celltypes_filtered",
    )

    # plot
    sc.pl.rank_genes_groups_dotplot(
        adata,
        groupby="cell_type",
        standard_scale="var",
        n_genes=5,
        key="dea_celltypes_filtered",
        save='pbmc_binned_deg_wilcoxon.png'
    )
    return adata

def umap(adata):
    print(f'adata umap: {adata}')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.tl.umap(adata)
    return adata

def plot_umaps_marker_genes(adata, celltypes):
    marker_genes = {'natural killer cell': ['GZMB', 'TYROBP', 'PRF1', 'FGFBP2', 'KLRF1'],
                    'naive B cell': ['IGHM', 'HLA-DRA', 'CD79A', 'CD79B', 'IGHD'],
                    'CD14-low, CD16-positive monocyte': ['LST1', 'IFI30', 'CST3', 'SERPINA1', 'MS4A7'],
                    'CD14-positive monocyte': ['LYZ', 'S100A8', 'S100A9', 'FCN1', 'CST3'],
                    'naive thymus-derived CD8-positive, alpha-beta T cell': ['CD8B', 'NC02446', 'CD8A', 'AIF1',
                                                                             'S100B']
                    }
    filenames = ['natural_killer_cell',
                'naive_B_cell',
                 'CD14_low_CD16_positive_monocyte',
                 'CD14_positive_monocyte',
                 'naive_thymus_derived_CD8_positive_alpha_beta_T_cell'
                 ]
    DIR_PATH = '/home/claassen/cxb257/scTransformer/fig/'
    # calc umap
    adata = umap(adata)
    for i, ct in enumerate(marker_genes.keys()):
        color_cells = np.repeat('other', len(celltypes))
        idx_select = np.where(celltypes == ct)[0]
        color_cells[idx_select] = 'cluster'
        adata.obs['cluster'] = color_cells
        fig = sc.pl.umap(adata, color=marker_genes[ct] + ['cluster'], return_fig=True, ncols=3)
        fig.suptitle(f'{ct}', fontsize=22)
        file_path = DIR_PATH + filenames[i] + '_UMAP_markers_pbmc_binned.png'
        fig.savefig(file_path)
        del adata.obs['cluster']



def main():
    path_data = '/mnt/qb/work/claassen/cxb257/data/preprocessed/pbmc/pbmc_1500.npy'
    path_celltypes = '/mnt/qb/work/claassen/cxb257/data/pbmc/pbmc_celltypes.npy'
    path_vocab = '/mnt/qb/work/claassen/cxb257/data/pbmc/pbmc_vocab.json'

    # path_data = '/home/claassen/cxb257/scTransformer/data/pmbc_smallTestSet.npy'
    # path_celltypes = '/home/claassen/cxb257/scTransformer/data/pmbc_smallTestSet_celltypes.npy'
    # path_vocab = '/home/claassen/cxb257/scTransformer/data/pmbc_smallTestSet_genes.npy'
    # gene_names = np.load(path_vocab, allow_pickle=True)
    # celltypes = np.delete(celltypes, 1691)
    # data = np.delete(data, 1691, axis=0)

    data = np.load(path_data, allow_pickle=True)
    celltypes = np.load(path_celltypes, allow_pickle=True)

    print(celltypes)

    with open(path_vocab, 'r') as f:
        vocab = json.load(f)
    gene_names = np.array(list(vocab.keys()))[1:]
    print('run wilcoxon test ..')
    adata = wilcoxon(data, gene_names, celltypes)
    print(f'adata wilcoxon: {adata}')
    print('get umap ..')
    adata = umap(adata)
    #adata.write_h5ad('/mnt/qb/work/claassen/cxb257/data/preprocessed/pbmc/pbmc_adata_umap.h5ad')
    # plot umaps with marker genes
    plot_umaps_marker_genes(adata, celltypes)

if __name__ == '__main__':
    main()




