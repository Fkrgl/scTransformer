import scanpy as scp
import sys
from preprocessor import Preprocessor
import numpy as np

def get_hvgs():
    path = sys.argv[1]
    data = scp.read_h5ad(path)
    cell_types = []
    n_cells = []
    hvgs = []
    print('hi')
    # print(data.obs.cell_type)
    # print(data.obs.cell_type.values)
    # print(list(data.obs.cell_type.values.categories))
    # print(type(data.obs.cell_type._get_values))
    cluster = list(data.obs.cell_type.values.categories)
    for cell_type in cluster:
        subset = data[data.obs.cell_type == cell_type]
        print(f'celltype: {cell_type}\tsubset size={subset.shape[0]}')
        p = Preprocessor(subset, 100, n_hvg=200, vocab=None)
        p.preprocess()
        cell_types.append(cell_type)
        n_cells.append(subset.shape[0])
        hvgs.append(p.get_gene_tokens())

    np.save('/home/claassen/cxb257/scTransformer/data/pbmc_celltypes.npy', np.array(cell_types))
    np.save('/home/claassen/cxb257/scTransformer/data/pbmc_ncells.npy', np.array(n_cells))
    np.save('/home/claassen/cxb257/scTransformer/data/pbmc_hvgs.npy', np.array(hvgs))

def save_celltype_datasets(path):
    data = scp.read_h5ad(path)
    cluster = ['CD4-positive, alpha-beta T cell', 'dendritic cell', 'plasmacytoid dendritic cell']
    filename = ['CD4_positive_subset.h5ad', 'dendritic_subset.h5ad', 'plasmacytoid_dendritic_subset.h5ad']
    out_path = '/mnt/qb/work/claassen/cxb257/data/pbmc/'
    for i in range(len(cluster)):
        subset = data[data.obs.cell_type == cluster[i]]
        file_path = out_path + filename[i]
        subset.write_h5ad(file_path)


def main():
    path = sys.argv[1]
    save_celltype_datasets(path)

if __name__ == '__main__':
    main()
