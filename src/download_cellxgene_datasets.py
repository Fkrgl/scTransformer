import cellxgene_census
import argparse
from preprocessor import *
import sys

def download(tissue:str, outpu_dir: str):
    print('download..')
    with cellxgene_census.open_soma() as census:
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            obs_value_filter=f"tissue_general == 'liver' and disease == 'normal' and  cell_type == 'hepatocyte'",
            column_names={"obs": []}
        )
        print(f'tissue {tissue} has {adata.X.shape[0]} samples')
        print(adata.X.shape)
        adata.write_h5ad(outpu_dir)
        return adata

def preprocess_data(adata, tissue):
    tokens = [500]
    for t in tokens:
        d = adata.copy()
        p = Preprocessor(d, 50, 10, t)
        p.preprocess()
        p.get_mean_number_of_nonZero_bins()
        data = p.binned_data
        print(f'after preprocessing tissue {tissue} has {data.shape[0]} cells left with {data.shape[1]} tokens')
        print(f'the number of non zeros mean bin is {p.mean_non_zero_bins}')
        print(f'masking prob: {p.mean_non_zero_bins/t}')
        print()


if __name__ == '__main__':
    print('hi')
    tissue = 'pancreas'
    output_dir = f'/mnt/qb/work/claassen/cxb257/data/cellxgene/hepatocyte.h5ad'
    adata = download(tissue, output_dir)
    #adata = sc.read_h5ad(output_dir)
    preprocess_data(adata, tissue)


