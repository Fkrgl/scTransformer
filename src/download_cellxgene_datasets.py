import cellxgene_census
import argparse
from preprocessor import *

def download(tissue:str, outpu_dir: str):
    with cellxgene_census.open_soma() as census:
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            obs_value_filter=f"tissue == '{tissue}' and disease == 'normal'",
            column_names={"obs": []}
        )
        print(f'tissue {tissue} has {adata.X.shape[0]} samples')
        print(adata.X.shape)
        adata.write_h5ad(outpu_dir)
        return adata

def preprocess_data(adata, tissue):
    p = Preprocessor(adata, 100, 10, 500)
    p.preprocess()
    data = p.binned_data
    print(f'after preprocessing tissue {tissue} has {data.shape[0]} cells left')


if __name__ == '__main__':
    tissue = 'colon'
    output_dir = '/mnt/qb/work/claassen/cxb257/data/cellxgene/colon_normal.h5ad'
    adata = download(tissue, output_dir)
    preprocess_data(adata, tissue)


