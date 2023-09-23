import cellxgene_census
import argparse
from preprocessor import *
import sys
from umap import UMAP
from DataSet import scDataSet

def download(tissue:str, outpu_dir: str):
    print('download..')
    with cellxgene_census.open_soma() as census:
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            obs_value_filter=f"tissue_general == 'heart' and disease == 'normal' and cell_type == 'endothelial cell of artery'",
            column_names={"obs": []}
        )
        # "tissue_general == 'liver' and disease == 'normal' and  cell_type == 'hepatocyte'"
        print(f'tissue {tissue} has {adata.X.shape[0]} samples')
        print(adata.X.shape)
        adata.write_h5ad(outpu_dir)
        return adata

def plot_UMAP(exp_profiles, umap_path, name):
    exp_profiles = np.round(exp_profiles, decimals=5)
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    proj_2d = umap_2d.fit_transform(exp_profiles)

    fig = plt.figure(figsize=(10, 7))
    plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=10, alpha=.3, label='data')
    plt.title(f'UMAP of {name}')

    plt.savefig(umap_path)

def preprocess_data(adata, tissue):
    tokens = 500
    d = adata.copy()
    p = Preprocessor(d, 50, 10, tokens)
    p.preprocess()
    p.get_mean_number_of_nonZero_bins()
    data = p.binned_data
    print(f'after preprocessing tissue {tissue} has {data.shape[0]} cells left with {data.shape[1]} tokens')
    print(f'the number of non zeros mean bin is {p.mean_non_zero_bins}')
    print(f'masking prob: {p.mean_non_zero_bins/tokens}')
    return p

def get_exp_profiles(p, dataset):
    '''
    translates bin values from expression value of a dataset into continous expression values
    '''
    bin_to_expression = p.bin_to_expression
    to_expression = lambda bins: [bin_to_expression[b] for b in bins]
    data_preprocessed = []
    print(len(dataset))
    for i in range(len(dataset)):
        sample, _, _, _ = dataset.__getitem__(i)
        data_preprocessed.append(to_expression(sample.detach().numpy()))

    data_preprocessed = np.vstack(data_preprocessed)
    return data_preprocessed

if __name__ == '__main__':
    print('hi')
    tissue = 'heart_endothelial_cell_of_artery'
    output_dir = f'/mnt/qb/work/claassen/cxb257/data/cellxgene/heart_endothelial_cell_of_artery.h5ad'
    adata = download(tissue, output_dir)
    #adata = sc.read_h5ad(output_dir)
    p = preprocess_data(adata, tissue)
    dataset = scDataSet(p.binned_data, 100, 4, 500)
    data = get_exp_profiles(p, dataset)
    plot_UMAP(data, f'/home/claassen/cxb257/scTransformer/fig/{tissue}.png', tissue)


