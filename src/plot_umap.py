from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import argparse
from preprocessor import Preprocessor
import scanpy as scp


def load_data(path: str):
    data = scp.read_h5ad(path)
    return data
def process_data(data, n_input_bins, min_counts_genes, n_token):
    p = Preprocessor(data, n_input_bins, min_counts_genes, n_token)
    p.preprocess()
    p.get_mean_number_of_nonZero_bins()
    return p

def plot_UMAP(exp_profiles, umap_path, name):
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    exp_profiles = np.round(exp_profiles, decimals=5)
    proj_2d = umap_2d.fit_transform(np.vstack(exp_profiles))

    # pca = PCA(n_components=50)
    # data_pca = pca.fit_transform(data)
    # data_pca = np.round(data_pca, decimals=5)
    # UMAP
    umap_2d_pca = UMAP(n_components=2, init='random', random_state=0)
    proj_2d_pca = umap_2d_pca.fit_transform(exp_profiles)
    # umap_2d = UMAP(n_components=2, init='random', random_state=0)
    # proj_2d = umap_2d.fit_transform(data)

    fig = plt.figure(figsize=(10, 7))
    plt.scatter(proj_2d_pca[:, 0], proj_2d_pca[:, 1], s=10, alpha=.3, label='data')
    plt.title(f'UMAP of {name}')

    plt.savefig(umap_path)

def plot_UMAP_binned_data(exp_profiles, umap_path, name):
    pca = PCA(n_components=50)
    data_pca = pca.fit_transform(exp_profiles)
    data_pca = np.round(data_pca, decimals=5)

    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    proj_2d = umap_2d.fit_transform(data_pca)

    fig = plt.figure(figsize=(10, 7))
    plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=10, alpha=.3, label='data')
    plt.title(f'UMAP of {name}')

    plt.savefig(umap_path)

def get_exp_profiles(p, data):
    '''
    translates bin values from expression value of a dataset into continous expression values
    '''
    bin_to_expression = p.bin_to_expression
    to_expression = lambda bins: [bin_to_expression[b] for b in bins]
    data_preprocessed = []
    print(len(data))
    for i in range(len(data)):
        sample = data[i]
        data_preprocessed.append(to_expression(sample))

    data_preprocessed = np.vstack(data_preprocessed)
    return data_preprocessed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to input data in h5ad format')
    parser.add_argument('umap_path', type=str, help='path to save UMAP plot')
    parser.add_argument('name', type=str, help='description of the dataset')

    args = parser.parse_args()

    # # load data
    # anndata = load_data(args.data_path)
    # # preprocess
    # n_token = 500
    # n_input_bins = 100
    # min_counts_genes = 10
    #
    # p = process_data(anndata, n_input_bins, min_counts_genes, n_token)
    # data = p.binned_data
    # exp_profiles = get_exp_profiles(p, data)
    # plot_UMAP(exp_profiles, args.umap_path, args.name)
    data = np.load(args.data_path, allow_pickle=True)
    plot_UMAP_binned_data(data, args.umap_path, args.name)

if __name__ == '__main__':
    main()