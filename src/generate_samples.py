import sys
from scTransformer import *
from preprocessor import Preprocessor
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scvelo as scv
import scanpy as scp
import numpy as np
from DataSet import scDataSet
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import argparse
import evaluate_samples
from metrics import c2st
from sklearn.decomposition import PCA

def get_gene_encode_decode(token_names: list):
    """
    Returns: encoder and decoder functions that map gene names to integer and vise versa
    """
    vocab_size = len(token_names)
    # map each gene to a unique id
    token_to_id = {token: i for i, token in enumerate(token_names)}
    id_to_token = {i: token for i, token in enumerate(token_names)}
    encode = lambda token_list: [token_to_id[t] for t in
                                 token_list]  # encoder: take a list of gene names, output list of integers
    decode = lambda index_list: [id_to_token[idx] for idx in
                                 index_list]  # decoder: take a list of integers, output a string
    return encode, decode


# load models
def load_model(model_path: str, d_model, dim_feedforward, nlayers, n_input_bins, n_token,
               nhead):
    model = TransformerModel(d_model=d_model,
                             dim_feedforward=dim_feedforward,
                             nlayers=nlayers,
                             n_input_bins=n_input_bins,
                             n_token=n_token,
                             nhead=nhead)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# load data
def load_data(path: str):
    data = scp.read_h5ad(path)
    return data

def process_data(data, n_input_bins, min_counts_genes, n_token):
    p = Preprocessor(anndata=data,
                     n_bins=n_input_bins,
                     min_counts_genes=min_counts_genes,
                     n_hvg=n_token,
                     vocab=None)
    p.preprocess()
    p.get_mean_number_of_nonZero_bins()
    return p

# generate
def generate_samples(p, dataset, x_src, model, n_samples, use_val_emb = True):
    '''
    samples from the transfomer by only using the src embedding of the genes but not the actual expression values
    '''
    bin_to_expression = p.bin_to_expression
    to_expression = lambda bins: [bin_to_expression[b] for b in bins]
    bins = np.arange(100)
    # generate data
    data_generated = []
    data_preprocessed = []
    for i in range(n_samples):
        sample, _, a, b = dataset.__getitem__(i)
        data_preprocessed.append(to_expression(sample.detach().numpy()))
        # mask where each value is False
        mask = torch.tensor(np.zeros_like(b, dtype=bool))
        sample_profile = model.generate(x_src, sample, a, mask, 'src_key_padding_mask', bins, False)
        expression = to_expression(sample_profile)
        data_generated.append(expression)
    expression_profiles = np.vstack(data_generated)
    data_preprocessed = np.vstack(data_preprocessed)
    return data_preprocessed, expression_profiles

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

def get_train_test_set(data, n_bin, n_mask, n_token, n_samples, split=0.9):
    np.random.seed(42)
    # split
    n_train = int(split * len(data))
    n_test = int(len(data) - n_train)
    idx_all = np.arange(len(data))
    idx_test_cells = np.random.choice(idx_all, size=n_test, replace=False)
    idx_rest = list(set(idx_all) - set(idx_test_cells))
    idx_train_cells = idx_rest
    # subsample
    idx_test_cells = np.random.choice(idx_test_cells, size=n_samples, replace=False)
    idx_train_cells = np.random.choice(idx_train_cells, size=n_samples, replace=False)
    # create datasets
    trainset = scDataSet(data[idx_train_cells], n_bin, n_mask, n_token)
    testset = scDataSet(data[idx_test_cells], n_bin, n_mask, n_token)
    return trainset, testset

def evaluate(X_train, X_test, Y):
    X, Y = evaluate_samples.get_projected_data(X_train, Y)
    X_test, _ = evaluate_samples.get_projected_data(X_test, Y)
    X = torch.tensor(X).type(torch.DoubleTensor)
    Y = torch.tensor(Y).type(torch.DoubleTensor)
    X_test = torch.tensor(X_test).type(torch.DoubleTensor)
    # get metrics
    euclid_XX = evaluate_samples.euclidean_distance_centroids(X, X_test)
    euclid_XY = evaluate_samples.euclidean_distance_centroids(X, Y)
    mmd_XX = evaluate_samples.MMD(X, X_test, kernel='rbf')
    mmd_XY = evaluate_samples.MMD(X, Y, kernel='rbf')
    c2st_XX = c2st(X, X_test, classifier='mlp')
    c2st_XY = c2st(X, Y, classifier='mlp')
    c2st_rf_XX = c2st(X, X_test, classifier='rf')
    c2st_rf_XY = c2st(X, Y, classifier='rf')
    # report metrics
    print('sample / generated  |  sample / control')
    print(f'euclid\nval={euclid_XY} | ref={euclid_XX}\n')
    print(f'MMD\nval={mmd_XY} | ref={mmd_XX}\n')
    print(f'c2st with mlp\nval={c2st_XY} | ref={c2st_XX}\n')
    print(f'c2st with rf\nval={c2st_rf_XY} | ref={c2st_rf_XX}\n')

# def generate_color_vector(anndata):
#     clusters = anndata.obs.clusters
#     categories = anndata.obs.clusters.unique()
#     cmap = plt.cm.get_cmap('hsv', len(categories) + 1)
#     cluster_to_color = {categories[i]: cmap(i) for i in range(len(categories))}
#     cluster_to_color['Alpha'] = 'yellow'
#     color = [cluster_to_color[clu] for clu in clusters]
#     return cluster_to_color, color
def plot_UMAP(generted_expression_profiles, src_expression_profiles, src_control, path):
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    src_expression_profiles = np.round(src_expression_profiles, decimals=5)
    proj_2d = umap_2d.fit_transform(np.vstack([src_expression_profiles, generted_expression_profiles]))
    n_src = src_expression_profiles.shape[0]
    print(f'n_src: {n_src}')
    fig, ax = plt.subplots(2, 2, figsize=(10, 4), sharex=True, sharey=True)
    # sample vs generated
    ax[0,0].scatter(proj_2d[:n_src, 0], proj_2d[:n_src, 1], c='blue', alpha=.5, label='data')
    ax[0,0].scatter(proj_2d[n_src:, 0], proj_2d[n_src:, 1], c='yellow', alpha=.5, label='generated')
    ax[0,0].legend()
    fig.suptitle(f'umap of binned expression profiles')
    fig.supxlabel('umap 1')
    fig.supylabel('umap 2')
    ax[0,1].scatter(proj_2d[:n_src, 0], proj_2d[:n_src, 1], alpha=.5)
    # sample vs control
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    src_control = np.round(src_control, decimals=5)
    proj_2d = umap_2d.fit_transform(np.vstack([src_expression_profiles, src_control]))

    ax[1,0].scatter(proj_2d[:n_src, 0], proj_2d[:n_src, 1], c='blue', alpha=.5, label='data')
    ax[1,0].scatter(proj_2d[n_src:, 0], proj_2d[n_src:, 1], c='yellow', alpha=.5, label='control')
    ax[1,1].scatter(proj_2d[:n_src, 0], proj_2d[:n_src, 1], alpha=.5)

    plt.savefig(path)

def plot_dataset(p, dataset, n_components=50):
    # translate bin into expression data
    print(dataset.data)
    print(dataset.data.shape)
    data = get_exp_profiles(p, dataset)
    data = np.round(data, decimals=5)
    # perform pca
    print(data)
    print(data.shape)
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    data_pca = np.round(data_pca, decimals=5)
    # UMAP
    umap_2d_pca = UMAP(n_components=2, init='random', random_state=0)
    proj_2d_pca = umap_2d_pca.fit_transform(data_pca)
    # umap_2d = UMAP(n_components=2, init='random', random_state=0)
    # proj_2d = umap_2d.fit_transform(data)

    fig = plt.figure(figsize=(10, 7))
    plt.scatter(proj_2d_pca[:,0], proj_2d_pca[:,1], alpha=.3, label='data')
    plt.title('UMAP of spleen dataset (with PCA)')

    plt.savefig('/home/claassen/cxb257/scTransformer/fig/spleen_processed_pca.png')

def save_exp_profiles(original, generated, path: str):
    if not path.endswith('/'):
        path += '/'
    np.save(path + 'original_unmasked.npy', original)
    np.save(path + 'generated_unmasked.npy', generated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument('n', type=int, help='number of samples to generate')
    parser.add_argument('--umap_path', type=str, help='path to save UMAP figure of sample vs. generated and '
                                                      'sample vs. control')
    parser.add_argument('--o', type=str, help='output dir for expression profiles')
    args = parser.parse_args()
    print(args.model_path)

    # model parameters
    d_model = 128
    n_token = 500
    nhead = 4
    dim_feedforward = 32
    nlayers = 4
    n_input_bins = 100
    min_counts_genes = 10

    # generate samples
    n_samples = args.n
    model = load_model(args.model_path, d_model, dim_feedforward, nlayers, n_input_bins, n_token, nhead)
    anndata = load_data(args.data_path)
    p = process_data(anndata, n_input_bins, min_counts_genes, n_token)
    data = p.binned_data
    # trainset = scDataSet(data, n_input_bins, 0, n_token)

    trainset, testset = get_train_test_set(data, n_input_bins, 0, n_token, n_samples, split=0.5)
    tokens = p.get_gene_tokens()
    encode, decode = get_gene_encode_decode(tokens)
    x_src = torch.tensor(encode(tokens))
    train, Y = generate_samples(p, trainset, x_src, model, n_samples)
    test = get_exp_profiles(p, testset)


    # evaluate samples
    print(f'train: {train.shape}')
    print(f'test: {test.shape}')
    print(f'Y: {Y.shape}')
    evaluate(train, test, Y)

    if args.o:
        save_exp_profiles(train, test, args.o)
    # plot UMAP
    if args.umap_path:
        plot_UMAP(train, Y, test, args.umap_path)
    dataset = scDataSet(data, n_input_bins, p.mean_non_zero_bins//2, n_token)
    plot_dataset(p, dataset)