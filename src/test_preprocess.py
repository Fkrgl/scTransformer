'''
This script processes, splits and saves data into train and test data. The saved data is than
seperatly loaded into a train and test data script to avoid contamination of the datasets
'''
import scvelo as scv
import numpy as np
from scTransformer import *
from preprocessor import Preprocessor
import sys

def load_data(path):
    data = scv.datasets.pancreas(path)
    data.obs = data.obs.reset_index()
    # insert number column to check if there is not overlap between split indices later on
    test_idx = np.arange(len(data.obs))
    data.obs['test_idx'] = test_idx
    return data

def preprocess_data(data, n_bin, min_counts_genes, n_token):
    p = Preprocessor(data, n_bin, min_counts_genes, n_token)
    p.permute()
    p.preprocess()
    tokens = p.get_gene_tokens()
    data = p.binned_data
    return data, tokens
def get_split(data, cell_type: str):
    torch.manual_seed(1234)
    min_test_set = 481
    max_test_set = 642
    n_train_set = len(data) - max_test_set
    if cell_type == 'None':
        idx_all = np.arange(len(data))
        idx_test_cells = np.random.choice(idx_all, size=min_test_set, replace=False)
        idx_rest = list(set(idx_all) - set(idx_test_cells))
        idx_train_cells = np.random.choice(np.array(idx_rest), size=n_train_set, replace=False)
    elif cell_type == 'endstates':
        endstates = ['Alpha', 'Beta', 'Delta', 'Epsilon']
        idx_enstates = data.obs[data.obs.clusters.isin(endstates)].index.values
        idx_not_endstates = data.obs[~data.obs.clusters.isin(endstates)].index.values
        idx_test_cells = idx_enstates
        idx_train_cells = idx_not_endstates
    elif cell_type == 'earlystates':
        earlystates = ['Ductal', 'Ngn3 low EP', 'Ngn3 high EP']
        idx_earlystates = data.obs[data.obs.clusters.isin(earlystates)].index.values
        idx_not_earlystates = data.obs[~data.obs.clusters.isin(earlystates)].index.values
        idx_test_cells = idx_earlystates
        idx_train_cells = idx_not_earlystates
    elif cell_type == 'multiple':
        multiple = ['Ductal', 'Ngn3 low EP', 'Ngn3 high EP', 'Alpha', 'Beta', 'Delta']
        idx_multiple = data.obs[data.obs.clusters.isin(multiple)].index.values
        idx_not_multiple = data.obs[~data.obs.clusters.isin(multiple)].index.values
        idx_test_cells = idx_multiple
        idx_train_cells = idx_not_multiple
    else:
        idx_test_cells = data.obs[data.obs.clusters == cell_type].index.values
        idx_train_cells = data.obs[data.obs.clusters != cell_type].index.values
        idx_test_cells = np.random.choice(idx_test_cells, size=min_test_set, replace=False)
        idx_train_cells = np.random.choice(idx_train_cells, size=n_train_set, replace=False)

    return set(idx_train_cells), set(idx_test_cells)

def test_idx(idx_train_cells, idx_test_cells):
    intersection = idx_train_cells.intersection(idx_test_cells)
    if len(intersection) == 0:
        print('there is no overlap between train and test idices')
    else:
        print('Overlpap!')

def save_dataset(dataset, path):
    np.save(file=path, arr=dataset)

def save_tokens(tokens, path='../data/tokens.npy'):
    np.save(file=path, arr=tokens)

if __name__ == '__main__':
    path = '../data/Pancreas/endocrinogenesis_day15.h5ad'
    cell_type = sys.argv[1]
    n_bin = 100
    min_counts_genes = 10
    n_token = 200
    data = load_data(path)
    idx_train_cells, idx_test_cells = get_split(data, cell_type)
    data_preprocessed, tokens = preprocess_data(data, n_bin, min_counts_genes, n_token)
    # test if indices overlap
    test_idx(idx_train_cells, idx_test_cells)
    # split
    trainset = data_preprocessed[np.array(list(idx_train_cells), dtype=int)]
    testset = data_preprocessed[np.array(list(idx_test_cells), dtype=int)]
    # save
    train_path = f'../data/trainset_{cell_type}.npy'
    test_path = f'../data/testset_{cell_type}.npy'
    save_dataset(trainset, train_path)
    save_dataset(testset, test_path)


