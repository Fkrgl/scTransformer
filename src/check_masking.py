import sys
from scTransformer import *
from preprocessor import Preprocessor
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scvelo as scv
import numpy as np
from DataSet import scDataSet
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# model parameters
d_model = 10
n_token = 50
nhead = 2
dim_feedforward = 100
nlayers = 2
n_input_bins = 30

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
model = TransformerModel(d_model=d_model,
                         dim_feedforward=dim_feedforward,
                         nlayers=nlayers,
                         n_input_bins=n_input_bins,
                         n_token=n_token,
                         nhead=nhead)
model.load_state_dict(torch.load('../data/model_4.pth', map_location=torch.device('cpu')))
model.eval()

# load data
dataset_path = '../data/Pancreas/endocrinogenesis_day15.h5ad'
data = scv.datasets.pancreas(dataset_path)
p = Preprocessor(data, n_input_bins, 10, n_token)
p.preprocess()
p.get_mean_number_of_nonZero_bins()
tokens = p.get_gene_tokens()
data = p.binned_data

print()

# create dataset
print(f'number of tokens: {n_token}')
print(f'number of non zero bins: {p.mean_non_zero_bins}')
dataset = scDataSet(data, 5, n_token)
# encode gene names
n_token = len(tokens)
encode, decode = get_gene_encode_decode(tokens)
tok = np.array(tokens)
x_src = torch.tensor(encode(tokens))


# analyse masking behaviour
value, attn_mask, key_padding_mask = dataset.__getitem__(420)
# 1st mask
mask_type = 'src_key_padding_mask'
mlm_output, masked_pred_exp, masked_label_exp = model(x_src, value, attn_mask, torch.tensor(key_padding_mask), mask_type,
                                                              get_reconstruction=True)
print(mlm_output)
mlm_output2, masked_pred_exp, masked_label_exp = model(x_src, value, attn_mask, torch.tensor(key_padding_mask), mask_type,
                                                              get_reconstruction=True)
print(mlm_output2)
print(torch.equal(mlm_output, mlm_output2))
print(key_padding_mask)

# 2nd mask
np.random.seed(420)
n_mask = 20
mask = np.zeros(n_token, dtype=bool)
rand_index = np.random.choice(np.arange(n_token), n_mask, replace=True)
mask[rand_index] = True
print(mask)
mlm_output3, masked_pred_exp3, masked_label_exp3 = model(x_src, value, attn_mask, torch.tensor(mask), mask_type,
                                                              get_reconstruction=True)
print(masked_pred_exp)
print(masked_pred_exp3)
print(torch.equal(masked_pred_exp, masked_pred_exp3))