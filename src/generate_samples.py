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

# model parameters
d_model = 10
n_token = 200
nhead = 2
dim_feedforward = 100
nlayers = 2
n_input_bins = 100


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
model.load_state_dict(torch.load('../data/model_1.pth', map_location=torch.device('cpu')))
model.eval()

model2 = TransformerModel(d_model=d_model,
                                     dim_feedforward=dim_feedforward,
                                     nlayers=nlayers,
                                     n_input_bins=n_input_bins,
                                     n_token=n_token,
                                     nhead=nhead)
model2.load_state_dict(torch.load('../data/model_2.pth', map_location=torch.device('cpu')))
model2.eval()

# load data
dataset_path = '../data/Pancreas/endocrinogenesis_day15.h5ad'
data = scv.datasets.pancreas(dataset_path)
p = Preprocessor(data, n_input_bins, 10, n_token)
p.permute()
p.preprocess()
p.get_mean_number_of_nonZero_bins()
tokens = p.get_gene_tokens()
data = p.binned_data
# split data
print(f'number of tokens: {n_token}')
print(f'number of non zero bins: {p.mean_non_zero_bins}')
dataset = scDataSet(data, p.mean_non_zero_bins, n_token)
# encode gene names
n_token = len(tokens)
encode, decode = get_gene_encode_decode(tokens)
x_src = torch.tensor(encode(tokens))

# generate
val, _ = dataset.__getitem__(0)
sample_profile = model.generate(x_src, val, np.arange(100))
print(sample_profile)
# next: translate bins into expression values
sample_profile = model2.generate(x_src, val, np.arange(100))
bin_to_expression = p.bin_to_expression
to_expression = lambda bins: [bin_to_expression[b] for b in bins]
# sample multiple profiles
n_samples = 1000
expression_profiles = np.array(to_expression(val.detach().numpy()))
for i in range(n_samples):
    sample_profile = model2.generate(x_src, val, np.arange(100))
    expression = to_expression(sample_profile)
    expression_profiles = np.vstack((expression_profiles, expression))
print(expression_profiles)
std = expression_profiles.std(axis=0)
print(f'std for each gene: {expression_profiles.std(axis=0)}')

 # umap
umap_2d = UMAP(n_components=2, init='random', random_state=0)
expression_profiles = np.round(expression_profiles, decimals=5)
proj_2d = umap_2d.fit_transform(expression_profiles)
print(proj_2d)
color = ['blue'] * expression_profiles.shape[0]
color[0] = 'orange'
plt.scatter(proj_2d[1:,0], proj_2d[1:,1], c='blue', alpha=.5)
plt.scatter(proj_2d[0,0], proj_2d[0,1], c='orange', s=100)
plt.title(f'umap of {n_samples} generated expression profiles')
plt.xlabel('umap 1')
plt.ylabel('umap 2')
#plt.show()
plt.savefig('Umap_generated_amples.png')