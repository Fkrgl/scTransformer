"""
this script simply generates expression profiles. It is intended to experiment with the generation process
"""

import argparse
import scanpy as scp
import numpy as np
from scTransformer import *
from preprocessor import Preprocessor
import torch

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

def process_data(data, n_input_bins, min_counts_genes, n_token):
    p = Preprocessor(anndata=data,
                     n_bins=n_input_bins,
                     min_counts_genes=min_counts_genes,
                     n_hvg=n_token,
                     vocab=None)
    p.preprocess()
    p.get_mean_number_of_nonZero_bins()
    return p

def generate_samples_one_sample(sample, x_src, model, n_samples, use_val_emb = True):
    '''
    samples from the transfomer by only using the src embedding of the genes but not the actual expression values
    '''
    bins = np.arange(100)
    # generate data
    data_generated = [sample]
    mask = torch.tensor(np.zeros_like(sample, dtype=bool))
    for i in range(n_samples):
        # mask where each value is False
        sample_profile = model.generate(x_src, sample, torch.tensor([0]), mask, 'src_key_padding_mask', bins, False)
        data_generated.append(sample_profile)
    expression_profiles = np.vstack(data_generated)
    return expression_profiles

def get_mean_profile(data):
    return torch.tensor(np.round(np.mean(data, axis=0)))

def save_exp_profiles(profiles, path: str):
    if not path.endswith('/'):
        path += '/'
    np.save(path + 'spleen_generated_samples_profile.npy', profiles)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument('n', type=int, help='number of samples to generate')
    parser.add_argument('o', type=str, help='output dir for expression profiles')
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

    data = np.load(args.data_path)
    model = load_model(args.model_path, d_model, dim_feedforward, nlayers, n_input_bins, n_token, nhead)
    #profile = get_mean_profile(data)
    print(torch.tensor())
    x_src = torch.tensor(np.arange(500))
    #profile = torch.zeros_like(x_src)
    profile = data[0]
    generated_profiles = generate_samples_one_sample(profile, x_src, model, args.n)
    save_exp_profiles(generated_profiles, args.o)