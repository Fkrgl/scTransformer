"""
this script simply generates expression profiles. It is intended to experiment with the generation process
"""

import argparse
import scanpy as scp
import numpy as np
from scTransformer import *
from preprocessor import Preprocessor
import torch
from GeneVocab import GeneVocab

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

def generate_samples(profiles, x_src, model):
    '''
    Use each input profile to generate a new profile
    '''
    bins = np.arange(100)
    # generate data
    data_generated = []
    # mask where each value is False
    mask = torch.tensor(np.zeros_like(profiles[0], dtype=bool))
    for sample in profiles:
        sample = torch.tensor(sample)
        sample_profile = model.generate(x_src, sample, torch.tensor([0]), mask, 'src_key_padding_mask', bins, False)
        data_generated.append(sample_profile)
    data_generated = np.vstack(data_generated)
    return data_generated
def generate_samples_one_sample(sample, x_src, model, n_samples, use_val_emb = True):
    '''
    samples from the transfomer by only using the src embedding of the genes but not the actual expression values
    '''
    bins = np.arange(100)
    # generate data
    data_generated = [sample]
    # mask where each value is False
    mask = torch.tensor(np.zeros_like(sample, dtype=bool))
    for i in range(n_samples):
        sample_profile = model.generate(x_src, sample, torch.tensor([0]), mask, 'src_key_padding_mask', bins, False)
        data_generated.append(sample_profile)
    expression_profiles = np.vstack(data_generated)
    return expression_profiles

def generate_sample_sequence(sample, x_src, model, n_samples, use_val_emb = True):
    '''
    samples from the transfomer by using one sample as input. Ech generated sample is used as the input for the
    next sample step
    '''
    bins = np.arange(100)
    # generate data
    data_generated = [sample]
    mask = torch.tensor(np.zeros_like(sample, dtype=bool))
    for i in range(n_samples):
        # mask where each value is False
        sample = torch.tensor(sample)
        sample_profile = model.generate(x_src, sample, torch.tensor([0]), mask, 'src_key_padding_mask', bins, False)
        data_generated.append(sample_profile)
        # generated sample is new input sample
        sample = sample_profile
    expression_profiles = np.vstack(data_generated)
    return expression_profiles

def get_subsample(data, n_sample):
    np.random.seed(42)
    random_indices = np.random.choice(data.shape[0], n_sample, replace=False)
    data = data[random_indices, :]

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
    parser.add_argument('token_path', type=str)
    parser.add_argument('vocab_path', type=str)
    parser.add_argument('n', type=int, help='number of samples to generate')
    parser.add_argument('o', type=str, help='output dir for expression profiles')
    args = parser.parse_args()
    print(args.model_path)

    data = np.load(args.data_path)
    gene_names = np.load(args.token_path)
    vocab = GeneVocab()
    vocab.load_from_file(args.vocab_path)

    # model parameters
    d_model = 256
    n_token = len(vocab.vocab)
    nhead = 8
    dim_feedforward = 64
    nlayers = 4
    n_input_bins = 100
    min_counts_genes = 10

    model = load_model(args.model_path, d_model, dim_feedforward, nlayers, n_input_bins, n_token, nhead)
    # create src
    vocab_size = len(vocab.vocab) - 1
    x_src = [vocab.vocab[gene] for gene in gene_names]
    x_src = torch.tensor(x_src)
    x_src = torch.cat([x_src, torch.full(size=(vocab_size - n_token,), fill_value=0)])
    #profile = torch.zeros_like(x_src)
    profiles = generate_samples(data, args.n)
    generated_profiles = generate_samples(profiles, x_src, model)
    #generated_profiles = generate_samples_one_sample(profile, x_src, model, args.n)
    save_exp_profiles(np.vstack([profiles ,generated_profiles]), args.o)