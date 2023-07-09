'''
This script exclusivly loads the training data, trains the model on it and saves the trained
model
'''
import numpy as np
from DataSet import scDataSet
import torch
from torch.utils.data import DataLoader
from scTransformer import *
import sys


def load_train_data(path):
    train_data = np.load(path)
    print(f'shape train data={train_data.shape}')
    return train_data

def load_tokens(path):
    tokens = np.load(path, allow_pickle=True)
    return tokens

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

def train(train_data, tokens, mlm_probability, batch_size, n_embd, dim_feedforward, n_layer, n_bin, n_token, n_head,
          learning_rate, n_epoch, device, model_path):
    encode, decode = get_gene_encode_decode(tokens)
    x_src = torch.tensor(encode(tokens))
    # generate data loader
    trainset = scDataSet(train_data, mlm_probability)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    n_train = len(trainset)
    # set up model
    model = TransformerModel(d_model=n_embd,
                             dim_feedforward=dim_feedforward,
                             nlayers=n_layer,
                             n_input_bins=n_bin,
                             n_token=n_token,
                             nhead=n_head)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()), 'parameters')
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(n_epoch):
        for i, (x_val, mask) in enumerate(train_loader):
            loss = model(x_src.to(device), x_val.to(device), mask.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch+1}/{n_epoch}, loss={loss}')
    #print(f'losses: {losses}\nfinal train loss = {losses[-1]}')
    # save model
    #torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    torch.manual_seed(1234)
    # file paths
    cell_type = sys.argv[1]
    path = f'../data/trainset_{cell_type}.npy'
    path_tokens = '../data/tokens.npy'
    model_path = f'../data/model_{cell_type}.pth'
    print(f'cell type: {cell_type}')
    # model parameters
    mlm_probability = 0.4
    batch_size = 256
    n_embd = 10
    dim_feedforward = 100
    n_layer = 2
    n_bin = 100
    n_token = 200
    n_head = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 3e-4
    n_epoch = 200

    train_data = load_train_data(path)
    tokens = load_tokens(path_tokens)
    train(train_data, tokens, mlm_probability, batch_size, n_embd, dim_feedforward, n_layer, n_bin, n_token, n_head,
          learning_rate, n_epoch, device, model_path)




