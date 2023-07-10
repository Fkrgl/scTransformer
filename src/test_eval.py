'''
this script exclusively loads the test data and the trained model. The model is evaluated
on the test data.
'''
import numpy as np
from scTransformer import *
from torch.utils.data import DataLoader
from DataSet import scDataSet
import sys

def load_test_data(path):
    test_data = np.load(path)
    print(f'shape test data={test_data.shape}')
    return test_data

def load_model(path, n_embd, dim_feedforward, n_layer, n_bin, n_token, n_head):
    model = TransformerModel(
                         d_model=n_embd,
                         dim_feedforward=dim_feedforward,
                         nlayers=n_layer,
                         n_input_bins=n_bin,
                         n_token=n_token,
                         nhead=n_head)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

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
def evaluate_model(model: TransformerModel, device, mlm_probability, batch_size, tokens):
    """
    uses a whole run of the validation set to compute the accuracy
    """
    encode, decode = get_gene_encode_decode(tokens)
    x_src = torch.tensor(encode(tokens))
    testset = scDataSet(test_data, 22, n_token)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    model.eval()
    acc = []
    loss = []
    for i, (x_val, mask) in enumerate(test_loader):
        # evaluate the loss
        l, a = model(x_src.to(device), x_val.to(device), mask.to(device), get_accuracy=True)
        loss.append(l.item())
        acc.append(a.item())
    print(f'mean test accurracy={np.mean(acc):.4f}\nmean test loss={np.mean(loss):.4f}')

if __name__ == '__main__':
    # file paths
    cell_type = sys.argv[1]
    path = f'../data/testset_{cell_type}.npy'
    path_model = f'../data/model_{cell_type}.pth'
    path_tokens = '../data/tokens.npy'
    print(f'cell type: {cell_type}')
    # model parameters
    n_embd = 10
    mlm_probability = 0.4
    dim_feedforward = 100
    n_layer = 2
    n_bin = 100
    n_token = 200
    n_head = 2
    batch_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data = load_test_data(path)
    model = load_model(path_model, n_embd, dim_feedforward, n_layer, n_bin, n_token, n_head)
    model = model.to(device)
    tokens = load_tokens(path_tokens)
    evaluate_model(model, device, mlm_probability, batch_size, tokens)

