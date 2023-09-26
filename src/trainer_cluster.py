import sys

from scTransformer import *
from preprocessor import Preprocessor
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scvelo as scv
import scanpy as scp
import numpy as np
from DataSet import scDataSet
import wandb
from typing import Tuple
from GeneVocab import GeneVocab
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self,
                 batch_size: int,
                 n_token: int,
                 n_epoch: int,
                 eval_interval: int,
                 learning_rate: float,
                 eval_iters: int,
                 split: float,
                 n_embd: int,
                 dim_feedforward: int,
                 n_head: int,
                 n_layer: int,
                 n_bin: int,
                 dropout: float,
                 min_counts_genes: int,
                 mlm_probability: int,
                 mean_non_zero_bins: int,
                 path_preprocessed: str,
                 path_tokens: str,
                 path_extern_testset: str,
                 path_vocab: str,
                 seed: Optional[int] = None,
                 subset: Optional[int] = None,
                 test_mode: Optional[bool] = False,
                 ):
        """

        Args:
            batch_size: number of samples that are processed together
            n_token: number of features (genes)
            n_epoch: number of epochs
            eval_interval: number of iterations until next evaluation of tarin and val loss
            learning_rate:
            eval_iters: n iterations to get the mean loss
            n_embd: embedding dimension of transformer encoder
            dim_feedforward: dimension of feed forward layer in attention layer
            n_head: number of heads of the transformer encoder self-attention layer
            n_layer: number of self-attention blocks (encoder)
            dropout: dropout prop
            seed: random seed for reproducability
            subset: fraction of training examples to take for testing
            test_mode: if true, test mode is active. In test mode, only the training split is considered
        """
        # hyperparameters
        self.mlm_probability = mlm_probability
        self.batch_size = batch_size
        self.n_token = n_token
        self.n_epoch = n_epoch
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.eval_iters = eval_iters
        self.split = split
        self.n_embd = n_embd
        self.dim_feedforward = dim_feedforward
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_bin = n_bin
        self.dropout = dropout
        self.min_counts_genes = min_counts_genes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if seed:
            torch.manual_seed(seed)
            print('random seed active')
        self.subset = subset
        self.test_mode = test_mode
        self.mean_non_zero_bins = mean_non_zero_bins
        self.path_preprocessed = path_preprocessed
        self.path_tokens = path_tokens
        self.path_extern_testset = path_extern_testset
        self.path_vocab = path_vocab
        print(f'cuda available: {torch.cuda.is_available()}')
        print(f'device: {self.device}')
        print(torch.zeros(1).cuda())

    def get_gene_encode_decode(self, token_names: list):
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

    def train(self, path: str, config=None) -> None:
        """
        creates and trains the Transformer model
        """
        # open training with wandb
        with wandb.init(config=config):
            # load and
            config = wandb.config
            cell_type = config.cell_type
            vocab = GeneVocab()
            vocab.load_from_file(self.path_vocab)
            n_token = len(vocab.vocab) - 1 # exclude the pad token
            ####### preprocess #######
            if config.do_preprocessing:
                # load_data
                data = scp.read_h5ad(path)
                print(f'data set has shape {data.shape}')
                # data = scv.datasets.bonemarrow(path)
                # data = scv.datasets.pbmc68k()
                min_test_set = 481
                max_test_set = 642
                n_train_set = len(data) - max_test_set
                data.obs.reset_index(inplace=True)
                # split for not omit any cell type
                if cell_type == 'None':
                    n_train = int(0.9 * len(data))
                    n_test = int(len(data) - n_train)
                    idx_all = np.arange(len(data))
                    idx_test_cells = np.random.choice(idx_all, size=n_test, replace=False)
                    idx_rest = list(set(idx_all) - set(idx_test_cells))
                    idx_train_cells = idx_rest
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

                # preprocess
                p = Preprocessor(data, config.n_bin, self.min_counts_genes, n_token, vocab=vocab)
                p.permute()
                p.preprocess()
                p.get_mean_number_of_nonZero_bins()
                #tokens = p.get_gene_tokens()
                data = p.binned_data
                mean_non_zero_bins = p.mean_non_zero_bins
            # preprocessing already done and saved on disc
            else:
                # load preprocessed
                print(f'path to preprocessed data:\n {self.path_preprocessed}')
                data = np.load(self.path_preprocessed)
                print(f'loaded processed data shape: {data.shape}')
                mean_non_zero_bins = self.mean_non_zero_bins
                # split
                n_train = int(0.9 * len(data))
                n_test = int(len(data) - n_train)
                idx_all = np.arange(len(data))
                idx_test_cells = np.random.choice(idx_all, size=n_test, replace=False)
                idx_rest = list(set(idx_all) - set(idx_test_cells))
                idx_train_cells = idx_rest
            # split data
            mask_prob = mean_non_zero_bins*2/n_token
            print(f'number of tokens: {n_token}')
            print(f'number of non zero bins: {mean_non_zero_bins}')
            print(f'masked_genes/all_genes={mask_prob}')
            print(f'randomization: {config.randomization}')
            # adapt mean_non_zero_bins to desired mlm_prob
            n_mask = mean_non_zero_bins*2
            # keep masking prob below given masking prob
            if mask_prob > config.mlm_probability:
                n_mask = int((n_token * config.mlm_probability) // 2)
            print(f'adapted masking={n_mask * 2 / n_token}')
            # use random split for test/train or all of the data as train and a extern set as test
            if config.extern_testset:
                test_data = np.load(self.path_extern_testset)
                trainset = scDataSet(data, config.n_bin, n_mask, n_token)
                testset = scDataSet(test_data, config.n_bin, n_mask, n_token)
            else:
                trainset = scDataSet(data[idx_train_cells], config.n_bin, n_mask, n_token)
                testset = scDataSet(data[idx_test_cells], config.n_bin, n_mask, n_token)
            print(f'len trainset: {len(trainset)}')
            print(f'len testset: {len(testset)}')
            # get gene tokens
            x_src = torch.tensor(vocab.get_token_ids()[1:])  # exclude padding token
            print(f'x_src: {x_src}')
            print(f'x_src type: {type(x_src)}')
            # generate data loaders
            train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=True, num_workers=4)
            n_train = len(trainset)
            # set up model
            # include the pad token again !
            model = TransformerModel(d_model=config.n_emb,
                                     dim_feedforward=config.dim_feedforward,
                                     nlayers=config.n_layer,
                                     n_input_bins=config.n_bin,
                                     n_token=n_token,
                                     nhead=config.n_head)
            wandb.watch(model, log='all', log_freq=np.ceil(n_train / self.batch_size))
            m = model.to(self.device)
            # print the number of parameters in the model
            print(sum(p.numel() for p in m.parameters()), 'parameters')
            # create a PyTorch optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
            # training loop
            for epoch in range(config.n_epoch):
                # print('train..')
                train_loss = []
                for i, (x_val, masked_x_val, attn_mask, mask) in enumerate(train_loader):
                    # evaluate the loss
                    # print(f'shape of mask: {mask.shape}')
                    loss = model(x_src.to(self.device), x_val.to(self.device), masked_x_val.to(self.device),
                                 attn_mask.to(self.device), mask.to(self.device), config.mask_type
                                 , randomize_masked_positions=config.randomization)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())

                # after each epoch, get test loss
                # add if clausal to evaluate only after x epochs
                # print('test..')
                train_loss = np.mean(train_loss)
                test_loss, test_accuracy = self.get_test_loss_and_accuracy(model, test_loader, x_src
                                                                           , config.randomization, config.mask_type)
                print(f'epoch: {epoch + 1}/{config.n_epoch}, train error = {train_loss:.4f}, test error = {test_loss:.4f}'
                      f', accuracy = {test_accuracy:.4f}')
                self.train_log(train_loss, test_loss, test_accuracy, epoch)
                # get model predictions
                # if epoch in check_instances:
                #     val_input, reconstructed_profiles, masks = self.get_valdiation_reconstructions(model, test_loader, x_src)
                #     print(reconstructed_profiles.shape)
                #     print(val_input.shape)
                #     torch.save(reconstructed_profiles, f'../data/predictions_{config.cell_type}_epoch_{epoch}.pt')
                #     torch.save(val_input, f'../data/input_{config.cell_type}_epoch_{epoch}.pt')
                #     torch.save(masks, f'../data/masks_{config.cell_type}_epoch_{epoch}.pt')
                # save model
            torch.save(m.state_dict(), '/mnt/qb/work/claassen/cxb257/models/heart/heart_large_1Mio_pretrain_finetune_ep2.pth')

    def get_test_loss_and_accuracy(self, model: TransformerModel, test_loader: DataLoader,
                                   x_src: Tensor, randomize_masked_positions: bool, mask_type: str) \
            -> Tuple[float, float]:
        """
        uses a whole run of the validation set to compute the accuracy
        """
        print('get_test_loss')
        model.eval()
        acc = []
        loss = []
        for i, (x_val, masked_x_val, attn_mask, mask) in enumerate(test_loader):
            l, a = model(x_src.to(self.device), x_val.to(self.device), masked_x_val.to(self.device),
                         attn_mask.to(self.device), mask.to(self.device), mask_type, get_accuracy=True,
                         randomize_masked_positions=randomize_masked_positions)
            loss.append(l.item())
            acc.append(a.item())
        model.train()
        return (float(np.mean(loss)), float(np.mean(acc)))

    def train_log(self, loss: float, test_loss: float, test_accuracy: float, epoch: int) -> None:
        """
        parameters tracked by wandb
        """
        wandb.log({"epoch": epoch + 1, "train_loss": loss, "test_loss": test_loss, "test_accuracy": test_accuracy}
                  , step=epoch)

    def get_valdiation_reconstructions(self, model: TransformerModel, test_loader: DataLoader, x_src: Tensor) -> Tensor:
        model.eval()
        reconstructed_profiles = []
        val = []
        masks = []
        for i, (x_val, mask) in enumerate(test_loader):
            # evaluate the loss
            reconstructed_profiles.append(
                model(x_src.to(self.device), x_val.to(self.device), mask.to(self.device), True))
            val.append(x_val)
            masks.append(mask)
        model.train()
        reconstructed_profiles = torch.cat(reconstructed_profiles, dim=0)
        masks = torch.cat(masks, dim=0)
        val = torch.cat(val, dim=0)
        return val, reconstructed_profiles, masks


if __name__ == '__main__':
    # hyperparameters
    batch_size = 264
    n_token = 200
    n_epoch = 150
    eval_interval = 100
    learning_rate = 3e-4
    eval_iters = 10
    split = 0.9
    n_embd = 10
    dim_feedforward = 100
    n_head = 2
    n_layer = 2
    n_bin = 200
    dropout = 0.5
    min_counts_genes = 10
    mlm_probability = None
    seed = 1234
    mean_non_zero_bins = 24
    path_preprocessed = '/mnt/qb/work/claassen/cxb257/data/preprocessed/heart/heart_1Mio_pretrain_finetune_1500_200.npy'
    path_tokens = '/mnt/qb/work/claassen/cxb257/data/preprocessed/heart/token_1500.npy'
    path_extern_testset = '/mnt/qb/work/claassen/cxb257/data/preprocessed/heart/heart_100K_preprocessed_1500_testset.npy'
    path_vocab = '/mnt/qb/work/claassen/cxb257/data/heart/heart_1Mio_1500_vocab.json'
    #dataset_path = '/mnt/qb/work/claassen/cxb257/data/Pancreas/endocrinogenesis_day15.h5ad'
    dataset_path = '/mnt/qb/work/claassen/cxb257/data/heart/heart_100K.h5ad'
    # create model
    trainer = Trainer(
        batch_size=batch_size,
        n_token=n_token,
        n_epoch=n_epoch,
        eval_interval=eval_interval,
        learning_rate=learning_rate,
        eval_iters=eval_iters,
        split=split,
        n_embd=n_embd,
        dim_feedforward=dim_feedforward,
        n_head=n_head,
        n_layer=n_layer,
        n_bin=n_bin,
        dropout=dropout,
        min_counts_genes=min_counts_genes,
        mlm_probability=mlm_probability,
        seed=seed,
        subset=None,
        test_mode=False,
        mean_non_zero_bins=mean_non_zero_bins,
        path_preprocessed=path_preprocessed,
        path_tokens=path_tokens,
        path_extern_testset=path_extern_testset,
        path_vocab=path_vocab
    )

    trainer.train(dataset_path)
