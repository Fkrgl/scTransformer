import sys
from scTransformer import *
from preprocessor import Preprocessor
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scvelo as scv
import numpy as np
from DataSet import scDataSet
import wandb
from typing import Tuple
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
                 seed: Optional[int] = None,
                 subset: Optional[int] = None,
                 test_mode: Optional[bool] = False
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
        print(f'cuda available: {torch.cuda.is_available()}')
        print(f'device: {self.device}')
        print(torch.zeros(1).cuda())

    # +---------------------------+ prepare the data +---------------------------+
    def load_data(self):
        """
        loads the data set
        """
        if self.subset:
            return scv.datasets.pancreas()[:self.subset]
        return scv.datasets.pancreas()

    def preprocess_data(self, data, bins, min_counts_genes, n_hvg):
        """
        perfoms all preprocessing steps for scRNA data
        """
        p = Preprocessor(data, bins, min_counts_genes, n_hvg)
        # permute for random split into train and val set later
        p.permute()
        p.preprocess()
        return p

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
        with wandb.init(config=config, dir='/mnt/qb/work/claassen/cxb257/wandb'):
            # load and
            config = wandb.config
            print(f'mlm_prob: {config.mlm_probability}')
            print(f'mlm_prob: {config.mlm_probability}')
            cell_type = config.cell_type
            ####### preprocess #######
            # load_data
            data = scv.datasets.pancreas(path)
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
                idx_train_cells = np.random.choice(np.array(idx_rest), size=n_train, replace=False)
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

            # split for omit cell type
            else:
                idx_test_cells = data.obs[data.obs.clusters == cell_type].index.values
                idx_train_cells = data.obs[data.obs.clusters != cell_type].index.values
                idx_test_cells = np.random.choice(idx_test_cells, size=min_test_set, replace=False)
                idx_train_cells = np.random.choice(idx_train_cells, size=n_train_set, replace=False)

            # check with plot
            # umap = data.obsm['X_umap']
            # plt.scatter(umap[idx_test_cells, 0], umap[idx_test_cells, 1], c='red', alpha=0.2, label='cut')
            # plt.scatter(umap[idx_train_cells, 0], umap[idx_train_cells, 1], c='blue', alpha=0.1, label='uncut')
            # plt.title(f'{cell_type}')
            # plt.legend()
            # file_path = f'../fig/cut_out_{cell_type}.png'
            # plt.savefig(file_path)
            # # check split
            # print(f'trainset clusters:\n{data[idx_train_cells].obs.clusters}')
            # print()
            # print(f'testset clusters:\n{data[idx_test_cells].obs.clusters}')
            # print()
            # preprocess
            p = Preprocessor(data, config.n_bin, self.min_counts_genes, self.n_token)
            #p.permute()
            p.preprocess()
            p.get_mean_number_of_nonZero_bins()
            tokens = p.get_gene_tokens()
            data = p.binned_data
            # split data
            print(f'number of tokens: {self.n_token}')
            print(f'number of non zero bins: {p.mean_non_zero_bins}')
            trainset = scDataSet(data[idx_train_cells], p.mean_non_zero_bins, self.n_token)
            testset = scDataSet(data[idx_test_cells], p.mean_non_zero_bins, self.n_token)
            # encode gene names
            n_token = len(tokens)
            encode, decode = self.get_gene_encode_decode(tokens)
            x_src = torch.tensor(encode(tokens))
            # generate data loaders
            train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            n_train = len(trainset)
            # set up model
            model = TransformerModel(d_model=self.n_embd,
                                     dim_feedforward=self.dim_feedforward,
                                     nlayers=self.n_layer,
                                     n_input_bins=config.n_bin,
                                     n_token=n_token,
                                     nhead=self.n_head)
            wandb.watch(model, log='all', log_freq=np.ceil(n_train/self.batch_size))
            m = model.to(self.device)
            # print the number of parameters in the model
            print(sum(p.numel() for p in m.parameters()), 'parameters')
            # create a PyTorch optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
            # generate masks:
            # masks = []
            # values = []
            # for i, (x_val, mask) in enumerate(train_loader):
            #     masks.append(mask)
            #     values.append(x_val)
            # masks = torch.cat(masks, dim=0)
            # values = torch.cat(values, dim=0)
            # torch.save(masks, f'../data/test_masks.pt')
            # torch.save(values, f'../data/test_values.pt')
            # training loop
            for epoch in range(self.n_epoch):
                for i, (x_val, mask) in enumerate(train_loader):
                    # evaluate the loss
                    # print(f'shape of mask: {mask.shape}')
                    loss = model(x_src.to(self.device), x_val.to(self.device), mask.to(self.device))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                # after each epoch, get test loss
                # add if clausal to evaluate only after x epochs
                test_loss, test_accuracy = self.get_test_loss_and_accuracy(model, test_loader, x_src)
                print(f'epoch: {epoch + 1}/{self.n_epoch}, train error = {loss:.4f}, test error = {test_loss:.4f}'
                      f', accuracy = {test_accuracy:.4f}')
                self.train_log(loss, test_loss, test_accuracy, epoch)
                # get model predictions
                # if epoch in check_instances:
                #     val_input, reconstructed_profiles, masks = self.get_valdiation_reconstructions(model, test_loader, x_src)
                #     print(reconstructed_profiles.shape)
                #     print(val_input.shape)
                #     torch.save(reconstructed_profiles, f'../data/predictions_{config.cell_type}_epoch_{epoch}.pt')
                #     torch.save(val_input, f'../data/input_{config.cell_type}_epoch_{epoch}.pt')
                #     torch.save(masks, f'../data/masks_{config.cell_type}_epoch_{epoch}.pt')
            # save model
            # torch.save(m.state_dict(), '../data/model_3.pth')



    def get_test_loss_and_accuracy(self, model: TransformerModel, test_loader: DataLoader, x_src: Tensor) \
            -> Tuple[float, float]:
        """
        uses a whole run of the validation set to compute the accuracy
        """
        model.eval()
        acc = []
        loss = []
        for i, (x_val, mask) in enumerate(test_loader):
            # evaluate the loss
            l, a = model(x_src.to(self.device), x_val.to(self.device), mask.to(self.device), get_accuracy=True)
            loss.append(l.item())
            acc.append(a.item())
        model.train()
        return (float(np.mean(loss)), float(np.mean(acc)))

    def train_log(self, loss: float, test_loss: float, test_accuracy:float, epoch: int) -> None:
        """
        parameters tracked by wandb
        """
        wandb.log({"epoch": epoch+1, "train_loss": loss, "test_loss": test_loss, "test_accuracy": test_accuracy}
                  , step=epoch)

    def get_valdiation_reconstructions(self, model: TransformerModel, test_loader: DataLoader, x_src: Tensor) -> Tensor:
        model.eval()
        reconstructed_profiles = []
        val = []
        masks = []
        for i, (x_val, mask) in enumerate(test_loader):
            # evaluate the loss
            reconstructed_profiles.append(model(x_src.to(self.device), x_val.to(self.device), mask.to(self.device), True))
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
    n_epoch = 200
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
    dataset_path = '../data/Pancreas/endocrinogenesis_day15.h5ad'

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
        test_mode=False
    )

    trainer.train(dataset_path)