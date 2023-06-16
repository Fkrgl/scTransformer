from scTransformer import *
from preprocessor import Preprocessor
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scvelo as scv
import numpy as np
from DataSet import scDataSet
import wandb


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

    def split_data(self, p: Preprocessor) -> tuple:
        """
        splits the data set into training and validation data
        """
        n = p.binned_data.shape[0]
        n_train = int(self.split * n)
        train_data = torch.tensor(p.binned_data[:n_train])
        val_data = torch.tensor(p.binned_data[n_train:])
        return train_data, val_data

    # batch loading
    def get_batch(self, data: Tensor):
        """
        generates a batch by random
        Args:
            data (Tensor): either train or validation data set

        Returns:
            x (Tensor): random subset of data
        """
        n = data.shape[0]
        rand_idx = np.random.randint(low=0, high=n, size=self.batch_size)
        batch_values = data[rand_idx]
        return batch_values

    def get_mask(self, expressions: torch.Tensor, mlm_probability: float = 0.15) -> torch.Tensor:
        """
        generates a mask for a proportion of genes in the input data. The masks genes are predicted later in the training
        process. More information here:
        https://github.com/pytorch/pytorch/blob/11f1014c05b902d3eef0fe01a7c432f818c2bdfe/torch/nn/functional.py#L3892
        Args:
            expressions: expression matrix (batch, seq_length)
            mlm_probability: probability fo a gene to get masked

        Returns:
            Boolean Tensor of shape (batch, n_token) with True where genes should be masked

        """
        shape = expressions.shape
        probability_matrix = torch.full(shape, mlm_probability)
        mask = torch.bernoulli(probability_matrix).bool()
        return mask

    def train(self, path: str, config: dict, wandb_project: str) -> None:
        """
        creates and trains the Transformer model
        """
        # open training with wandb
        with wandb.init(project=wandb_project, config=config):
            # load and
            config = wandb.config
            print(f'mlm_prob: {config.mlm_probability}')
            data = scDataSet(path, self.n_bin, self.min_counts_genes, self.n_token, config.mlm_probability)
            # encode gene names
            gene_tokens = data.gene_tokens
            n_token = len(gene_tokens)
            encode, decode = self.get_gene_encode_decode(gene_tokens)
            x_src = torch.tensor(encode(gene_tokens))
            # split data
            trainset, testset = random_split(data, [self.split, 1 - self.split])
            train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            n_train = len(trainset)
            # set up model
            model = TransformerModel(d_model=self.n_embd,
                                     dim_feedforward=self.dim_feedforward,
                                     nlayers=self.n_layer,
                                     n_input_bins=self.n_bin,
                                     n_token=n_token,
                                     nhead=self.n_head)
            wandb.watch(model, log='all', log_freq=np.ceil(n_train/self.batch_size))
            m = model.to(self.device)
            # print the number of parameters in the model
            print(sum(p.numel() for p in m.parameters()), 'parameters')
            # create a PyTorch optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

            for epoch in range(self.n_epoch):
                for i, (x_val, mask) in enumerate(train_loader):
                    # evaluate the loss
                    # print(f'shape of mask: {mask.shape}')
                    loss = model(x_src, x_val, mask)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                # after each epoch, get test loss
                # add if clausal to evaluate only after x epochs
                test_loss = self.get_test_loss(model, test_loader, x_src)
                print(f'epoch: {epoch + 1}/{self.n_epoch}, train error = {loss:.4f}, test error = {test_loss:.4f}')
                self.train_log(loss, test_loss, epoch)
                # if last epoch is reached, get validation reconstructions and perform Classifier2SampleTest
                if epoch == self.n_epoch - 1:
                    reconstructed_profiles, masks = self.get_valdiation_reconstructions(model, test_loader, x_src)
                    val_input = testset[:][0]
                    print(reconstructed_profiles.shape)
                    print(val_input.shape)
                    # torch.save(reconstructed_profiles, '../data/reconstructed_profiles_50_epochs.pt')
                    # torch.save(val_input, '../data/val_input_50_epochs.pt')
                    # torch.save(masks, '../data/masks_50_epochs.pt')

    def get_test_loss(self, model: TransformerModel, test_loader: DataLoader, x_src: Tensor) -> float:
        """
        runs whole training set for evaluating the test error
        """
        model.eval()
        losses = []
        for i, (x_val, mask) in enumerate(test_loader):
            # evaluate the loss
            losses.append(model(x_src, x_val, mask).item())
        model.train()
        return np.mean(losses)

    def train_log(self, loss: float, test_loss: float, epoch: int) -> None:
        """
        parameters tracked by wandb
        """
        wandb.log({"epoch": epoch+1, "train_loss": loss, "test_loss": test_loss}, step=epoch)

    def get_valdiation_reconstructions(self, model: TransformerModel, test_loader: DataLoader, x_src: Tensor) -> Tensor:
        model.eval()
        reconstructed_profiles = []
        masks = []
        for i, (x_val, mask) in enumerate(test_loader):
            # evaluate the loss
            reconstructed_profiles.append(model(x_src, x_val, mask, True))
            masks.append(mask)
        model.train()
        reconstructed_profiles = torch.cat(reconstructed_profiles, dim=0)
        masks = torch.cat(masks, dim=0)
        return reconstructed_profiles, masks

# +-----------------------+ test code +-----------------------+
# project name needs to be ths same as sweep name
wandb_project = 'sweep_mlm_prob4'

# hyperparameters
batch_size = 10
n_token = 200
n_epoch = 20
eval_interval = 100
learning_rate = 3e-4
eval_iters = 10
split = 0.9
n_embd = 10
dim_feedforward = 100
n_head = 2
n_layer = 2
n_bin = 10
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

# set up wandb
wandb.login(key='c697f25b0981fe76f7062d1c3fec4872f9f9c469')
# define hyperparameters for wandb
config = dict(
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
    dataset=dataset_path
)
# start training
#trainer.train(dataset_path, config, wandb_project)
# make the first try of grid search smaller ! First we are most interested in mlm_probability
sweep_configuration = {
    'program': 'trainer.py',
    'method': 'grid',
    'name': wandb_project,
    'metric': {
        'goal': 'minimize',
        'name': 'test_loss'
        },
    'parameters': {
        'mlm_probability' : {'values': [0.05, 0.5, 0.95]},
     }
}
# generate a sweep id
sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandb_project)
# create an agant that manages the hyp. param. search
# agent expects a funtcion as input, that is why we need the lambda call
wandb.agent(sweep_id=sweep_id, function=lambda: trainer.train(dataset_path, config, wandb_project))

