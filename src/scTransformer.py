import sys

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict
# encoder layers are based on paper "Attention is all you need"
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchmetrics import Accuracy
import pandas as pd


class TransformerModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_token: int,
                 nhead: int,
                 dim_feedforward: int,
                 nlayers: int,
                 n_input_bins: int,
                 most_freq_hvg_bins,
                 dropout: Optional[float] = None,
                 pad_token: str = "<pad>",
                 pad_value: int = 0,
                 ):
        """

        Args:
            d_model: dimension of the embedding vectors
            n_token: size of the vocabulary (set of genes)
            nhead: number of attention heads
            dim_feedforward: dim of the hidden layer in the feed forward network of the encoder
            nlayers: number of attention layers
            dropout:
        """
        super().__init__()
        # parameters
        self.d_model = d_model
        self.activation = "relu"
        self.n_input_bins = n_input_bins
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.most_freq_hvg_bins = most_freq_hvg_bins
        self.n_token = n_token

        # define the gene encoder
        self.encoder = GeneEncoder(n_token, d_model)
        self.value_encoder = ValueEncoder(self.n_input_bins, d_model)
        # define the transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                 nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 activation=self.activation,
                                                 batch_first=True,
                                                 norm_first=False
                                                 )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # one decoder for all genes
        self.decoder = ExprDecoder(self.d_model, self.n_input_bins)
        self.index = np.arange(n_token)

        self.loss = nn.CrossEntropyLoss()
        self.acc_function = Accuracy(task='multiclass', num_classes=self.n_input_bins)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: check if this initialization is helpful and shall we apply to all?
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(self,
                src: Tensor,
                values: Tensor,
                key_padding_mask: Tensor,
                ) -> Tensor:
        """

        Args:
            src:
            values:
            src_key_padding_mask:

        Returns:
            embedding tensor: (batch, seq_len, embsize)
        """
        # gene embedding
        src = self.encoder(src)
        values = self.value_encoder(values)
        # combined embedding (broadcasting)
        total_embedding = src + values
        output = self.transformer_encoder(total_embedding, src_key_padding_mask=key_padding_mask)

        return output.to(self.device)  # (batch, seq_len, embsize)

    def forward(self,
                src: Tensor,
                values: Tensor,
                key_padding_mask: Tensor,
                get_reconstruction: bool = False,
                get_accuracy: bool = False):
        """

        Args:
            src: token ids, shape (batch_size, seq_len)
            values: token values, shape (batch_size, seq_len)
            key_padding_mask: mask for src, shape (batch_size, seq_len)

        Returns:
            Tensor of expression prediction
        """

        transformer_output = self._encode(src, values, key_padding_mask)
        # each gene embedding gets its own expression value prediction
        mlm_output = self.decoder(transformer_output) # (batch, seq_len, n_bin)
        # get only vectors of masked genes
        #masked_pred_exp, masked_label_exp = self.get_masked_exp(mlm_output, values, key_padding_mask)
        ############### 1) predict only zero
        # masked_pred_exp = torch.zeros_like(masked_pred_exp)
        # masked_pred_exp[:,0] = 1
        # masked_pred_exp = masked_pred_exp.requires_grad_(True)
        ############### 2) predict bin randomly uniform (100 bins with bin values 0 to 99)masked_pred_exp
        # masked_pred_exp = np.zeros(shape=(masked_pred_exp.shape[0], masked_pred_exp.shape[1]))
        # rand = np.random.choice(np.arange(self.n_input_bins), size=masked_pred_exp.shape[0])
        # for i in range(masked_pred_exp.shape[0]):
        #     masked_pred_exp[i][rand[i]] = 1
        # masked_pred_exp = torch.tensor(masked_pred_exp, dtype=float).requires_grad_(True)
        ################ 3) predict most frequent bin
        masked_pred_exp, masked_label_exp = self.get_masked_exp2(mlm_output, values, key_padding_mask)
        # print(f'masked_pred_exp: {masked_pred_exp}\n{masked_pred_exp.shape}')
        # print(f'masked_label_exp: {masked_label_exp}\n{masked_label_exp.shape}')
        # sys.exit()
        loss = self.loss(masked_pred_exp, masked_label_exp)
        output = loss
        # get reconstructed profiles
        if get_reconstruction:
            output = mlm_output
        # accuracy here
        # accuracy is only computed using masked values
        if get_accuracy:
            acc_value = self.acc_function(masked_pred_exp, masked_label_exp)
            output = (loss, acc_value)

        return output

    def get_masked_exp2(self, mlm_output, values, key_padding_mask):
        """
        calculates the loss per cell using only the masked positions
        Args:
            mlm_output: predicted expr (batch, seq_len)
            key_padding_mask: gene expression mask: (batch, seq_len)

        Returns:
            predicted and ground truth expression of masked genes
        """
        masked_pred_exp = []
        masked_label_exp = torch.Tensor([]).to(self.device)
        Gen_idx = []
        targets = []
        predictions = []
        ###### braucen wir diesen loop ueberhaupt noch?!
        for i in range(mlm_output.shape[0]):
            pred = mlm_output[i]
            value = values[i]
            mask = key_padding_mask[i]
            true_bins = value[mask]
            idx = np.arange(self.n_token)
            # print(f'{masked_pred}\n{masked_pred.shape}')
            # print(f'{true_bins}\n{true_bins.shape}')

            idx_masked = idx[mask]
            # print(mask)
            # print(f'indices of masked positions:\n{idx_masked}')
            for j in range(len(true_bins)):
                # print(f'j={j}')
                masked_pred = torch.tensor(np.zeros(shape=self.n_input_bins))
                masked_pred[0] = 1
                # print(f'masked_pred:\n{masked_pred}')
                pred_val = 0
                true_val = true_bins[j]
                # print(f'true expression bin: {true_val}')
                if true_val != 0:
                    # print(f'correct masked_pred..')
                    masked_pred[0] = 0
                    pred_val = self.most_freq_hvg_bins[idx_masked[j]]
                    # print(f'Gen with index: {idx_masked[j]} is assigned to its most frequent bin {pred_val}')
                    masked_pred[pred_val] = 1
                    # print(f'test: {np.where(masked_pred == 1)}')
                    # print(f'masked_pred:\n{masked_pred}')
                #     print(f'corrected masked_pred:\n{masked_pred}')
                # print(self.most_freq_hvg_bins)
                Gen_idx.append(idx_masked[j])
                targets.append(true_val.item())
                predictions.append(pred_val)
                # print(f'Gen {idx_masked[j]}: {true_val} -> {pred_val}')
                masked_pred_exp.append(masked_pred.to(self.device))

            masked_label_exp = torch.cat((masked_label_exp, true_bins.to(self.device)))
            # print(f'masked_pred= {masked_pred}')
        masked_pred_exp = torch.vstack(masked_pred_exp)

        # print(f'{masked_label_exp.shape}')
        # print(f'{masked_pred_exp.shape}')
        # print()
        # print(mlm_output)
        # for i in range(10):
        #     print(f'label: {masked_label_exp[i]}\n prediction:\n {masked_pred_exp[i]}')
        #     print(f'values: {values[i]}')
        #     print(f'values[mask] : \n{values[i][key_padding_mask[i]]}')
        # save_frame = pd.DataFrame({'gen_idx' : Gen_idx, 'predictions' : predictions, 'targets': targets})
        # print(save_frame)
        # save_frame.to_csv('acc_most_frequent.csv', index=False)
        return masked_pred_exp.requires_grad_(True), masked_label_exp.to(dtype=torch.long)

    def get_masked_exp(self, mlm_output, values, key_padding_mask):
        """
        calculates the loss per cell using only the masked positions
        Args:
            mlm_output: predicted expr (batch, seq_len)
            key_padding_mask: gene expression mask: (batch, seq_len)

        Returns:
            predicted and ground truth expression of masked genes
        """
        masked_pred_exp = torch.Tensor([]).to(self.device)
        masked_label_exp = torch.Tensor([]).to(self.device)
        ###### braucen wir diesen loop ueberhaupt noch?!
        # Gen_idx = []
        # targets = []
        # predictions = []
        idx = np.arange(self.n_token)
        for i in range(mlm_output.shape[0]):
            pred = mlm_output[i]
            value = values[i]
            mask = key_padding_mask[i]
            masked_pred = pred[mask]
            true_bins = value[mask]
            # idx_masked = idx[mask]
            # for j in range(len(true_bins)):
            #     Gen_idx.append(idx_masked[j])
            #     targets.append(true_bins[j].item())
            #     predictions.append(masked_pred[j].detach().numpy())

            #print(f'masked_pred= {masked_pred}')
            # print(f'{masked_pred}\n{masked_pred.shape}')
            # print(f'{true_bins}\n{true_bins.shape}')
            masked_pred_exp = torch.cat((masked_pred_exp, masked_pred.to(self.device)), dim=0)
            masked_label_exp = torch.cat((masked_label_exp, true_bins.to(self.device)))
        # print(f'{masked_label_exp}\n{masked_label_exp.shape}')
        # print(f'{masked_pred_exp}\n{masked_pred_exp.shape}')
        # save_frame = pd.DataFrame({'gen_idx' : Gen_idx, 'predictions' : predictions, 'targets': targets})
        # print(save_frame)
        # save_frame.to_csv('acc_normal.csv', index=False)
        return masked_pred_exp.requires_grad_(True), masked_label_exp.to(dtype=torch.long)

    def generate(self,
                 src: Tensor,
                 values: Tensor,
                 bins):
        n_gen = self.n_token
        # embedd genes of target cell and feed in transformer encoder
        encoder_output = self._encode(src, values)
        # decode transformer encoded gene vectors
        decoder_output = self.decoder(encoder_output)
        print(f'decoder_output_shape: {decoder_output.shape}')
        # get softmax
        s = nn.Softmax(dim=0)
        softmax_output = s(decoder_output)
        print(f'softmax:\n{softmax_output}')
        # sample from softmax
        sample_profile = np.zeros(shape=n_gen)
        for i in range(n_gen):
            prob = softmax_output[:, i]
            bin = np.random.choice(bins, size=1, p=prob)
            sample_profile[i] = bin
        return sample_profile

class GeneEncoder(nn.Module):
    """
    given a gene expression vector of a cell, the function embedds the vector into lower
    gene embedding dimension. Insted of using a pretrained embedding such as gene2vec, the
    model learns its own gene embeddings. Each gene gets its own gene id which is than fed
    into the embedding layer
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None):
        super().__init__()
        # dict that saves the embedding vectors
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx
                                      )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x

class ValueEncoder(nn.Module):
    """
    encodes the discrete, value binned expression vectors
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x

class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_bins: int,
        use_batch_labels: bool = False
    ):
        super().__init__()
        # how big should be the hidden layer?
        d_in = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, n_bins)
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        x is the output of the transformer (batch, seq_len, d_model)
        The function predicts an expression value for each gene though only the masked genes
        will used in the loss function
        """
        # linear layer excepts tensors of arbitray shape as input, only the last dimension (d_model) has to match with
        # the input dimension of the linear layer. While going through the layers, only the last dimension will change
        # to the specified output dimension of the linear layer
        pred_value = self.fc(x)  # (batch, seq_len, 1)
        # pred_value = pred_value.squeeze(-1)  # (batch, seq_len)
        return pred_value




