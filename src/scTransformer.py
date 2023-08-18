import sys

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict
# encoder layers are based on paper "Attention is all you need"
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchmetrics import Accuracy


class TransformerModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_token: int,
                 nhead: int,
                 dim_feedforward: int,
                 nlayers: int,
                 n_input_bins: int,
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
        self.nhead = nhead
        self.activation = "relu"
        self.n_input_bins = n_input_bins
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # define the gene encoder
        self.encoder = GeneEncoder(n_token, d_model)
        self.value_encoder = ValueEncoder(self.n_input_bins, d_model)
        # define the transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                 nhead=self.nhead,
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
                attn_mask: Tensor,
                key_padding_mask: Tensor,
                mask_type: str,
                get_accuracy: bool
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
        # for test runs, randomize all value embeddings of masked genes
        # if get_accuracy:
        #     values = self.randomize_maked_position_encodeings(values, key_padding_mask)
        # combined embedding (broadcasting)
        total_embedding = src + values
        if mask_type == 'src_key_padding_mask':
            output = self.transformer_encoder(total_embedding, src_key_padding_mask=key_padding_mask)  # , mask=attn_mask
        elif mask_type == 'attn_mask':
            # before the attn_mask can be fed into the encoder it has to transformed to shape (N*n_head, L, S):
            # N = batch size, n_head =n umber of heads, L = length of source sequence, S = length of target sequence
            attn_mask = torch.vstack([attn_mask] * self.nhead)
            output = self.transformer_encoder(total_embedding, mask=attn_mask)
        elif mask_type == 'None':
            output = self.transformer_encoder(total_embedding)

        return output.to(self.device)  # (batch, seq_len, embsize)

    def randomize_maked_position_encodeings(self, output, key_padding_mask):
        """
        function takes the output of the transformer and randomizes the encodeings for all masked genes
        """
        # print(f'output:\n{output}')
        rand = torch.FloatTensor(output.shape[0], output.shape[1], output.shape[2]).uniform_(-5, 5).to(self.device)
        # print(f'rand\n{rand}')
        output[key_padding_mask] = rand[key_padding_mask]
        # print(f'mask\n{key_padding_mask}')
        # print(f'new output\n{output}')
        return output


    def randomize_masked_positions(self, values, key_padding_mask):
        '''
        each masked value (TRUE in key_padding_mask) is assigned a random value. Unmasked values stay the same
        Args:
            values:
            key_padding_mask:

        Returns:

        '''
        values = values.cpu()
        values = values.detach().numpy()
        key_padding_mask = key_padding_mask.cpu()
        key_padding_mask = key_padding_mask.detach().numpy()
        # i = 5
        # print(f'original value:\n{values[i]}')
        random_val = np.random.choice(np.arange(self.n_input_bins), size=values.shape)
        values[key_padding_mask] = random_val[key_padding_mask]
        values = torch.tensor(values).to(self.device)

        return values

    def zero_masked_positions(self, values, key_padding_mask):
        '''
        each masked value (TRUE in key_padding_mask) is assigned a random value. Unmasked values stay the same
        Args:
            values:
            key_padding_mask:

        Returns:

        '''
        values = values.cpu()
        values = values.detach().numpy()
        key_padding_mask = key_padding_mask.cpu()
        key_padding_mask = key_padding_mask.detach().numpy()
        # i = 5
        # print(f'original value:\n{values[i]}')
        values[key_padding_mask] = 0
        # print(f'mask:\n{key_padding_mask[i]}')
        # print(f'modified value:\n{values[i]}')
        # print(f'random:\n{random_val[i]}')
        values = torch.tensor(values).to(self.device)

        return values


    def forward(self,
                src: Tensor,
                values: Tensor,
                attn_mask: Tensor,
                key_padding_mask: Tensor,
                mask_type: str,
                get_reconstruction: bool = False,
                get_accuracy: bool = False,
                randomize_masked_positions: bool = False
                ):
        """

        Args:
            src: token ids, shape (batch_size, seq_len)
            values: token values, shape (batch_size, seq_len)
            key_padding_mask: mask for src, shape (batch_size, seq_len)

        Returns:
            Tensor of expression prediction
        """
        # if test samples are processed, randomize masked positions
        # print(f'values: {values[0]}')
        labels = values.clone().to(self.device)
        if not get_accuracy and randomize_masked_positions:
            # print('randomize')
            values = self.randomize_masked_positions(values, key_padding_mask)
        # print(f'values: {values[0]}')
        # print(f'labels: {labels[0]}')
        transformer_output = self._encode(src, values, attn_mask, key_padding_mask, mask_type, get_accuracy)
        # if get_accuracy:
        #     transformer_output = self.randomize_maked_position_encodeings(transformer_output, key_padding_mask)
        # each gene embedding gets its own expression value prediction
        mlm_output = self.decoder(transformer_output) # (batch, seq_len, n_bin)
        # get only vectors of masked genes
        masked_pred_exp, masked_label_exp = self.get_masked_exp(mlm_output, labels, key_padding_mask)
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

        if get_accuracy and get_reconstruction:
            output = (loss, acc_value, mlm_output)

        return output

    def get_masked_exp(self, mlm_output, values, key_padding_mask):
        """
        calculates the loss per cell using only the masked positions
        Args:
            mlm_output: predicted expr (batch, seq_len, n_bin)
            key_padding_mask: gene expression mask: (batch, seq_len)

        Returns:
            predicted and ground truth expression of masked genes
        """
        masked_pred_exp = torch.Tensor([]).to(self.device)
        masked_label_exp = torch.Tensor([]).to(self.device)
        # print('get masked expressions')
        ###### braucen wir diesen loop ueberhaupt noch?!
        for i in range(mlm_output.shape[0]):
            pred = mlm_output[i]
            value = values[i]
            mask = key_padding_mask[i]
            masked_pred = pred[mask]
            true_bins = value[mask]
            masked_pred_exp = torch.cat((masked_pred_exp, masked_pred.to(self.device)), dim=0)
            masked_label_exp = torch.cat((masked_label_exp, true_bins.to(self.device)))
        # print(f'masked_label_exp[0]:\n{masked_label_exp}')
        # print(f'masked_pred_exp[0]:\n{masked_pred_exp}')
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
        # print(f'decoder_output_shape: {decoder_output.shape}')
        # get softmax
        s = nn.Softmax(dim=0)
        softmax_output = s(decoder_output)
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




