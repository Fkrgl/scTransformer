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
        self.activation = "relu"
        self.n_input_bins = n_input_bins

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

        return output  # (batch, seq_len, embsize)

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
        masked_pred_exp, masked_label_exp = self.get_masked_exp(mlm_output, values, key_padding_mask)
        loss = self.loss(masked_label_exp, masked_pred_exp)
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

    def get_masked_exp(self, mlm_output, values, key_padding_mask):
        """
        calculates the loss per cell using only the masked positions
        Args:
            mlm_output: predicted expr (batch, seq_len)
            key_padding_mask: gene expression mask: (batch, seq_len)

        Returns:
            predicted and ground truth expression of masked genes
        """
        masked_pred_exp = torch.Tensor([])
        masked_label_exp = torch.Tensor([])
        for i in range(mlm_output.shape[0]):
            pred = mlm_output[i]
            value = values[i]
            mask = key_padding_mask[i]
            masked_pred = pred[mask]
            true_bins = value[mask]
            one_hot_ture_bins = self.get_one_hot(true_bins)
            masked_pred_exp = torch.cat((masked_pred_exp, masked_pred))
            masked_label_exp = torch.cat((masked_label_exp, one_hot_ture_bins))
        return masked_pred_exp, masked_label_exp

    def get_one_hot(self, true_bins: list) -> Tensor:
        '''
        given a list of true bins, function creates a one hot vector for each bin
        THINK: Does it make sense to create all one hot vectors in the beginning or is this to memory intense?
        :return:
        '''
        bins_one_hot = []
        for bin in true_bins:
            true_expression = torch.zeros(self.n_input_bins)
            true_expression[bin] = 1
            bins_one_hot.append(true_expression)
        return torch.vstack(bins_one_hot)
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
            nn.Linear(d_model, n_bins),
            nn.Softmax()
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




