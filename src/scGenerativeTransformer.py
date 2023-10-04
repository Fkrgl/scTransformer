import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict
# encoder layers are based on paper "Attention is all you need"
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchmetrics import Accuracy

class scGenerativeTransformer(nn.Module):
    '''
    The idea is to make the transformer more sample generating oriented. I will try to achive this by
    letting the transformer as before generate expression profiles for the whole cell. But this time the
    whole generated profile will be used in the loss function
    '''
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
        #self.dropout = dropout
        self.n_input_bins = n_input_bins
        self.n_token = n_token
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_gene = self.n_token - 1

        # define the gene encoder
        self.encoder = GeneEncoder(self.n_token, self.d_model)
        self.value_encoder = ValueEncoder(self.n_input_bins+1, self.d_model) # , padding_idx=self.n_input_bins, +1 beacuse we added bin -1 as mask_value
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
        self.generator = ExprDecoder(self.d_model, self.n_input_bins)
        self.index = np.arange(self.n_token)

        self.loss = nn.CrossEntropyLoss()
        self.acc_function = Accuracy(task='multiclass', num_classes=self.n_input_bins)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
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
        output = self.transformer_encoder(total_embedding, src_key_padding_mask=key_padding_mask)

        return output.to(self.device)  # (batch, seq_len, embsize)

    def forward(self,
                src: Tensor,
                values: Tensor,
                masked_values: Tensor,
                attn_mask: Tensor,
                key_padding_mask: Tensor,
                mask_type: str,
                get_accuracy: bool = False,
                ):
        """

        Args:
            src: token ids, shape (batch_size, seq_len)
            values: token values, shape (batch_size, seq_len)
            masked_values: binned expression values where the expression of masked genes is substitutes by mask_value
            key_padding_mask: mask for src, shape (batch_size, seq_len)

        Returns:
            Tensor of expression prediction
        """
        # if test samples are processed, randomize masked positions
        # print(f'values: {values[0]}')
        target = values.clone().to(self.device)
        transformer_output = self._encode(src, masked_values, attn_mask, key_padding_mask, mask_type, get_accuracy)
        # if get_accuracy:
        #     transformer_output = self.randomize_maked_position_encodeings(transformer_output, key_padding_mask)
        # each gene embedding gets its own expression value prediction
        mlm_output = self.generator(transformer_output)  # (batch, seq_len, n_bin)
        # get loss
        # reshape input and target for loss
        mlm_output, target = self.reshape_for_loss(mlm_output, target)
        loss = self.loss(mlm_output, target)
        output = loss
        # accuracy is only computed using masked values
        if get_accuracy:
            acc_value = self.acc_function(mlm_output, target)
            output = (loss, acc_value)

        return output
    def reshape_for_loss(self, input_, target):
        '''
        reshape input and target tensor got the cross entorpy loss function.
        Args:
            input_: (batch, genes, bins)
            target: (batch, genes)

        Returns:
            input_: ((batch x genes), bins)
            target: ((batch x genes), 1)

        '''
        N = target.shape[0]  # batch size
        target = torch.reshape(target, ((N*self.n_gene),))
        input_ = torch.reshape(input_, ((N*self.n_gene), self.n_input_bins))
        return input_, target
    def generate(self,
                 src: Tensor,
                 values: Tensor,
                 attn_mask: Tensor,
                 key_padding_mask: Tensor,
                 mask_type: str,
                 bins,
                 get_accuracy: bool
                 ):
        n_gen = self.n_token-1
        # embedd genes of target cell and feed in transformer encoder
        encoder_output = self._encode(src, values, attn_mask, key_padding_mask, mask_type, get_accuracy)
        # decode transformer encoded gene vectors
        decoder_output = self.generator(encoder_output)
        # print(f'decoder_output_shape: {decoder_output.shape}')
        # get softmax
        s = nn.Softmax(dim=1)
        softmax_output = s(decoder_output)
        # print(softmax_output.sum(dim=0))
        # print(softmax_output.sum(dim=0).shape)
        # print(f'softmax:\n{softmax_output}')
        # print(softmax_output[0].sum())
        # print(softmax_output[0].shape)
        # sample from softmax
        #np.save('/home/claassen/cxb257/scTransformer/data/heart_endothelial_decoderOutput.npy', decoder_output.detach().numpy())
        sample_profile = np.zeros(shape=n_gen)
        # print(softmax_output[0])
        # print(softmax_output[0].shape)
        # print(softmax_output[0].sum())
        for i in range(n_gen-1):
            # maybe the softmax is too strict
            prob = softmax_output[i].detach().numpy()
            #print(f'prob: {prob}')
            assert len(prob) == len(bins)
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