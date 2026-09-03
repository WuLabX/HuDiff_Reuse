import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sequence_models.layers import PositionFeedForward, DoubleEmbedding
from sequence_models.convolutional import ByteNetBlock, MaskedConv1d
from .cross_attention import TransformerNet, SelfAttNet, precompute_freqs_cis
# from .modules import TransformerLayer, ESM1LayerNorm

# Framework load.
import math

# abnativ_scoring removed: AMP-Diff uses PepNet+HemoPI2 scorers instead


# Abnativ res list.
alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
own_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

class MLP(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.ln1 = nn.Linear(n_embd, 2 * n_embd)
        self.gelu = nn.GELU()
        self.ln2 = nn.Linear(2 * n_embd, n_embd)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        x = self.dropout(x)
        return x

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model=8, length=500):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        """
        Used for encoding timestep in diffusion models

        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if self.d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) * -(np.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        device = x.device
        pe = pe.to(device)
        return pe[x] # .to(x.device)


class PositionalEncoding(nn.Module):

    """
    2D Positional encoding for transformer
    :param d_model: dimension of the model
    :param max_len: max number of positions
    """

    def __init__(self, d_model, max_len=152):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        y = self.pe[:x.size(1)]
        x = x + y.reshape(y.shape[1], y.shape[0], y.shape[2])
        return x


class NanoByteNetTime(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers

         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L, d)
    """

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu', down_embed=False,
                 timesteps=None, aa_h_length=152, aa_l_length=139):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :padding_idx: location of padding token in ordered alphabet
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()
        :param rank: rank of compressed weight matrices
        :param n_frozen_embs: number of frozen embeddings
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu' or 'gelu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        :param timesteps: None or int providing max timesteps in DM model
        """
        super().__init__()
        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_embedding, timesteps) # Timestep encoding
        if n_tokens is not None:
            if n_frozen_embs is None:
                self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
            else:
                self.embedder = DoubleEmbedding(n_tokens - n_frozen_embs, n_frozen_embs,
                                                d_embedding, padding_idx=padding_idx)
        else:
            self.embedder = nn.Identity()
        if down_embed:
            self.up_embedder = PositionFeedForward(d_embedding, d_model)
        else:
            self.up_embedder = nn.Identity()
            assert d_model == d_embedding
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout
        self.aa_h_length = aa_h_length
        self.aa_l_length = aa_l_length

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length)
        :param y: (batch)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x, timesteps=self.timesteps)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x, timesteps=None):
        e = self.embedder(x)
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e


class NanoSideEmbedder(nn.Module):
    def __init__(self, n_side, s_embedding, d_side, aa_h_length=152):
        super().__init__()
        self.side_embeddinng = nn.Embedding(n_side, s_embedding)
        self.side_mlp = nn.Sequential(
            nn.Linear(s_embedding, d_side),
            nn.LayerNorm(d_side),
            nn.ReLU(),
            nn.Linear(d_side, d_side),
        )
        self.aa_h_length = aa_h_length

    def forward(self, side, mask=None):
        emb_side = self.side_embeddinng(side.view(-1, 1))
        emb_side = self.side_mlp(emb_side)
        emb_side = emb_side.repeat(1, self.aa_h_length, 1)
        return emb_side


class NanoRegionEmbedder(nn.Module):

    def __init__(self, r_pos, r_embedding, r_model, rank=None):
        super().__init__()
        self.region_embedding = nn.Embedding(r_pos, r_embedding)
        self.region_layer1 = nn.Sequential(
            nn.LayerNorm(r_embedding),
            nn.ReLU(),
            PositionFeedForward(r_embedding, r_model, rank=rank),
            nn.LayerNorm(r_model),
            nn.ReLU()
        )

    def forward(self, pos_seq, mask=None):
        """
        :param pos_seq:
        :param mask:
        :return:
        """
        x = self.region_embedding(pos_seq)
        x = self.region_layer1(x)
        return x


class NanoPosEmbedder(nn.Module):
    """
    This position embedding method is PE encoding.
    """
    def __init__(self, p_emb, max_len):
        super().__init__()
        self.pos_embedding = PositionalEncoding(p_emb, max_len)
        self.pos_lin = MLP(n_embd=p_emb)

    def forward(self, region_emb):
        x = self.pos_embedding(region_emb)
        pos_emb = self.pos_lin(x)
        x = x + pos_emb
        return x


class ByteNetLMTime(nn.Module):

    def __init__(self):
        super().__init__()
        pass


class NanoConv(nn.Module):

    def __init__(self, d_model, n_layers, kernel_size, r, rank=None,
                causal=False, dropout=0.0, slim=True, activation='gelu', timesteps=None,
                 aa_h_length=152, aa_l_length=139):
        super().__init__()

        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.aa_h_length = aa_h_length
        self.aa_l_length = aa_l_length
        self.dropout = dropout

    def _conv(self, s):
        for layer in self.layers:
            s = layer(s)
            if self.dropout > 0.0:
                s = F.dropout(s)
        return s

    def forward(self, s, batch=None, mask=None):
        s = self._conv(s)
        return s




class TransformerEncoder(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, att_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embed = nn.Embedding(n_tokens, d_embedding)
        self.up_embedder = PositionFeedForward(d_embedding, d_model)
        self.att_net = TransformerNet(d_model, att_model, nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)

    def forward(self, x):
        emb_x = self.embed(x)
        up_emb_x = self.up_embedder(emb_x)
        x = self.att_net(up_emb_x)
        return x


class NanoAntiTFNet(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_encoder_layers, aa_kernel_size, r,
                 n_region, r_embedding, r_model,
                 n_pos_model, max_len,
                 sum_d_model, dual_layers,
                 att_model, dim_feedforward, nhead, cs_layers,
                 rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu',
                 down_embed=False, timesteps=None):
        super().__init__()

        self.aa_encoder = NanoByteNetTime(n_tokens, d_embedding, d_model, n_encoder_layers, aa_kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs,
                                timesteps=timesteps)
        self.region_encoder = NanoRegionEmbedder(n_region, r_embedding, r_model)
        self.pos_encoder = NanoPosEmbedder(n_pos_model, max_len)
        self.nano_conv_block = NanoConv(sum_d_model, dual_layers, aa_kernel_size, r, dropout=dropout)
        self.self_at = SelfAttNet(sum_d_model, att_model, dim_feedforward, nhead, rolength=max_len, num_cross_layers=cs_layers)
        self.last_norm = nn.LayerNorm(sum_d_model)
        self.decoder = nn.Linear(sum_d_model, n_tokens)

    def _encoder(self, aa_seq, region_type, chn_type):
        emb = self.aa_encoder(aa_seq)
        region_emb = self.region_encoder(region_type)
        pos_emb = self.pos_encoder(region_emb)
        emb = emb + pos_emb
        feature = torch.cat((emb, pos_emb), dim=-1)
        return feature  

    def _att(self, h):
        h = self.self_at(h)
        return h

    def forward(self, H_seq, H_region_type, H_chn_type):
        """

        :param H_L_seq: (Batch, length);
        :param H_L_pos_type: (Batch, length); distinguish the different region of Chain.
        :param H_L_chn_type: (Batch); gene
        :param H_L_batch: (Batch); distinguish the type of Chain.
        :param H_L_mask: None
        :return: (Batch, length, feature)
        """
        h_feature = self._encoder(
            aa_seq=H_seq.int(),
            region_type=H_region_type.int(),
            chn_type=H_chn_type
        )
        h = self.nano_conv_block(h_feature)
        h = self._att(h)
        h = self.decoder(self.last_norm(h))
        return h


