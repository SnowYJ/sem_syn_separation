import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from sentence_transformers.models import Pooling


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.role_type_embeddings = nn.Embedding( ,config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class DSVAE(nn.Module):
    def __init__(self, encoder, decoder, decoder_type, sr_dim, en_nlayers, de_nlayers, pool_mode, semantic_pad, args):
        super().__init__()
        d_model = 768
        dropout = 0.5
        nsem = sr_dim
        latent_size = args.latent_size

        self.args = args
        self.semantic_mask = semantic_pad
        self.decoder_type = decoder_type

        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.src_embedding = encoder.embeddings # nn.Embedding(nsem, d_model)
        self.pooling = Pooling(768, pooling_mode=pool_mode)

        # bert embedding layer
        self.tgt_embedding = nn.Embedding(nsem, d_model) # encoder.embeddings
        self.encoder = nn.Transformer(
            d_model=d_model,
            nhead=12,
            num_encoder_layers=en_nlayers,
            num_decoder_layers=de_nlayers,
            dropout=dropout,
        )

        self.linear = nn.Linear(d_model, 2 * latent_size)

        if self.decoder_type == 'gpt2':
            self.decoder = decoder
        else:
            self.shrink_layer = nn.Linear(latent_size, 768)
            self.decoder = nn.Transformer(
                d_model=d_model,
                nhead=12,
                num_encoder_layers=en_nlayers,
                num_decoder_layers=de_nlayers,
                dropout=dropout,
            )
        # self.init_weights()

    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     self.encoder.weight.data.uniform_(-initrange, initrange)
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, r: Tensor, tgt: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        semantic_mask = (src != self.semantic_mask).float().T
        reconstrution_mask = (tgt != 50257).float() # 50257 is the padding token for GPT2
        sent_length = torch.sum(reconstrution_mask, dim=1)

        src = self.src_embedding(src)
        r = self.tgt_embedding(r)

        transformer_out = self.encoder(src, r, src_key_padding_mask=semantic_mask, tgt_key_padding_mask=semantic_mask)
        pool_out = self.pooling({'token_embeddings': transformer_out.transpose(0, 1), 'attention_mask': semantic_mask})

        if self.args.fb_mode == 0:
            latent_z, loss_kl = self.connect(pool_out['sentence_embedding'])
            latent_z = latent_z.squeeze(1)
            if self.decoder_type == 'gpt2':
                outputs = self.decoder(input_ids=tgt, past=latent_z, labels=tgt.long(), label_ignore=50257, is_role=False, role_label_ignore=False, role_ids=False)
                loss_rec = outputs[0]
            else:

                tgt = self.src_embedding(tgt)
                self.decoder(latent_z, tgt, src_key_padding_mask=semantic_mask, tgt_key_padding_mask=semantic_mask)
        else:
            mu, logvar = self.linear(pool_out['sentence_embedding']).chunk(2, -1)
            latent_z = self.reparameterize(mu, logvar, nsamples=1)
            loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
            kl_mask = (loss_kl > self.args.dim_target_kl).float()
            loss_kl = (kl_mask * loss_kl).sum(dim=1)
            latent_z = latent_z.squeeze(1)
            outputs = self.decoder(input_ids=tgt, past=latent_z, labels=tgt.long(), label_ignore=50257, is_role=False, role_label_ignore=False, role_ids=False)
            loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

        if self.args.length_weighted_loss:
            loss = loss_rec / sent_length + self.args.beta * loss_kl
        else:
            loss = loss_rec + self.args.beta * loss_kl

        return loss_rec.mean(), loss_kl.mean(), loss.mean()

    def connect(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.linear(bert_fea).chunk(2, -1)
        # pdb.set_trace()
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)

        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)