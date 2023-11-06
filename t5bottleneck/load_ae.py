from torch import nn
import torch
# from sentence_transformers.models import Pooling
# from python_transformers import CONFIG_MAPPING

from python_transformers.modeling_t5 import T5LayerFF
# from torch.nn import MultiheadAttention
from t5bottleneck.load_attention import AttentionStack, new_config
from sentence_transformers.models import Pooling, Dense, Normalize


class FullSeqAE(nn.Module):
    def __init__(self, encoder, decoder, model_args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = model_args

    def _model_forward(self, encoding, attention_mask):
        latent_z = self.encoder(encoding, attention_mask)
        return self.decoder(latent_z), latent_z

    def forward(self, input_encoding: torch.Tensor, attention_mask: torch.Tensor, just_get_latent=False, just_get_encoding=False):
        recon_encoding, latent = self._model_forward(input_encoding, attention_mask)
        if just_get_latent:
            return latent
        if just_get_encoding:
            return recon_encoding

        if self.args.latent_vec in ['pooling_as_mem', 'attention_as_mem', 'shrinking_as_mem', 'sentenceT5_as_mem']:
            recon_loss = 0
        else:
            recon_loss = torch.nn.MSELoss(reduction="mean")(input_encoding, recon_encoding)

        # reg_loss = self._regularliser_loss(input_encoding, latent)
        reg_loss = 0
        return recon_loss, reg_loss, recon_encoding


class LatentEncoderLargeTanh_1kLatent(nn.Module):
    def __init__(self, dim_m, set_input_size, latent_size, model_args):
        super().__init__()
        assert(dim_m > 100)

        self.latent_vec = model_args.latent_vec
        self.set_input_size = set_input_size

        if self.latent_vec in ['pooling', 'pooling_as_mem', 'pooling_as_input', 'pooling_as_output']:
            self.pool = Pooling(dim_m)
            self.dense = nn.Linear(dim_m, latent_size)

        elif self.latent_vec in ['shrinking', 'shrinking_as_mem', 'shrinking_as_input', 'shrinking_as_output']:
            self.shrink_tokens = nn.Linear(dim_m, 100)
            self.shrink_sequence = nn.Linear(100 * set_input_size, latent_size) # 100*70, 1000

        elif self.latent_vec in ['attention', 'attention_as_mem', 'attention_as_input', 'attention_as_output']:
            self.pool = Pooling(dim_m)
            self.dense = nn.Linear(dim_m, latent_size)
            self.attn_layer = AttentionStack(new_config)

        elif self.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
            self.pool = Pooling(dim_m)
            self.dense = Dense(dim_m, dim_m, activation_function=nn.Identity())
            self.normalize = Normalize()
            self.dense1 = nn.Linear(dim_m, latent_size)

        else:
            exit('LatentEncoderLargeTanh_1kLatent')

        self.tanh = nn.Tanh()

    def forward(self, encoding, attention_mask) -> torch.Tensor:
        batch_size = encoding.size(0)

        if self.latent_vec in ['pooling', 'pooling_as_mem', 'pooling_as_input', 'pooling_as_output']:
            # 1. mean pooling the encoding and feed it into latent space.
            final_encoding_token = self.pool({'token_embeddings': encoding, 'attention_mask': attention_mask})
            final_encoding_token = self.dense(final_encoding_token['sentence_embedding'])
        elif self.latent_vec in ['shrinking', 'shrinking_as_mem', 'shrinking_as_input', 'shrinking_as_output']:
            # 2. shrinking each tokens encoding: convert 768 -> 100 and then join them together.
            encoding = self.shrink_tokens(encoding) # [28, 70, 100]
            final_encoding_token = self.shrink_sequence(encoding.view(batch_size, -1)) # [28, 1000]
        elif self.latent_vec in ['attention', 'attention_as_mem', 'attention_as_input', 'attention_as_output']:
            # sentence = self.pool({'token_embeddings': encoding, 'attention_mask': attention_mask})['sentence_embedding'].unsqueeze(1)
            # sentence = sentence.repeat(1, self.set_input_size, 1)
            # final_encoding_token, attn_output_weights = self.self_attn(encoding, encoding, encoding) # query, key, value
            # final_encoding_token = self.dense(final_encoding_token[:, 0])
            final_encoding_token = self.attn_layer(encoding)[0][:, 0] # choose the first token as sentence embedding
            final_encoding_token = self.dense(final_encoding_token)

        elif self.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
            final_encoding_token = self.pool({'token_embeddings': encoding, 'attention_mask': attention_mask})
            final_encoding_token = self.dense(final_encoding_token)
            final_encoding_token = self.normalize(final_encoding_token)
            final_encoding_token = self.dense1(final_encoding_token['sentence_embedding'])

        else:
            final_encoding_token = 0
            exit('LatentEncoderLargeTanh_1kLatent')

        # 3. choose the first token into latent space.
        # self.dense(encoding[:, 0])

        return self.tanh(final_encoding_token)


class LatentDecoderLargeT5NormFF(nn.Module):
    def __init__(self, dim_m, set_input_size, latent_size, model_args, config):
        super().__init__()

        if model_args.latent_vec_dec == 'linear':
            self.pool_linear = nn.Linear(latent_size, set_input_size*dim_m)

        elif model_args.latent_vec_dec == 'shrinking':
            self.decode_latent = nn.Linear(latent_size, 10 * set_input_size) # 1000 by 10*70
            self.grow_sequence = nn.Linear(10 * set_input_size, 100 * set_input_size) # 10*70 by 100*70
            self.grow_tokens = nn.Linear(100, dim_m)
            self.relu = nn.ReLU()

        else:
            exit('LatentDecoderLargeT5NormFF')

        old_drop = config.dropout_rate
        config.dropout_rate = 0
        self.norm = T5LayerFF(config)
        config.dropout_rate = old_drop

        self.model_args = model_args
        self.dim_m = dim_m
        self.set_input_size = set_input_size

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)

        if self.model_args.latent_vec_dec == 'shrinking':
            # grow each tokens encoding
            latent = self.decode_latent(latent)
            latent = self.grow_sequence(latent)
            out = self.norm(self.grow_tokens(latent.view(batch_size, -1, 100)))

        elif self.model_args.latent_vec_dec == 'linear':
            out = self.norm(self.pool_linear(latent).view(batch_size, self.set_input_size, self.dim_m))

        else:
            print('LatentDecoderLargeT5NormFF')
            out = 0

        return out


def _get_ae_encoder_decoder(t5_model_config, model_args, training_args):
    args = (t5_model_config.d_model, model_args.set_seq_size, model_args.ae_latent_size, model_args)
    return LatentEncoderLargeTanh_1kLatent(*args), LatentDecoderLargeT5NormFF(*(args + (t5_model_config,)))


def _get_ae(t5_model_config, model_args, training_args):
    encoder, decoder = _get_ae_encoder_decoder(t5_model_config, model_args, training_args)
    return FullSeqAE(encoder, decoder, model_args)


if __name__ == '__main__':
    pass