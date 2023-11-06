from torch import nn
import torch
# from sentence_transformers.models import Pooling
from python_transformers.modeling_t5 import T5LayerNorm, T5LayerFF
from torch.nn import MultiheadAttention


class FullSeqVAE(nn.Module):
    '''
        An VAE to add to encoder-decoder modules.
        Encodes all token encodings into a single vector & spits them back out.
        Switching to an autoencoder to prevent posterior collapse.
    '''
    def __init__(self, encoder, decoder, model_args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(model_args.ae_latent_size, 2*model_args.ae_latent_size, bias=False)
        self.args = model_args

    def _model_forward(self, encoding, attention_mask):
        latent = self.encoder(encoding, attention_mask)
        mu, logvar = self.linear(latent).chunk(2, -1)
        latent_z = self.reparameterize(mu, logvar)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return self.decoder(latent_z), latent_z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

    def forward(self, input_encoding: torch.Tensor, attention_mask: torch.Tensor, just_get_latent=False, just_get_encoding=False):
        recon_encoding, latent, loss_KL = self._model_forward(input_encoding, attention_mask)

        if just_get_latent:
            return latent

        if just_get_encoding:
            return recon_encoding

        recon_loss = torch.nn.MSELoss(reduction="mean")(input_encoding, recon_encoding)

        # reg_loss = self._regularliser_loss(input_encoding, latent)
        reg_loss = 0
        return recon_loss, reg_loss, recon_encoding, loss_KL.mean()


class LatentEncoderLargeTanh_1kLatent(nn.Module):
    def __init__(self, dim_m, set_input_size, latent_size, model_args):
        super().__init__()
        assert(dim_m > 100)
        self.dense = nn.Linear(dim_m, latent_size)
        self.latent_vec = model_args.latent_vec

        if self.latent_vec == 'pooling':
            self.pool = Pooling(latent_size)
        elif self.latent_vec == 'shrinking':
            self.shrink_tokens = nn.Linear(dim_m, 100)
            self.shrink_sequence = nn.Linear(100 * set_input_size, latent_size) # 100*70, 1000
        else:
            self.self_attn = MultiheadAttention(dim_m, dim_m, 10)

        self.tanh = nn.Tanh()

    def forward(self, encoding, attention_mask) -> torch.Tensor:
        batch_size = encoding.size(0)
        if self.latent_vec == 'pooling':
            # 1. mean pooling the encoding and feed it into latent space.
            final_encoding_token = self.pool({'token_embeddings': encoding, 'attention_mask': attention_mask})
            final_encoding_token = self.dense(final_encoding_token['sentence_embedding'])
        else:
            # 2. shrinking each tokens encoding: convert 768 -> 100 and then join them together.
            encoding = self.shrink_tokens(encoding) # [28, 70, 100]
            final_encoding_token = self.shrink_sequence(encoding.view(batch_size, -1)) # [28, 1000]

        # 3. choose the first token into latent space.
        # self.dense(encoding[:, 0])

        return self.tanh(final_encoding_token)


class LatentDecoderLargeT5NormFF(nn.Module):
    def __init__(self, dim_m, set_input_size, latent_size, model_args, config):
        super().__init__()
        self.decode_latent = nn.Linear(latent_size, 10 * set_input_size) # latent_size
        self.grow_sequence = nn.Linear(10 * set_input_size, 100 * set_input_size)
        self.grow_tokens = nn.Linear(100, dim_m)

        old_drop = config.dropout_rate
        config.dropout_rate = 0
        self.norm = T5LayerFF(config)
        config.dropout_rate = old_drop

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)
        # grow each tokens encoding
        latent = self.decode_latent(latent)
        latent = self.grow_sequence(latent)
        return self.norm(self.grow_tokens(latent.view(batch_size, -1, 100)))


def _get_vae_encoder_decoder(t5_model_config, model_args, training_args):
    args = (t5_model_config.d_model, model_args.set_seq_size, model_args.ae_latent_size, model_args)
    return LatentEncoderLargeTanh_1kLatent(*args), LatentDecoderLargeT5NormFF(*(args + (t5_model_config,)))


def _get_vae(t5_model_config, model_args, training_args):
    encoder, decoder = _get_vae_encoder_decoder(t5_model_config, model_args, training_args)
    return FullSeqVAE(encoder, decoder, model_args)


if __name__ == '__main__':
    pass