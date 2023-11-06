from python_transformers import PreTrainedModel
from torch.nn import CrossEntropyLoss
from python_transformers import T5ForConditionalGeneration, T5Tokenizer
from t5bottleneck.load_vae import _get_vae
from t5bottleneck.load_ae import _get_ae
from python_transformers import AutoConfig, CONFIG_MAPPING
import torch
from torch import nn
# from sentence_transformers import SentenceTransformer
# from transformers import BartTokenizer
# from transformers import BartModel
from transformers import BartForConditionalGeneration
from transformers import BartTokenizer, BartModel


def _get_config(model_args):
    if model_args.config_name:
        return AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_path:
        return AutoConfig.from_pretrained(model_args.model_path, cache_dir=model_args.cache_dir)
    else:
        print("You are instantiating a new config instance from scratch.")
        # print(CONFIG_MAPPING[model_args.model_type]())
        return CONFIG_MAPPING[model_args.model_type]()


def _get_t5_vae_requirements(model_args, training_args):
    config = _get_config(model_args)
    name = model_args.t5_model_name # "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(name)
    t5_model = T5ForConditionalGeneration.from_pretrained(name, latent_size=None)
    # t5_model, tokenizer = _get_t5_model(model_args.t5_model_name, model_args.tokenizer_name, model_args.cache_dir)
    vae = _get_vae(t5_model.config, model_args, training_args)
    return config, t5_model, tokenizer, vae


def _get_t5_ae_requirements(model_args, training_args):
    config = _get_config(model_args)
    name = model_args.t5_model_name # "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(name)

    t5_model = T5ForConditionalGeneration.from_pretrained(name, latent_size=None)
    # t5_model, tokenizer = _get_t5_model(model_args.t5_model_name, model_args.tokenizer_name, model_args.cache_dir)
    ae = _get_ae(t5_model.config, model_args, training_args)
    return config, t5_model, tokenizer, ae


def _get_t5_origin_requirements(model_args, training_args):
    config = _get_config(model_args)
    name = model_args.t5_model_name # "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(name)
    t5_model = T5ForConditionalGeneration.from_pretrained(name)
    new_words = ['{', '}', '^', '\\']
    tokenizer.add_tokens(new_words)
    t5_model.resize_token_embeddings(len(tokenizer))

    return config, t5_model, tokenizer


def new_t5_vae(model_args, training_args):
    config, t5_model, tokenizer, vae = _get_t5_vae_requirements(model_args, training_args)
    return t5_AE(config, t5_model, vae, model_args.set_seq_size, tokenizer, model_args, training_args)


def load_t5_vae(model_args, training_args):
    config, t5_model, tokenizer, vae = _get_t5_vae_requirements(model_args, training_args)
    t5_ae = t5_AE(config, t5_model, vae, model_args.set_seq_size, tokenizer, model_args, training_args)
    checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
    t5_ae.load_state_dict(checkpoint)
    print('loading pretrained t5_vae successful.')
    return t5_ae


def load_t5_ae(model_args, training_args):
    config, t5_model, tokenizer, vae = _get_t5_ae_requirements(model_args, training_args)
    t5_ae = t5_AE(config, t5_model, vae, model_args.set_seq_size, tokenizer, model_args, training_args)

    # if model_args.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
    #     from load_sentenceT5 import load_sentenceT5_weight
    #     from sentence_transformers import SentenceTransformer
    #     sent_t5 = SentenceTransformer('sentence-transformers/sentence-t5-base')
    #     new_state_dict = load_sentenceT5_weight(t5_ae, sent_t5)
    #     t5_ae.load_state_dict(new_state_dict)
    #
    #     s1 = (t5_ae.state_dict()['t5_model.lm_head.weight'] == t5_ae.state_dict()['t5_model.shared.weight']).all()
    #     s2 = (t5_ae.state_dict()['t5_model.encoder.embed_tokens.weight'] == t5_ae.state_dict()['t5_model.shared.weight']).all()
    #     s3 = (t5_ae.state_dict()['t5_model.decoder.embed_tokens.weight'] == t5_ae.state_dict()['t5_model.shared.weight']).all()
    #
    #     if s1 and s2 and s3:
    #         print('loading sentenceT5 to T5encoder successful.')
    #     else:
    #         exit('ERROR in new_t5_ae')

    checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
    t5_ae.load_state_dict(checkpoint, strict=False)
    print('loading pretrained t5_ae successful.')
    return t5_ae


def new_t5_ae(model_args, training_args):
    config, t5_model, tokenizer, vae = _get_t5_ae_requirements(model_args, training_args)
    t5_ae = t5_AE(config, t5_model, vae, model_args.set_seq_size, tokenizer, model_args, training_args)

    if model_args.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
        from load_sentenceT5 import load_sentenceT5_weight
        from sentence_transformers import SentenceTransformer
        sent_t5 = SentenceTransformer('sentence-transformers/sentence-t5-base')
        new_state_dict = load_sentenceT5_weight(t5_ae, sent_t5)
        t5_ae.load_state_dict(new_state_dict)

        s1 = (t5_ae.state_dict()['t5_model.lm_head.weight'] == t5_ae.state_dict()['t5_model.shared.weight']).all()
        s2 = (t5_ae.state_dict()['t5_model.encoder.embed_tokens.weight'] == t5_ae.state_dict()['t5_model.shared.weight']).all()
        s3 = (t5_ae.state_dict()['t5_model.decoder.embed_tokens.weight'] == t5_ae.state_dict()['t5_model.shared.weight']).all()

        if s1 and s2 and s3:
            print('loading sentenceT5 to T5encoder successful.')
        else:
            exit('ERROR in new_t5_ae')

    return t5_ae


def load_t5_origin(model_args, training_args):
    config, t5_model, tokenizer = _get_t5_origin_requirements(model_args, training_args)
    t5 = t5_AE(config, t5_model, None, model_args.set_seq_size, tokenizer, model_args, training_args)
    checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
    t5.load_state_dict(checkpoint)
    print('loading pretrained t5 successful.')
    return t5


def new_t5_origin(model_args, training_args):
    config, t5_model, tokenizer = _get_t5_origin_requirements(model_args, training_args)
    return t5_AE(config, t5_model, None, model_args.set_seq_size, tokenizer, model_args, training_args)


# ----------------------------------------------------------------------------------------------------
def _get_bart_origin_requirements(model_args, training_args):
    config = _get_config(model_args)
    name = model_args.t5_model_name # "t5-base"

    # BART model
    bart_model = BartForConditionalGeneration.from_pretrained(name)
    tokenizer = BartTokenizer.from_pretrained(name)

    tokenizer.add_special_tokens({'sep_token': '</s>'})
    bart_model.resize_token_embeddings(len(tokenizer))

    return config, bart_model, tokenizer


def load_bart_origin(model_args, training_args):
    config, t5_model, tokenizer = _get_bart_origin_requirements(model_args, training_args)
    t5 = t5_AE(config, t5_model, None, model_args.set_seq_size, tokenizer, model_args, training_args)
    checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
    t5.load_state_dict(checkpoint)
    print('loading pretrained bart successful.')
    return t5


def new_bart_origin(model_args, training_args):
    config, t5_model, tokenizer = _get_bart_origin_requirements(model_args, training_args)
    return t5_AE(config, t5_model, None, model_args.set_seq_size, tokenizer, model_args, training_args)
# ----------------------------------------------------------------------------------------------------


# T5 + VAE
class t5_AE(PreTrainedModel):
    # base_model_prefix = 't5_vae'
    def __init__(self, config, t5_model, vae, set_seq_size, tokenizer, model_args, training_args):
        super().__init__(config=config)
        self.t5_model = t5_model

        if model_args.latent_type == 'T5_vae':
            self.vae = vae
        elif model_args.latent_type == 'T5_ae':
            self.ae = vae
        else:
            pass

        if model_args.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
            size1, size2 = t5_model.state_dict()['shared.weight'].shape
            self.enc_embed_weight = nn.Embedding(size1, size2)

        self.config = config
        self.set_seq_size = set_seq_size
        self.tokenizer = tokenizer
        self.latent_type = model_args.latent_type
        self.latent_vec = model_args.latent_vec
        self.model_name = model_args.model_type

        self.batch_size = training_args.per_device_train_batch_size
        self.set_seq_size = model_args.set_seq_size

        # if model_args.latent_vec == 'pooling_as_mem':
        #     self.linear = nn.Linear(model_args.ae_latent_size, config.d_model * config.num_layers, bias=False) # different latent vector for each layer

    def _decoder_logits(self, decoder_input_ids, encoding, encoder_attention_mask):
        # decoder_attention_mask = (decoder_input_ids>0).long
        if self.latent_vec in ['pooling_as_mem', 'attention_as_mem', 'shrinking_as_mem', 'sentenceT5_as_mem']:
            batch_size = encoding.shape[0]
            sequence_size = encoding.shape[1]
            past_key_value_states = encoding.view(batch_size, 12, sequence_size, -1)
            sequence_output = self.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoding,
                                                    latent_mem=past_key_value_states)[0]
        elif self.latent_vec in ['pooling_as_input', 'attention_as_input', 'shrinking_as_input', 'sentenceT5_as_input']:
            sequence_output = self.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=None, latent_mem=None, latent_input=encoding)[0]

        elif self.latent_vec in ['attention_as_output', 'pooling_as_output', 'shrinking_as_output', 'sentenceT5_as_output']:
            sequence_output = self.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=None, latent_mem=None, latent_input=None)[0]
            sequence_output += encoding
        else:
            if self.model_name == 't5':
                sequence_output = self.t5_model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoding, output_attentions=True)
                sequence_output = sequence_output[0]
            elif self.model_name == 'bart':
                sequence_output = self.t5_model.model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoding, output_attentions=True)
                sequence_output = sequence_output[0]
            else:
                sequence_output = None


        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        if self.model_name == 't5':
            sequence_output = sequence_output * (self.t5_model.model_dim ** -0.5)
            logits = self.t5_model.lm_head(sequence_output)
        elif self.model_name == 'bart':
            # sequence_output = sequence_output * (768 ** -0.5)
            logits = self.t5_model.lm_head(sequence_output)
        else:
            logits = None

        return logits

    def decoder_loss(self, outputs, labels, encoding, ignore_index=0, encoder_attention_mask=None):
        # shift right to make it started with 1
        # decoder_input_ids = self.t5_model._shift_right(labels)
        """
        the decoder input looks like: [1, w1, w2, ..., wn, 0, 0, ...]
        the decoder label looks like: [w1, w2, ..., wn, 1, 0, 0, ...]
        """
        decoder_input_ids = outputs.contiguous()
        labels = labels.contiguous()

        logits = self._decoder_logits(decoder_input_ids, encoding, encoder_attention_mask)
        loss_fct = CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.long().view(-1))
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        return loss

    def get_latent(self, input_ids):
        attention_mask = input_ids.ne(self.config.pad_token_id).long()
        if self.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
            input_embed = self.enc_embed_weight(input_ids)
            encoding = self.t5_model.encoder(input_ids=None, inputs_embeds=input_embed, attention_mask=attention_mask)[0]
        else:
            encoding = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        if self.latent_type == 'T5_vae':
            return self.vae(encoding, attention_mask=attention_mask, just_get_latent=True)
        elif self.latent_type == 'T5_ae':
            return self.ae(encoding, attention_mask=attention_mask, just_get_latent=True)
        else:
            return encoding

    def get_hidden(self, input_ids):
        attention_mask = input_ids.ne(self.config.pad_token_id).long()

        if self.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
            input_embed = self.enc_embed_weight(input_ids)
            encoding = self.t5_model.encoder(input_ids=None, inputs_embeds=input_embed, attention_mask=attention_mask)[0]
        else:
            if self.model_name == 't5':
                encoding = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            elif self.model_name == 'bart':
                encoding = self.t5_model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            else:
                encoding = None

        if self.latent_type == 'T5_vae':
            return self.vae(encoding, attention_mask=attention_mask, just_get_encoding=True)
        elif self.latent_type == 'T5_ae':
            return self.ae(encoding, attention_mask=attention_mask, just_get_encoding=True)
        else:
            return encoding

    def get_attention(self, input_ids):
        attention_mask = input_ids.ne(self.config.pad_token_id).long()
        attention = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        return attention

    def get_cross_attention(self, input_ids, output_ids):
        pass

    def forward(self, input_ids, output_ids, label_ids):
        recon_loss, reg_loss, loss_KL = 0, 0, 0
        attention_mask = input_ids.ne(self.config.pad_token_id).long()

        if self.latent_vec in ['sentenceT5', 'sentenceT5_as_mem', 'sentenceT5_as_input', 'sentenceT5_as_output']:
            input_embed = self.enc_embed_weight(input_ids)
            encoding = self.t5_model.encoder(input_ids=None, inputs_embeds=input_embed, attention_mask=attention_mask)[0]
        else:
            if self.model_name == 't5':
                encoding = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            elif self.model_name == 'bart':
                encoding = self.t5_model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            else:
                encoding = None
                exit('Error: wrong model_name t5 or bart.')

        # vae or ae.
        if self.latent_type == 'T5_vae':
            recon_loss, reg_loss, encoding, loss_KL = self.vae(encoding, attention_mask)
        elif self.latent_type == 'T5_ae':
            recon_loss, reg_loss, encoding = self.ae(encoding, attention_mask)
        else:
            pass

        decoder_ce = self.decoder_loss(output_ids, label_ids, encoding, ignore_index=self.config.pad_token_id, encoder_attention_mask=attention_mask)

        return decoder_ce, recon_loss, reg_loss, loss_KL


if __name__ == '__main__':
    pass