# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pdb
import numpy as np
import collections
import json
import logging
import math
import os
import sys
from io import open

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter

from .modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from .configuration_gpt2 import GPT2Config
from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {"gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
                                     "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
                                     "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin"}

def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'w' or l[0] == 'g':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'b':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'wpe' or l[0] == 'wte':
                pointer = getattr(pointer, l[0])
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LMFLayer(nn.Module):
    def __init__(self, rank, hidden_size, latent_size):
        super(LMFLayer, self).__init__()
        self.rank = rank
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.text_factor = nn.ModuleList([nn.Linear(self.hidden_size + 1, self.hidden_size) for _ in range(self.rank)])
        self.latent_factor = nn.ModuleList([nn.Linear(self.latent_size + 1, self.hidden_size) for _ in range(self.rank)])

    def forward(self, hidden_states, latent):
        '''
        Args:
            hidden_states: tensor of shape (batch_size, sequence_len, text_in)
            latent: tensor of shape (batch_size, latent_size)
            torch.Size([2, 42, 768])
            torch.Size([2, 32])
        '''
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device
        text_cat = torch.ones(batch_size, seq_len, 1, dtype=torch.float, device=device)
        latent_cat = torch.ones(batch_size, 1, dtype=torch.float, device=device)
        hidden_states = torch.cat((hidden_states, text_cat), dim=-1)
        latent = torch.cat((latent, latent_cat), dim=-1)
        text_fusion_output = []
        latent_fusion_output = []
        for text_factor, latent_factor in zip(self.text_factor, self.latent_factor):
            text_fusion = text_factor(hidden_states)
            latent_fusion = latent_factor(latent)
            text_fusion_output.append(text_fusion)
            latent_fusion_output.append(latent_fusion)
        text_fusion = torch.stack(text_fusion_output).sum(0)
        latent_fusion = torch.stack(latent_fusion_output).sum(0)
        text_fusion = text_fusion.transpose(1, 0)
        output = text_fusion * latent_fusion
        output = output.transpose(1, 0)
        return output


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, fuse=None):
        super(Attention, self).__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

        self.fuse = fuse
        if fuse in ['fuse_syntax_query', 'fuse_syntax_key', 'fuse_syntax_value']:
            self.lmf = LMFLayer(4, 768, 768)
        elif fuse == 'fuse_syntax_query_key_value':
            self.lmf_q = LMFLayer(4, 768, 768)
            self.lmf_k = LMFLayer(4, 768, 768)
            self.lmf_v = LMFLayer(4, 768, 768)
        elif fuse == 'fuse_syntax_query_key_value_one_fuse_model':
            self.lmf = LMFLayer(4, 768, 768)
        elif fuse in ['fuse_syntax_query_key_value_sep_head', 'fuse_syntax_query_key_value_sep_head_no_mask']:
            self.lmf_q = LMFLayer(4, 384, 384)
            self.lmf_k = LMFLayer(4, 384, 384)
            self.lmf_v = LMFLayer(4, 384, 384)
        elif fuse in ['fuse_syntax_query_key_value_sep_head_one_fuse_model', 'fuse_syntax_query_key_value_sep_head_one_fuse_model_no_mask']:
            self.lmf = LMFLayer(4, 384, 384)
        else:
            exit('wrong injection name in Attention')

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2*self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        """
        W shape: torch.Size([16, 12, 28, 29]) 
        V shape: torch.Size([16, 12, 29, 64])
        
        W*V = torch.Size([16, 12, 28, 64]) 
        """

        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        """
        b:  torch.Size([1, 1, 28, 29]) casual mask!!!!!! auto-regressive LM
        w:  torch.Size([16, 12, 28, 29])
        """
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):

        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        """
        new_x_shape:  torch.Size([16, 28, 12, 64])
        or
        new_x_shape:  torch.Size([16, 1, 12, 64])
        """

        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states

        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, latent_syntax=None):

        """
        input x shape: torch.Size([16, 28, 768])
        """
        x = self.c_attn(x)
        """
        conv1D layer: think it as a linear layer.
        x shape:  torch.Size([16, 28, 2304]) 2304/3 = 768
        """

        query, key, value = x.split(self.split_size, dim=2)

        """
        query:  torch.Size([16, 28, 768])
        key:  torch.Size([16, 28, 768])
        value:  torch.Size([16, 28, 768])
        """
        # performing fusion between syntax and query.
        if self.fuse == 'fuse_syntax_query':
            query = self.lmf(query, latent_syntax.squeeze(1))
        elif self.fuse == 'fuse_syntax_key':
            key = self.lmf(key, latent_syntax.squeeze(1))
        elif self.fuse == 'fuse_syntax_value':
            value = self.lmf(value, latent_syntax.squeeze(1))
        elif self.fuse == 'fuse_syntax_query_key_value':
            query = self.lmf_q(query, latent_syntax.squeeze(1))
            key = self.lmf_k(key, latent_syntax.squeeze(1))
            value = self.lmf_v(value, latent_syntax.squeeze(1))
        elif self.fuse == 'fuse_syntax_query_key_value_one_fuse_model':
            query = self.lmf(query, latent_syntax.squeeze(1))
            key = self.lmf(key, latent_syntax.squeeze(1))
            value = self.lmf(value, latent_syntax.squeeze(1))
        elif self.fuse in ['fuse_syntax_query_key_value_sep_head', 'fuse_syntax_query_key_value_sep_head_one_fuse_model',
                           'fuse_syntax_query_key_value_sep_head_no_mask', 'fuse_syntax_query_key_value_sep_head_one_fuse_model_no_mask']:
            #  head sep
            query_mem, query_fuse = torch.split(query, split_size_or_sections=int(query.shape[2]/2), dim=2)
            key_mem, key_fuse = torch.split(key, split_size_or_sections=int(key.shape[2]/2), dim=2)
            value_mem, value_fuse = torch.split(value, split_size_or_sections=int(value.shape[2]/2), dim=2)

            if self.fuse in ['fuse_syntax_query_key_value_sep_head_one_fuse_model', 'fuse_syntax_query_key_value_sep_head_one_fuse_model_no_mask']:
                query_fuse = self.lmf(query_fuse, latent_syntax.squeeze(1))
                key_fuse = self.lmf(key_fuse, latent_syntax.squeeze(1))
                value_fuse = self.lmf(value_fuse, latent_syntax.squeeze(1))
            else:
                query_fuse = self.lmf_q(query_fuse, latent_syntax.squeeze(1))
                key_fuse = self.lmf_k(key_fuse, latent_syntax.squeeze(1))
                value_fuse = self.lmf_v(value_fuse, latent_syntax.squeeze(1))

            query = torch.concat([query_mem, query_fuse], dim=2)
            key = torch.concat([key_mem, key_fuse], dim=2)
            value = torch.concat([value_mem, value_fuse], dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        """
        size inf: (batch=16, seq_length=28, head=12, head_features=64) 
        new query:  torch.Size([16, 12, 28, 64])
        new key:  torch.Size([16, 12, 64, 28])
        new value:  torch.Size([16, 12, 28, 64])
        """

        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]  # transpose back cf below torch.Size([16, 1, 768])
            
            past_key = self.split_heads(past_key, k=True)
            past_value = self.split_heads(past_value)
            # pdb.set_trace()
            """
            past_key:  torch.Size([16, 12, 64, 1])
            past_value:  torch.Size([16, 12, 1, 64])
            """
            if self.fuse in ['fuse_syntax_query_key_value_sep_head', 'fuse_syntax_query_key_value_sep_head_one_fuse_model']:
                # set [16, 6:12, 64, 1]  = 0
                zero_tensor = torch.zeros(past_key.shape[0], int(past_key.shape[1]/2), past_key.shape[2], past_key.shape[3])
                one_tensor = torch.ones(past_key.shape[0], int(past_key.shape[1]/2), past_key.shape[2], past_key.shape[3])
                mem_mask = torch.cat([one_tensor, zero_tensor], 1).to(past_key.device)

                past_key = past_key * mem_mask
                past_value = past_value * mem_mask.permute(0, 1, 3, 2)

            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
            """
            final key:  torch.Size([16, 12, 64, 29])
            final value:  torch.Size([16, 12, 29, 64])
            """

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)

        """
        attn_outputs shape:  torch.Size([16, 12, 28, 64])
        """
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        """
        a shape: torch.Size([16, 28, 768])
        """

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False, fuse=None):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale, fuse=fuse)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, latent_syntax=None):
        output_attn = self.attn(self.ln_1(x),
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask,
                                latent_syntax=latent_syntax)
        a = output_attn[0]  # output_attn: a, present, (attentions)
        """
        a shape: torch.Size([16, 28, 768])
        """

        x = x + a # this step is for last self-attention operation.
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = GPT2Config
    pretrained_model_archive_map = GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super(GPT2PreTrainedModel, self).__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


GPT2_START_DOCSTRING = r"""    OpenAI GPT-2 model was proposed in
    `Language Models are Unsupervised Multitask Learners`_
    by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
    It's a causal (unidirectional) transformer pre-trained using  language modeling on a very large
    corpus of ~40 GB of text data.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Language Models are Unsupervised Multitask Learners`:
        https://openai.com/blog/better-language-models/

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            GPT-2 is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.
            Indices can be obtained using :class:`pytorch_transformers.GPT2Tokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

# @add_start_docstrings("The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
#                       GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING)


class GPT2Model(GPT2PreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """
    def __init__(self, config, latent_size, vocab_role_size, fuse):
        super(GPT2Model, self).__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True, fuse=fuse) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.role_type_embeddings = nn.Embedding(vocab_role_size, 768) if vocab_role_size is not None else None

        try:
            self.latent_size = latent_size
        except:
            self.latent_size = 32 # default size is 32

        # self.linear = nn.Linear(self.latent_size, config.hidden_size * config.n_layer, bias=False) # different latent vector for each layer
        if fuse in ['uni_direction', 'bi_direction']:
            self.linear_content = nn.Linear(int(self.latent_size/2), config.hidden_size * int(config.n_layer/2), bias=False)
            self.linear_struct = nn.Linear(int(self.latent_size/2), config.hidden_size * int(config.n_layer/2), bias=False)
        elif fuse == 'tensor_fuse':
            # self.linear_content = nn.Linear(int(self.latent_size/2), int(config.hidden_size/2) * config.n_layer, bias=False)
            # self.linear_struct = nn.Linear(int(self.latent_size/2), int(config.hidden_size/2) * config.n_layer, bias=False)
            self.linear = nn.Linear(self.latent_size, config.hidden_size * config.n_layer, bias=False) # different latent vector for each layer
        elif fuse in ['fuse_syntax_query', 'fuse_syntax_key', 'fuse_syntax_value']:
            self.linear_content = nn.Linear(int(self.latent_size/2), config.hidden_size * config.n_layer, bias=False)
            self.linear_struct = nn.Linear(int(self.latent_size/2), config.hidden_size * config.n_layer, bias=False)
        elif fuse in ['fuse_syntax_query_key_value', 'fuse_syntax_query_key_value_one_fuse_model']:
            self.linear_content = nn.Linear(int(self.latent_size/2), config.hidden_size * config.n_layer, bias=False)
            self.linear_struct = nn.Linear(int(self.latent_size/2), config.hidden_size * config.n_layer, bias=False)
        elif fuse in ['fuse_syntax_query_key_value_sep_head', 'fuse_syntax_query_key_value_sep_head_no_mask',
                      'fuse_syntax_query_key_value_sep_head_one_fuse_model', 'fuse_syntax_query_key_value_sep_head_one_fuse_model_no_mask']:
            self.linear_content = nn.Linear(int(self.latent_size/2), config.hidden_size * config.n_layer, bias=False)
            self.linear_struct = nn.Linear(int(self.latent_size/2), int(config.hidden_size/2) * config.n_layer, bias=False)
        else:
            self.linear_content = nn.Linear(int(self.latent_size/2), int(config.hidden_size/2) * config.n_layer, bias=False)
            self.linear_struct = nn.Linear(int(self.latent_size/2), int(config.hidden_size/2) * config.n_layer, bias=False)

        self.linear_emb = nn.Linear(self.latent_size, config.hidden_size, bias=False) # share the same latent vector as the embeddings

        self.config = config
        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        self.wte = self._get_resized_embeddings(self.wte, new_num_tokens)
        return self.wte

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, latent_as_gpt_emb=False, latent_as_gpt_memory=True, is_role=False, role_ids=None, fuse=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_content, past_struct = torch.split(past, split_size_or_sections=int(past.shape[1]/2), dim=1)

            if latent_as_gpt_emb:
                past_emb = self.linear_emb(past) # used as embeddings to add on other three embeddings
                exit('only support latent_as_gpt_memory')
                """
                torch.Size([16, 768]) 
                """

            if latent_as_gpt_memory:
                """
                input past (latent_z) shape:  torch.Size([16, 32])
                """
                # past = self.linear(past)
                if fuse == 'uni_direction':
                    past_struct = self.linear_struct(past_struct + past_content)
                    past_content = self.linear_content(past_content)
                    past = torch.cat((past_content, past_struct), dim=1) # 10, 9216
                elif fuse == 'bi_direction':
                    past_struct1 = self.linear_struct(past_struct + past_content)
                    past_content1 = self.linear_content(past_content + past_struct)
                    past = torch.cat((past_content1, past_struct1), dim=1)
                elif fuse == 'layer_disentangle':
                    past_struct1 = self.linear_struct(past_struct).view(past_struct.shape[0], self.config.n_layer, 1, -1)
                    past_content1 = self.linear_content(past_content).view(past_content.shape[0], self.config.n_layer, 1, -1)
                    past = torch.cat([past_struct1, past_content1], 2).view(past_struct.shape[0], self.config.n_layer, -1)
                elif fuse == 'tensor_fuse':
                    past = self.lmf(past_struct, past_content)
                    past = self.linear(past)
                elif fuse in ['fuse_syntax_query', 'fuse_syntax_key', 'fuse_syntax_value']:
                    past_struct = self.linear_struct(past_struct)
                    past = self.linear_content(past_content)
                elif fuse in ['fuse_syntax_query_key_value', 'fuse_syntax_query_key_value_one_fuse_model']:
                    past_struct = self.linear_struct(past_struct)
                    past = self.linear_content(past_content)
                elif fuse in ['fuse_syntax_query_key_value_sep_head', 'fuse_syntax_query_key_value_sep_head_no_mask',
                              'fuse_syntax_query_key_value_sep_head_one_fuse_model', 'fuse_syntax_query_key_value_sep_head_one_fuse_model_no_mask']:
                    # shape: [16, 12, 1, 384]
                    past_struct = self.linear_struct(past_struct)
                    past = self.linear_content(past_content)
                else:
                    past = self.linear(past)
                    exit('only support uni_direction or bi_direction')
                """
                output past shape: ([16, 9216]) GPT-2 has 12 layers.
                """
                share_latent = False
                if share_latent: 
                    # the same latent vector shared by all layers
                    past = [past.unsqueeze(-2), past.unsqueeze(-2)] # query, key
                    past = [past] * len(self.h)
                    past_length = past[0][0].size(-2)
                else:
                    if fuse == 'layer_disentangle':
                        past_split = [past[:, i:i+1, :] for i in range(past.shape[1])]
                    else:
                        if fuse in ['fuse_syntax_query_key_value_sep_head', 'fuse_syntax_query_key_value_sep_head_no_mask',
                        'fuse_syntax_query_key_value_sep_head_one_fuse_model', 'fuse_syntax_query_key_value_sep_head_one_fuse_model_no_mask']:
                            # different latent vectors for each layer each size is torch.Size([16, 1, 768])
                            past_split = torch.split(past.unsqueeze(1), self.config.hidden_size, dim=2)
                            past_struct = torch.split(past_struct.unsqueeze(1), int(self.config.hidden_size/2), dim=2)
                        else:
                            past_split = torch.split(past.unsqueeze(1), self.config.hidden_size, dim=2)
                            past_struct = torch.split(past_struct.unsqueeze(1), self.config.hidden_size, dim=2)

                    # past_split_content = torch.split(past_content, int(self.config.hidden_size/2), dim=2)
                    # past_split_struct = torch.split(past_struct, int(self.config.hidden_size/2), dim=2)
                    """
                    past_split shape: length 12, each item shape: torch.Size([16, 1, 768])
                    """
                    past = list(zip(past_split, past_split))
                    # past = list(zip(past_merge, past_merge))
                    """
                    past length: 12, each item (key, value) => (torch.Size([16, 1, 768]), torch.Size([16, 1, 768]))
                    """
                    # past = past.view(batch_size,len(self.h),-1)
                    # past = [[past[:,i,:].unsqueeze(-2), past[:,i,:].unsqueeze(-2) ] for i in range(len(self.h))]
                    past_length = 1 # past[0][0].size(-2)
            else:
                past_length = 0
                past = [None] * len(self.h)


        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)


        # Attention mask.
        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))


        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        """
        exp2 or exp1 or exp3
        """
        if is_role:
            hidden_states = self.role_type_embeddings(input_ids) + position_embeds
        else:
            if role_ids is None:
                # exp 1 or conditional_optimus
                hidden_states = inputs_embeds + position_embeds + token_type_embeds
            else:
                # exp3
                hidden_states = inputs_embeds + position_embeds #+ self.role_type_embeddings(role_ids)

        """
        hidden_states shape: torch.Size([16, 28, 768])
        """
        """
        past_emb shape:  torch.Size([16, 768])
        past_emb.unsqueeze(1): torch.Size([16, 1, 768])
        """

        if latent_as_gpt_emb:
            # pdb.set_trace()
            hidden_states = hidden_states + past_emb.unsqueeze(1)

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()

        # 12 layers. past = list(zip(past_split, past_split)) past_split = [16, 1, 768]
        for i, (block, layer_past, layer_syntax) in enumerate(zip(self.h, past, past_struct)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            """
            layer_past: [(k, v), (k, v), ..., (k, v)] => [(torch.Size([16, 1, 768]), torch.Size([16, 1, 768])), ...]
            hidden_states (original input of GPT2): torch.size([16, 28, 768]) where 28 is the length of sequence.
            """
            outputs = block(hidden_states,
                            layer_past=layer_past,
                            attention_mask=attention_mask,
                            head_mask=head_mask[i],
                            latent_syntax=layer_syntax)

            """
            hidden states: [16, 28, 768] 
            present: [2, 16, 12, 29, 64]
            """
            hidden_states, present = outputs[:2]

            presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)

        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, presents, (all hidden_states), (attentions)


@add_start_docstrings("""The GPT2 Model transformer with a language modeling head on top
(linear layer with weights tied to the input embeddings). """, GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.tie_weights()


    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, label_ignore=None):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=label_ignore, reduce=False) # 50258 is the padding id, otherwise -1 is used for masked LM.
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            loss = torch.sum(loss.view(-1, shift_labels.shape[-1]), -1)
            outputs = (loss,) + outputs


        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)



@add_start_docstrings("""The GPT2 Model transformer with a language modeling head on top
(linear layer with weights tied to the input embeddings). """, GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING)
class GPT2ForLatentConnector(GPT2PreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """
    def __init__(self, config, latent_size, latent_as_gpt_emb=False, latent_as_gpt_memory=True, vocab_role_size=1, fuse=None):
        
        super(GPT2ForLatentConnector, self).__init__(config, vocab_role_size)
        
        self.transformer = GPT2Model(config, latent_size, vocab_role_size, fuse)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.tie_weights()

        self.latent_as_gpt_emb = latent_as_gpt_emb
        self.latent_as_gpt_memory = latent_as_gpt_memory
        self.fuse = fuse

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, label_ignore=None, is_role=False, role_label_ignore=None, role_ids=None):

        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask, 
                                               latent_as_gpt_emb=self.latent_as_gpt_emb,
                                               latent_as_gpt_memory=self.latent_as_gpt_memory,
                                               is_role=is_role,
                                               role_ids=role_ids,
                                               fuse=self.fuse)

        hidden_states = transformer_outputs[0]
        """
        [16, 28, 768]
        """
        lm_logits = self.lm_head(hidden_states)
        # [5, 6, 50260] batch, len, vocab

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous() # [5, 5, 50260]
            shift_labels = labels[..., 1:].contiguous() # [5, 5]
            # Flatten the tokens
            label_ignore = label_ignore if not is_role else role_label_ignore
            loss_fct = CrossEntropyLoss(ignore_index=label_ignore, reduction='none') # 50258 is the padding id, otherwise -1 is used for masked LM.
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = torch.mean(loss.view(-1, shift_labels.shape[-1]), -1) # [5] batch size
            outputs = (loss,) + outputs


        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

@add_start_docstrings("""The GPT2 Model transformer with a language modeling and a multiple-choice classification
head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.
The language modeling head has its weights tied to the input embeddings,
the classification head takes as input the input of a specified classification token index in the input sequence).
""", GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    r"""
        **mc_token_ids**: (`optional`, default to index of the last token of the input) ``torch.LongTensor`` of shape ``(batch_size, num_choices)``:
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        **mc_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **lm_loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **mc_loss**: (`optional`, returned when ``multiple_choice_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Multiple choice classification loss.
        **lm_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **mc_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)``
            Prediction scores of the multiplechoice classification head (scores for each choice before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from pytorch_transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
        
        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary
        
        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

    """
    def __init__(self, config):
        super(GPT2DoubleHeadsModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                mc_token_ids=None, lm_labels=None, mc_labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask)

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)),
                            mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)
