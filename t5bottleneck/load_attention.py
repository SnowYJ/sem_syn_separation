from python_transformers.modeling_t5 import *


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


class AttentionStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList()
        self.num_layers = config.num_layers
        for i in range(config.num_layers):
            self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=True))

    def forward(self, hidden_states, attention_mask=None):

        for i in range(self.num_layers):
            self_attention_outputs = self.layer[i](hidden_states=hidden_states, attention_mask=attention_mask)
            hidden_states = self_attention_outputs[0]

        return self_attention_outputs



new_config = {
    "d_ff": 2048,
    "d_kv": 64,
    "d_model": 768,
    "dropout_rate": 0.1,
    "initializer_factor": 1.0,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "n_positions": 512,
    "num_heads": 12,
    "num_layers": 8,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "is_decoder": False
}
new_config = DotDict(new_config)

if __name__ == '__main__':
    pass