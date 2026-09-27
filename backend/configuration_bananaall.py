from transformers import PretrainedConfig


class BananaAllConfig(PretrainedConfig):
    model_type = "bananaall"

    def __init__(self, vocab_size=8192, hidden_size=384, num_hidden_layers=14,
                 num_attention_heads=6, num_key_value_heads=2, head_dim=64,
                 intermediate_size=1024, max_position_embeddings=4096,
                 rope_theta=100000.0, rms_norm_eps=1e-6, architecture_style="bananamind2",
                 lft=False, ternary=False, **kwargs):
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.architecture_style = architecture_style
        self.lft = lft
        self.ternary = ternary
        self.use_cache = False
