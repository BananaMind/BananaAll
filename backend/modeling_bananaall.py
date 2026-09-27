"""BananaMind 2 style decoder with optional LFT or ternary fake quantization.

The model uses pre-RMSNorm, RoPE, grouped-query attention, SwiGLU and tied
embeddings. BananaMind 2 mode also applies QK norm. LFT changes only routing.
Ternary mode uses floating-point master weights and fake-quantized projections.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_bananaall import BananaAllConfig
except ImportError:
    from configuration_bananaall import BananaAllConfig


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class TernaryLinear(nn.Linear):
    """W1.58A8 fake quantization with straight-through gradients.

    The master weights remain floating point for optimization and checkpoints.
    This layer does not pack ternary weights or use a low-bit inference kernel.
    """

    def forward(self, x):
        weights = self.weight.float()
        weight_scale = weights.detach().abs().mean().clamp_min(1e-6)
        quantized_weights = (weights / weight_scale).round().clamp(-1, 1) * weight_scale
        fake_weights = self.weight + (quantized_weights.to(self.weight.dtype) - self.weight).detach()

        activations = x.float()
        activation_scale = activations.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 127
        quantized_activations = (activations / activation_scale).round().clamp(-127, 127) * activation_scale
        fake_activations = x + (quantized_activations.to(x.dtype) - x).detach()
        return F.linear(fake_activations, fake_weights, self.bias)


def apply_rope(x, theta, position_ids):
    dim = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = position_ids.float().unsqueeze(-1) * inv
    cos = angles.cos().unsqueeze(1).to(x.dtype)
    sin = angles.sin().unsqueeze(1).to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, d, kv = config.num_attention_heads, config.head_dim, config.num_key_value_heads
        self.h, self.d, self.kv, self.theta = h, d, kv, config.rope_theta
        linear = TernaryLinear if config.ternary else nn.Linear
        self.q_proj = linear(config.hidden_size, h * d, bias=False)
        self.k_proj = linear(config.hidden_size, kv * d, bias=False)
        self.v_proj = linear(config.hidden_size, kv * d, bias=False)
        self.o_proj = linear(h * d, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(d, config.rms_norm_eps) if config.architecture_style == "bananamind2" else nn.Identity()
        self.k_norm = RMSNorm(d, config.rms_norm_eps) if config.architecture_style == "bananamind2" else nn.Identity()

    def forward(self, x, attention_mask=None):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.h, self.d).transpose(1, 2))
        k = self.k_norm(self.k_proj(x).view(b, t, self.kv, self.d).transpose(1, 2))
        v = self.v_proj(x).view(b, t, self.kv, self.d).transpose(1, 2)
        positions = torch.arange(t, device=x.device).unsqueeze(0)
        q, k = apply_rope(q, self.theta, positions), apply_rope(k, self.theta, positions)
        k = k.repeat_interleave(self.h // self.kv, dim=1)
        v = v.repeat_interleave(self.h // self.kv, dim=1)
        if attention_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            causal = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
            mask = causal[None, None] & attention_mask[:, None, None, :].bool()
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, self.h * self.d))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        linear = TernaryLinear if config.ternary else nn.Linear
        self.gate_proj = linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask)
        z = self.norm2(x)
        return x + self.down_proj(F.silu(self.gate_proj(z)) * self.up_proj(z))


class BananaAllForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = BananaAllConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    # forward() returns a mean loss per microbatch and ignores **kwargs.
    # Tell Trainer to divide it by the gradient-accumulation count.
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = (TernaryLinear if config.ternary else nn.Linear)(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = self.embed_tokens(input_ids)
        if self.config.lft and len(self.layers) > 1:
            x = self.layers[0](x, attention_mask)
            for i in range(1, len(self.layers)):
                x = self.layers[i](x, attention_mask)
                if i < len(self.layers) - 1:
                    x = self.layers[i - 1](x, attention_mask)
                    x = self.layers[i](x, attention_mask)
        else:
            for layer in self.layers:
                x = layer(x, attention_mask)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1, :].contiguous().float()
            shifted_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def register():
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        AutoConfig.register("bananaall", BananaAllConfig)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(BananaAllConfig, BananaAllForCausalLM)
    except ValueError:
        pass
