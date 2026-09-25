"""Continuous BananaAll model sizing, with the published shapes as anchors."""
import math

ANCHORS = {
    3: dict(vocab_size=2048, hidden_size=128, num_hidden_layers=9, num_attention_heads=4, num_key_value_heads=2, head_dim=32, intermediate_size=512, max_position_embeddings=4096, rope_theta=100000.0),
    10: dict(vocab_size=8192, hidden_size=256, num_hidden_layers=10, num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=768, max_position_embeddings=4096, rope_theta=100000.0),
    25: dict(vocab_size=8192, hidden_size=384, num_hidden_layers=14, num_attention_heads=6, num_key_value_heads=2, head_dim=64, intermediate_size=1024, max_position_embeddings=4096, rope_theta=100000.0),
    50: dict(vocab_size=12288, hidden_size=512, num_hidden_layers=12, num_attention_heads=8, num_key_value_heads=2, head_dim=64, intermediate_size=1920, max_position_embeddings=3072, rope_theta=100000.0),
    140: dict(vocab_size=32768, hidden_size=640, num_hidden_layers=24, num_attention_heads=8, num_key_value_heads=4, head_dim=80, intermediate_size=1920, max_position_embeddings=3072, rope_theta=100000.0),
    200: dict(vocab_size=32768, hidden_size=768, num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=4, head_dim=64, intermediate_size=2016, max_position_embeddings=3072, rope_theta=100000.0),
}


def estimated_parameters(spec, qk_norm=True):
    hidden = spec["hidden_size"]
    heads = spec["num_attention_heads"]
    kv_heads = spec["num_key_value_heads"]
    head_dim = spec["head_dim"]
    intermediate = spec["intermediate_size"]
    layers = spec["num_hidden_layers"]
    embeddings = spec["vocab_size"] * hidden
    attention = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
    feed_forward = 3 * hidden * intermediate
    norms = 2 * hidden + (2 * head_dim if qk_norm else 0)
    return embeddings + layers * (attention + feed_forward + norms) + hidden


def architecture_for_size(parameters_m):
    target = float(parameters_m)
    if not 3 <= target <= 200:
        raise ValueError("Model size must be between 3M and 200M parameters")
    for anchor, spec in ANCHORS.items():
        if abs(target - anchor) < 1e-9:
            return dict(spec)
    lower = max(size for size in ANCHORS if size < target)
    upper = min(size for size in ANCHORS if size > target)
    left, right = ANCHORS[lower], ANCHORS[upper]
    fraction = (target - lower) / (upper - lower)
    between = lambda field: left[field] + fraction * (right[field] - left[field])
    desired_hidden = between("hidden_size")
    preferred_heads = between("num_attention_heads")
    choices = []
    for anchor in (left, right):
        heads = anchor["num_attention_heads"]
        kv_heads = anchor["num_key_value_heads"]
        base_dim = math.floor(desired_hidden / heads / 8 + 0.5)
        for step in (-1, 0, 1):
            head_dim = max(16, (base_dim + step) * 8)
            hidden = heads * head_dim
            score = abs(hidden - desired_hidden) + abs(heads - preferred_heads) * 1.5
            choices.append((score, hidden, heads, kv_heads, head_dim))
    _, hidden, heads, kv_heads, head_dim = min(choices)
    layers = max(2, math.floor(between("num_hidden_layers") + 0.5))
    vocab = max(2048, math.floor(between("vocab_size") / 256 + 0.5) * 256)
    context = max(512, math.floor(between("max_position_embeddings") / 128 + 0.5) * 128)
    fixed = vocab * hidden + layers * (2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim + 2 * hidden + 2 * head_dim) + hidden
    intermediate = max(2 * hidden, math.floor(((target * 1_000_000 - fixed) / (3 * hidden * layers)) / 32 + 0.5) * 32)
    return dict(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers,
                num_attention_heads=heads, num_key_value_heads=kv_heads,
                head_dim=head_dim, intermediate_size=intermediate,
                max_position_embeddings=context, rope_theta=100000.0)
