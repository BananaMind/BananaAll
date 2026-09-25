export type ArchitectureSpec = {
  vocab_size: number; hidden_size: number; num_hidden_layers: number;
  num_attention_heads: number; num_key_value_heads: number; head_dim: number;
  intermediate_size: number; max_position_embeddings: number; rope_theta: number;
};

export const architectureAnchors: Record<number, ArchitectureSpec> = {
  3: { vocab_size: 2048, hidden_size: 128, num_hidden_layers: 9, num_attention_heads: 4, num_key_value_heads: 2, head_dim: 32, intermediate_size: 512, max_position_embeddings: 4096, rope_theta: 100000 },
  10: { vocab_size: 8192, hidden_size: 256, num_hidden_layers: 10, num_attention_heads: 4, num_key_value_heads: 2, head_dim: 64, intermediate_size: 768, max_position_embeddings: 4096, rope_theta: 100000 },
  25: { vocab_size: 8192, hidden_size: 384, num_hidden_layers: 14, num_attention_heads: 6, num_key_value_heads: 2, head_dim: 64, intermediate_size: 1024, max_position_embeddings: 4096, rope_theta: 100000 },
  50: { vocab_size: 12288, hidden_size: 512, num_hidden_layers: 12, num_attention_heads: 8, num_key_value_heads: 2, head_dim: 64, intermediate_size: 1920, max_position_embeddings: 3072, rope_theta: 100000 },
  140: { vocab_size: 32768, hidden_size: 640, num_hidden_layers: 24, num_attention_heads: 8, num_key_value_heads: 4, head_dim: 80, intermediate_size: 1920, max_position_embeddings: 3072, rope_theta: 100000 },
  200: { vocab_size: 32768, hidden_size: 768, num_hidden_layers: 28, num_attention_heads: 12, num_key_value_heads: 4, head_dim: 64, intermediate_size: 2016, max_position_embeddings: 3072, rope_theta: 100000 },
};

export function estimateParameters(spec: ArchitectureSpec, qkNorm = true): number {
  const h = spec.hidden_size, n = spec.num_hidden_layers, d = spec.head_dim;
  const attention = 2 * h * spec.num_attention_heads * d + 2 * h * spec.num_key_value_heads * d;
  return spec.vocab_size * h + n * (attention + 3 * h * spec.intermediate_size + 2 * h + (qkNorm ? 2 * d : 0)) + h;
}

export function architectureForSize(parametersM: number): ArchitectureSpec {
  const target = Math.max(3, Math.min(200, parametersM));
  const anchors = Object.keys(architectureAnchors).map(Number).sort((a, b) => a - b);
  const exact = anchors.find(size => Math.abs(size - target) < 1e-9);
  if (exact !== undefined) return { ...architectureAnchors[exact] };
  const lower = Math.max(...anchors.filter(size => size < target));
  const upper = Math.min(...anchors.filter(size => size > target));
  const left = architectureAnchors[lower], right = architectureAnchors[upper];
  const fraction = (target - lower) / (upper - lower);
  const between = (field: keyof ArchitectureSpec) => left[field] + fraction * (right[field] - left[field]);
  const desiredHidden = between('hidden_size');
  const preferredHeads = between('num_attention_heads');
  const choices: { score: number; hidden: number; heads: number; kv: number; headDim: number }[] = [];
  for (const anchor of [left, right]) {
    const heads = anchor.num_attention_heads, kv = anchor.num_key_value_heads;
    const baseDim = Math.round(desiredHidden / heads / 8);
    for (const step of [-1, 0, 1]) {
      const headDim = Math.max(16, (baseDim + step) * 8);
      const hidden = heads * headDim;
      const score = Math.abs(hidden - desiredHidden) + Math.abs(heads - preferredHeads) * 1.5;
      choices.push({ score, hidden, heads, kv, headDim });
    }
  }
  choices.sort((a, b) => a.score - b.score || a.hidden - b.hidden || a.heads - b.heads || a.kv - b.kv || a.headDim - b.headDim);
  const selected = choices[0];
  const h = selected.hidden, heads = selected.heads, kv = selected.kv, d = selected.headDim;
  const layers = Math.max(2, Math.round(between('num_hidden_layers')));
  const vocab = Math.max(2048, Math.round(between('vocab_size') / 256) * 256);
  const context = Math.max(512, Math.round(between('max_position_embeddings') / 128) * 128);
  const fixed = vocab * h + layers * (2 * h * heads * d + 2 * h * kv * d + 2 * h + 2 * d) + h;
  const intermediate = Math.max(2 * h, Math.round(((target * 1_000_000 - fixed) / (3 * h * layers)) / 32) * 32);
  return { vocab_size: vocab, hidden_size: h, num_hidden_layers: layers,
    num_attention_heads: heads, num_key_value_heads: kv, head_dim: d,
    intermediate_size: intermediate, max_position_embeddings: context, rope_theta: 100000 };
}
