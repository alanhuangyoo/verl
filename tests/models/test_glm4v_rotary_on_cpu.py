# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Regression test: the GLM-4V attention patch has to use the transformers 5 rotary API.

`glm4v_attn_forward` replaces `Glm4vTextAttention.forward` whenever remove-padding or
Ulysses SP is on, so it runs on the ordinary GLM-4V training path. It used to import
`apply_multimodal_rotary_pos_emb` and hand it an `mrope_section`, which was the
transformers 4 shape. transformers 5 folds `mrope_section` into cos/sin inside
`Glm4vTextRotaryEmbedding` and exposes only `apply_rotary_pos_emb`, so the import raised
`ImportError` at the first attention layer on every version this repo supports
(`transformers>=5.5.3,!=5.6.0,<5.11`).

The flash-attention call is CUDA-only, so it is patched out here; what this pins is the
rotary step, which is what changed.
"""

import torch
from transformers.models.glm4v.configuration_glm4v import Glm4vTextConfig
from transformers.models.glm4v.modeling_glm4v import (
    Glm4vTextAttention,
    Glm4vTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from verl.models.transformers import glm4v as verl_glm4v


def _tiny_attention():
    # head_dim has to be 2 * sum(mrope_section) for the rotary to split its freqs.
    config = Glm4vTextConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=256,
        vocab_size=64,
    )
    return config, Glm4vTextAttention(config, layer_idx=0)


def test_glm4v_attn_forward_applies_the_upstream_rotary(monkeypatch):
    torch.manual_seed(0)
    config, attn = _tiny_attention()
    rotary = Glm4vTextRotaryEmbedding(config)

    batch, seq_len = 1, 8
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    # mRoPE position ids carry a leading section dim of 3.
    position_ids = torch.arange(seq_len).view(1, 1, seq_len).expand(3, batch, seq_len).contiguous()
    cos, sin = rotary(hidden_states, position_ids)

    seen = {}

    def fake_flash_attention_forward(query_states, key_states, value_states, *args, **kwargs):
        seen["query"] = query_states
        seen["key"] = key_states
        return torch.zeros(batch, seq_len, attn.num_heads, attn.head_dim)

    monkeypatch.setattr(verl_glm4v, "_custom_flash_attention_forward", fake_flash_attention_forward)
    # Only bound when flash-attention or NPU is importable, which no CPU runner has.
    monkeypatch.setattr(verl_glm4v, "_flash_use_top_left_mask", False, raising=False)

    output, _ = verl_glm4v.glm4v_attn_forward(
        attn,
        hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
    )

    assert output.shape == (batch, seq_len, config.hidden_size)

    # The same rotary transformers' own Glm4vTextAttention.forward applies, recomputed here.
    query = attn.q_proj(hidden_states).view(batch, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    key = attn.k_proj(hidden_states).view(batch, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    expected_query, expected_key = apply_rotary_pos_emb(query, key, cos, sin)

    torch.testing.assert_close(seen["query"], expected_query.transpose(1, 2))
    # The patch repeats KV up to the query head count before flash attention.
    expected_key = expected_key.repeat_interleave(attn.num_key_value_groups, dim=1)
    torch.testing.assert_close(seen["key"], expected_key.transpose(1, 2))


def test_transformers_5_folds_mrope_section_into_cos_sin():
    """Why the mrope_section argument had to go, rather than being restored."""
    import transformers.models.glm4v.modeling_glm4v as modeling

    assert not hasattr(modeling, "apply_multimodal_rotary_pos_emb")

    config, attn = _tiny_attention()
    rotary = Glm4vTextRotaryEmbedding(config)
    # The sectioning lives on the rotary now, sized to the head dim it will produce.
    assert sum(rotary.mrope_section) * 2 == attn.head_dim

    batch, seq_len = 1, 8
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).view(1, 1, seq_len).expand(3, batch, seq_len).contiguous()
    cos, _ = rotary(hidden_states, position_ids)

    # cos comes back already collapsed: no leading section dim left for a caller to split.
    assert cos.shape == (batch, seq_len, attn.head_dim)
