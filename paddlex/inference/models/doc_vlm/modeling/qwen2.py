# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from typing import List, Optional, Tuple, Union

import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import sequence_parallel_utils

from .....utils import logging
from .....utils.env import get_device_type
from ...common.vlm import fusion_ops
from ...common.vlm.activations import ACT2FN
from ...common.vlm.transformers import PretrainedConfig, PretrainedModel
from ...common.vlm.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        GatherOp,
        ScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    pass

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None


Linear = nn.Linear
ColumnParallelLinear = mpu.ColumnParallelLinear
RowParallelLinear = mpu.RowParallelLinear
ColumnSequenceParallelLinear = sequence_parallel_utils.ColumnSequenceParallelLinear
RowSequenceParallelLinear = sequence_parallel_utils.RowSequenceParallelLinear


class Qwen2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    """

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        seq_length=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        pad_token_id=0,
        bos_token_id=151643,
        eos_token_id=151643,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        rope_scaling_factor=1.0,
        rope_scaling_type=None,
        dpo_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.use_cache = use_cache
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_scaling_type = rope_scaling_type

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.dpo_config = dpo_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def parallel_matmul(
    x: Tensor, y: Tensor, transpose_y=True, tensor_parallel_output=True
):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False

    if paddle.in_dynamic_mode():
        y_is_distributed = y.is_distributed
    else:
        y_is_distributed = tensor_parallel_degree > 1

    if is_fleet_init and tensor_parallel_degree > 1 and y_is_distributed:
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        input_parallel = paddle.distributed.collective._c_identity(
            x, group=model_parallel_group
        )
        logits = paddle.matmul(input_parallel, y, transpose_y=transpose_y)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group
        )

    else:
        logits = paddle.matmul(x, y, transpose_y=transpose_y)
        return logits


def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    attn_mask_startend_row_indices=None,
    training=True,
    sequence_parallel=False,
    skip_recompute=False,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, _, _ = value_states.shape

    #  [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
    query_states = paddle.transpose(query_states, [0, 2, 1, 3])
    # merge with the next transpose
    key_states = paddle.transpose(key_states, [0, 2, 1, 3])
    value_states = paddle.transpose(value_states, [0, 2, 1, 3])

    # Add pre divided factor to fix nan under float16.
    if paddle.in_dynamic_mode() and query_states.dtype == paddle.float16:
        pre_divided_factor = 32
    else:
        pre_divided_factor = 1

    attn_weights = paddle.matmul(
        query_states / (math.sqrt(head_dim) * pre_divided_factor),
        key_states.transpose([0, 1, 3, 2]),
    )

    if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
        raise ValueError(
            f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.shape}"
        )

    if attention_mask is None:
        attention_mask = get_triangle_upper_mask(attn_weights)

    attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
    if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
        raise ValueError(
            f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
        )

    attn_weights = attn_weights + attention_mask

    if not paddle.in_dynamic_mode():
        attn_weights = F.softmax(
            attn_weights * pre_divided_factor, axis=-1, dtype="float32"
        ).astype(query_states.dtype)
    else:
        with paddle.amp.auto_cast(False):
            attn_weights = F.softmax(
                attn_weights.astype("float32") * pre_divided_factor,
                axis=-1,
                dtype="float32",
            ).astype(query_states.dtype)

    attn_weights = F.dropout(
        attn_weights, p=config.attention_dropout, training=training
    )

    attn_output = paddle.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose([0, 2, 1, 3])

    if sequence_parallel:
        attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
    else:
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
    return (attn_output, attn_weights) if output_attentions else attn_output


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.concat(
            [paddle.ones([target_length, past_key_values_length], dtype="bool"), mask],
            axis=-1,
        )

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand(
        [batch_size, 1, target_length, target_length + past_key_values_length]
    )


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


class Qwen2RMSNorm(nn.Layer):
    def __init__(self, config: Qwen2Config):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, hidden_states):
        if self.config.use_fused_rms_norm:
            return fusion_ops.fusion_rms_norm(
                hidden_states, self.weight, self.variance_epsilon, False
            )

        if paddle.in_dynamic_mode():
            with paddle.amp.auto_cast(False):
                variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
                hidden_states = (
                    paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
                )
        else:
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = (
                paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
            )

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


class Qwen2RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        self.inv_freq = 1.0 / (
            self.base
            ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim)
        )
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids is None:
        # Note: Only for Qwen2MoEForCausalLMPipe model pretraining
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Layer):
    def __init__(self, config: Qwen2Config, is_shared=False, skip_recompute_ops=None):
        super().__init__()
        if skip_recompute_ops is None:
            skip_recompute_ops = {}
        self.skip_recompute_ops = skip_recompute_ops
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fuse_attention_ffn = config.fuse_attention_ffn

        self.tensor_parallel_degree = config.tensor_parallel_degree

        if config.sequence_parallel:
            ColumnParallelLinear = ColumnSequenceParallelLinear
            RowParallelLinear = RowSequenceParallelLinear

        if config.tensor_parallel_degree > 1:
            if self.fuse_attention_ffn:
                self.gate_up_fused_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size * 2,
                    gather_output=False,
                    has_bias=False,
                )
            else:
                self.gate_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    gather_output=False,
                    has_bias=False,
                )
                self.up_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    gather_output=False,
                    has_bias=False,
                )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            if self.fuse_attention_ffn:
                self.gate_up_fused_proj = Linear(
                    self.hidden_size, self.intermediate_size * 2, bias_attr=False
                )
            else:
                self.gate_proj = Linear(
                    self.hidden_size, self.intermediate_size, bias_attr=False
                )  # w1
                self.up_proj = Linear(
                    self.hidden_size, self.intermediate_size, bias_attr=False
                )  # w3
            self.down_proj = Linear(
                self.intermediate_size, self.hidden_size, bias_attr=False
            )  # w2

        if config.hidden_act == "silu":
            self.act_fn = fusion_ops.swiglu
            self.fuse_swiglu = True
        else:
            self.act_fn = ACT2FN[config.hidden_act]
            self.fuse_swiglu = False

    def forward(self, x):
        if self.fuse_attention_ffn:
            x = self.gate_up_fused_proj(x)
            if self.fuse_swiglu:
                y = None
            else:
                x, y = x.chunk(2, axis=-1)
        else:
            x, y = self.gate_proj(x), self.up_proj(x)

        if self.fuse_swiglu:
            x = self.act_fn(x, y)
        else:
            x = self.act_fn(x) * y

        return self.down_proj(x)


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(hidden_states, n_rep, axis=1). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states.unsqueeze(-2).tile([1, 1, 1, n_rep, 1])
    return hidden_states.reshape([batch, slen, num_key_value_heads * n_rep, head_dim])


class Qwen2Attention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(
        self,
        config: Qwen2Config,
        layerwise_recompute: bool = True,
        skip_recompute_ops=None,
    ):
        super().__init__()
        if skip_recompute_ops is None:
            skip_recompute_ops = {}
        self.config = config
        self.skip_recompute_ops = skip_recompute_ops
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.head_dim = self.hidden_size // config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads
        assert config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.gqa_or_mqa = config.num_attention_heads != config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.seq_length = config.seq_length
        self.sequence_parallel = config.sequence_parallel

        self.fuse_attention_qkv = config.fuse_attention_qkv

        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity
        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree

            assert (
                self.num_key_value_heads % config.tensor_parallel_degree == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_key_value_heads = (
                self.num_key_value_heads // config.tensor_parallel_degree
            )

        self.use_fused_rope = config.use_fused_rope
        if self.use_fused_rope:
            if (
                get_device_type() not in ["gpu", "xpu"]
                or fused_rotary_position_embedding is None
            ):
                logging.warning(
                    "Enable fuse rope in the config, but fuse rope is not available. "
                    "Will disable fuse rope. Try using latest gpu version of Paddle."
                )
                self.use_fused_rope = False

        if config.sequence_parallel:
            ColumnParallelLinear = ColumnSequenceParallelLinear
            RowParallelLinear = RowSequenceParallelLinear

        if config.tensor_parallel_degree > 1:
            if self.fuse_attention_qkv:
                self.qkv_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size
                    + 2 * self.config.num_key_value_heads * self.head_dim,
                    has_bias=True,
                    gather_output=False,
                )
            else:
                self.q_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    has_bias=True,
                    gather_output=False,
                )
                self.k_proj = ColumnParallelLinear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, has_bias=True, gather_output=False)  # fmt:skip
                self.v_proj = ColumnParallelLinear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, has_bias=True, gather_output=False)  # fmt:skip
            self.o_proj = RowParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                input_is_parallel=True,
            )
        else:
            if self.fuse_attention_qkv:
                self.qkv_proj = Linear(
                    self.hidden_size,
                    self.hidden_size
                    + 2 * self.config.num_key_value_heads * self.head_dim,
                )
            else:
                self.q_proj = Linear(self.hidden_size, self.hidden_size, bias_attr=True)
                self.k_proj = Linear(
                    self.hidden_size,
                    self.config.num_key_value_heads * self.head_dim,
                    bias_attr=True,
                )
                self.v_proj = Linear(
                    self.hidden_size,
                    self.config.num_key_value_heads * self.head_dim,
                    bias_attr=True,
                )
            self.o_proj = Linear(self.hidden_size, self.hidden_size, bias_attr=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.attn_func = scaled_dot_product_attention

    def forward(
        self,
        hidden_states,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)

        if self.fuse_attention_qkv:
            mix_layer = self.qkv_proj(hidden_states)
            if self.sequence_parallel:
                target_shape = [
                    -1,
                    self.seq_length,
                    self.num_key_value_heads,
                    (self.num_key_value_groups + 2) * self.head_dim,
                ]
            else:
                target_shape = [
                    0,
                    0,
                    self.num_key_value_heads,
                    (self.num_key_value_groups + 2) * self.head_dim,
                ]
            mix_layer = paddle.reshape_(mix_layer, target_shape)
            query_states, key_states, value_states = paddle.split(
                mix_layer,
                num_or_sections=[
                    self.num_key_value_groups * self.head_dim,
                    self.head_dim,
                    self.head_dim,
                ],
                axis=-1,
            )
            if self.gqa_or_mqa:
                query_states = paddle.reshape_(
                    query_states, [0, 0, self.num_heads, self.head_dim]
                )
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if self.sequence_parallel:
                target_query_shape = [
                    -1,
                    self.seq_length,
                    self.num_heads,
                    self.head_dim,
                ]
                target_key_value_shape = [
                    -1,
                    self.seq_length,
                    self.num_key_value_heads,
                    self.head_dim,
                ]
            else:
                target_query_shape = [0, 0, self.num_heads, self.head_dim]
                target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
            query_states = query_states.reshape(shape=target_query_shape)
            key_states = key_states.reshape(shape=target_key_value_shape)
            value_states = value_states.reshape(shape=target_key_value_shape)

        kv_seq_len = key_states.shape[-3]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]
        if self.use_fused_rope:
            assert past_key_value is None, "fuse rotary not support cache kv for now"
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states, _ = fused_rotary_position_embedding(
                query_states,
                key_states,
                v=None,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
                use_neox_rotary_style=False,
            )
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)
        past_key_value = (key_states, value_states) if use_cache else None

        # TODO(wj-Mcat): use broadcast strategy when n_kv_heads = 1
        # repeat k/v heads if n_kv_heads < n_heads
        paddle_version = float(paddle.__version__[:3])
        if not self.config.use_flash_attention or (
            (paddle_version != 0.0) and (paddle_version <= 2.6)
        ):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        outputs = self.attn_func(
            query_states,
            self.config,
            key_states,
            value_states,
            attention_mask,
            output_attentions,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            training=self.training,
            sequence_parallel=self.sequence_parallel,
        )
        if output_attentions:
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class Qwen2DecoderLayer(nn.Layer):
    def __init__(
        self,
        config: Qwen2Config,
        layerwise_recompute: bool = False,
        skip_recompute_ops=None,
    ):
        super().__init__()
        if skip_recompute_ops is None:
            skip_recompute_ops = {}
        self.config = config
        self.skip_recompute_ops = skip_recompute_ops
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(
            config, layerwise_recompute, skip_recompute_ops=skip_recompute_ops
        )

        self.mlp = Qwen2MLP(config, skip_recompute_ops=skip_recompute_ops)
        self.input_layernorm = Qwen2RMSNorm(config)
        self.post_attention_layernorm = Qwen2RMSNorm(config)

        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """

        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        outputs = self.self_attn(
            hidden_states,
            position_ids,
            past_key_value,
            attention_mask,
            output_attentions,
            use_cache,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        if type(outputs) is tuple:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        if output_attentions:
            self_attn_weights = outputs[1]

        if use_cache:
            present_key_value = outputs[2 if output_attentions else 1]

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class Qwen2PretrainedModel(PretrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "qwen2"
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: Qwen2Config, is_fuse=False):
        # return parameter fuse utils
        from ...common.vlm.conversion_utils import split_or_fuse_func

        fn = split_or_fuse_func(is_fuse=is_fuse)

        # last key is fused key, other keys are to be fused.
        fuse_qkv_keys = [
            (
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.qkv_proj.weight",
            ),
            (
                "layers.0.self_attn.q_proj.bias",
                "layers.0.self_attn.k_proj.bias",
                "layers.0.self_attn.v_proj.bias",
                "layers.0.self_attn.qkv_proj.bias",
            ),
        ]

        fuse_gate_up_keys = (
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.gate_up_fused_proj.weight",
        )
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)
        fuse_attention_ffn = getattr(config, "fuse_attention_ffn", False)

        final_actions = {}
        if is_fuse:
            if fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_qkv_keys:
                        keys = tuple(
                            [
                                key.replace("layers.0.", f"layers.{i}.")
                                for key in fuse_keys
                            ]
                        )
                        final_actions[keys] = partial(
                            fn,
                            is_qkv=True,
                            num_heads=num_heads,
                            num_key_value_heads=num_key_value_heads,
                        )
            if fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    keys = tuple(
                        [
                            key.replace("layers.0.", f"layers.{i}.")
                            for key in fuse_gate_up_keys
                        ]
                    )
                    final_actions[keys] = fn
        else:
            if not fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_qkv_keys:
                        keys = tuple(
                            [
                                key.replace("layers.0.", f"layers.{i}.")
                                for key in fuse_keys
                            ]
                        )
                        final_actions[keys] = partial(
                            fn,
                            split_nums=3,
                            is_qkv=True,
                            num_heads=num_heads,
                            num_key_value_heads=num_key_value_heads,
                        )
            if not fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    keys = tuple(
                        [
                            key.replace("layers.0.", f"layers.{i}.")
                            for key in fuse_gate_up_keys
                        ]
                    )
                    final_actions[keys] = partial(fn, split_nums=2)
        return final_actions


class Qwen2Model(Qwen2PretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.hidden_size = config.hidden_size
        self.sequence_parallel = config.sequence_parallel
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = (
            config.no_recompute_layers if config.no_recompute_layers is not None else []
        )

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        if (
            config.tensor_parallel_degree > 1
            and config.vocab_size % config.tensor_parallel_degree == 0
        ):
            self.embed_tokens = mpu.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList(
            [
                Qwen2DecoderLayer(
                    config=config,
                    layerwise_recompute=layer_idx not in self.no_recompute_layers,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @staticmethod
    def _prepare_decoder_attention_mask(
        attention_mask, input_shape, past_key_values_length, dtype
    ):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(
                    attention_mask, dtype, tgt_length=input_shape[-1]
                )
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape,
                        past_key_values_length=past_key_values_length,
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape,
                past_key_values_length=past_key_values_length,
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        if get_device_type() == "xpu":
            x = paddle.to_tensor(0.0, dtype="float32")
            y = paddle.to_tensor(-1.7005809656952787e38, dtype="float32")
            expanded_attn_mask = paddle.where(expanded_attn_mask, x, y)
        else:
            expanded_attn_mask = paddle.where(
                expanded_attn_mask.cast("bool"), 0.0, paddle.finfo(dtype).min
            ).astype(dtype)
        return expanded_attn_mask

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # fmt:skip
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length
        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids)

        if self.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        # [bs, seq_len]
        attention_mask = (
            paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
            if attention_mask is None
            else attention_mask
        )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
        )  # [bs, 1, seq_len, seq_len]
        if self.config.use_flash_attention:
            attention_mask = None if is_casual_mask(attention_mask) else attention_mask

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand(
                (batch_size, seq_length)
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            has_gradient = not hidden_states.stop_gradient
            if (
                self.enable_recompute
                and idx not in self.no_recompute_layers
                and has_gradient
                and self.recompute_granularity == "full"
            ):
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None
            if type(layer_outputs) is tuple:
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2PretrainingCriterion(nn.Layer):
    """
    Criterion for Mixtral.
    It calculates the final loss.
    """

    def __init__(self, config: Qwen2Config):
        super(Qwen2PretrainingCriterion, self).__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.enable_parallel_cross_entropy = (
            config.tensor_parallel_degree > 1 and config.tensor_parallel_output
        )

        if (
            self.enable_parallel_cross_entropy
        ):  # and False: # and lm_head is distributed
            self.loss_func = mpu.ParallelCrossEntropy(ignore_index=self.ignore_index)
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none", ignore_index=self.ignore_index
            )

    def forward(self, prediction_scores, masked_lm_labels):
        if self.enable_parallel_cross_entropy:
            if prediction_scores.shape[-1] == self.config.vocab_size:
                logging.warning(
                    f"enable_parallel_cross_entropy, the vocab_size should be splitted: {prediction_scores.shape[-1]}, {self.config.vocab_size}"
                )
                self.loss_func = paddle.nn.CrossEntropyLoss(
                    reduction="none", ignore_index=self.ignore_index
                )

        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(
                prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(2)
            )

            # skip ignore_index which loss == 0
            # masked_lm_loss = masked_lm_loss[masked_lm_loss > 0]
            # loss = paddle.mean(masked_lm_loss)
            binary_sequence = paddle.where(
                masked_lm_loss > 0,
                paddle.ones_like(masked_lm_loss),
                paddle.zeros_like(masked_lm_loss),
            )
            count = paddle.sum(binary_sequence)
            if count == 0:
                loss = paddle.sum(masked_lm_loss * binary_sequence)
            else:
                loss = paddle.sum(masked_lm_loss * binary_sequence) / count

        return loss


class Qwen2LMHead(nn.Layer):
    def __init__(self, config: Qwen2Config, embedding_weights=None, transpose_y=False):
        super(Qwen2LMHead, self).__init__()
        self.config = config
        if (
            config.tensor_parallel_degree > 1
            and config.vocab_size % config.tensor_parallel_degree == 0
        ):
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        self.transpose_y = transpose_y
        if transpose_y:
            if embedding_weights is not None:
                self.weight = embedding_weights
            else:
                self.weight = self.create_parameter(
                    shape=[vocab_size, config.hidden_size],
                    dtype=paddle.get_default_dtype(),
                )
        else:
            if vocab_size != config.vocab_size:
                self.weight = self.create_parameter(
                    shape=[config.hidden_size, vocab_size],
                    dtype=paddle.get_default_dtype(),
                )
            else:
                self.weight = self.create_parameter(
                    shape=[config.hidden_size, vocab_size],
                    dtype=paddle.get_default_dtype(),
                )

        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = (
            True if (vocab_size != config.vocab_size) else False
        )
        if self.weight.is_distributed:
            # for tie_word_embeddings
            self.weight.split_axis = 0 if self.transpose_y else 1

    def forward(self, hidden_states, tensor_parallel_output=None):
        if self.config.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
            seq_length = self.config.seq_length
            hidden_states = paddle.reshape_(
                hidden_states, [-1, seq_length, self.config.hidden_size]
            )

        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output

        logits = parallel_matmul(
            hidden_states,
            self.weight,
            transpose_y=self.transpose_y,
            tensor_parallel_output=tensor_parallel_output,
        )
        return logits


class Qwen2ForCausalLM(Qwen2PretrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.qwen2 = Qwen2Model(config)
        if config.tie_word_embeddings:
            self.lm_head = Qwen2LMHead(
                config,
                embedding_weights=self.qwen2.embed_tokens.weight,
                transpose_y=True,
            )
            self.tie_weights()
        else:
            self.lm_head = Qwen2LMHead(config)
        self.criterion = Qwen2PretrainingCriterion(config)
        self.vocab_size = config.vocab_size

    def get_input_embeddings(self):
        return self.qwen2.embed_tokens

    def set_input_embeddings(self, value):
        self.qwen2.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.qwen2 = decoder

    def get_decoder(self):
        return self.qwen2

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=False,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape
        position_ids = kwargs.get(
            "position_ids", paddle.arange(seq_length).expand((batch_size, seq_length))
        )
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(
                shape=[None, None], dtype="int64"
            ),
            "position_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    @staticmethod
    def update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=False
    ):
        # update cache
        if (
            isinstance(outputs, tuple)
            and len(outputs) > 1
            and not isinstance(outputs[1], paddle.Tensor)
        ):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithPast) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat(
                [position_ids, position_ids[..., -1:] + 1], axis=-1
            )

        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            # TODO: support attention mask for other models
            attention_mask = model_kwargs["attention_mask"]
            if len(attention_mask.shape) == 2:
                model_kwargs["attention_mask"] = paddle.concat(
                    [
                        attention_mask,
                        paddle.ones(
                            [attention_mask.shape[0], 1], dtype=attention_mask.dtype
                        ),
                    ],
                    axis=-1,
                )
            elif len(attention_mask.shape) == 4:
                model_kwargs["attention_mask"] = paddle.concat(
                    [
                        attention_mask,
                        paddle.ones(
                            [*attention_mask.shape[:3], 1], dtype=attention_mask.dtype
                        ),
                    ],
                    axis=-1,
                )[:, :, -1:, :]

        return model_kwargs

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logging.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.qwen2(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = outputs[0]

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is together with ParallelCrossEntropy
        tensor_parallel_output = (
            self.config.tensor_parallel_output
            and self.config.tensor_parallel_degree > 1
        )

        logits = self.lm_head(
            hidden_states, tensor_parallel_output=tensor_parallel_output
        )
        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
