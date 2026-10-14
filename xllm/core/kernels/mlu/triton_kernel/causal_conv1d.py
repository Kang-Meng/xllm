# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MLU causal convolution kernels for prefill and decode."""

import triton
import triton.language as tl

# Prefill is adapted from torch_mlu_ops 1.12.2+torch2.10.0,
# torch_mlu_ops/triton/conv/kernels.py:tmo_causal_conv1d_fwd_vllm_kernel.
# xLLM adds independent source slots without changing the convolution order.
# Distinct destinations must be unique and must not alias another row's source.
# The upstream prefill portions retain the following BSD 3-Clause license:
# BSD 3-Clause License
#
# Copyright (c) 2023-2026, Cambricon Technologies
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


@triton.jit(
    do_not_specialize=["num_cache_lines", "NT", "cu_seqlen"],
    do_not_specialize_on_alignment=["num_cache_lines", "NT", "cu_seqlen"],
)
def causal_conv1d_prefill_kernel(  # continuous batching
    x_ptr: tl.tensor,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr: tl.tensor,  # (dim, width)
    bias_ptr: tl.tensor,
    initial_states_ptr: tl.tensor,  # conv_states_ptr
    cache_indices_ptr: tl.tensor,  # Final slots or checkpoint table.
    read_cache_indices_ptr: tl.tensor,  # Optional initial slot per request.
    has_initial_states_ptr: tl.tensor,
    query_start_loc_ptr: tl.tensor,
    batch_ptr: tl.tensor,
    token_chunk_offset_ptr: tl.tensor,
    block_idx_first_scheduled_token: tl.tensor,  # (batch,)
    block_idx_last_scheduled_token: tl.tensor,  # (batch,)
    initial_state_idx: tl.tensor,  # (batch,)
    num_computed_tokens: tl.tensor,  # (batch,)
    o_ptr: tl.tensor,  # (dim, cu_seqlen)
    dim: tl.constexpr,
    NT: tl.int32,
    cu_seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.int32,  # added to support vLLM larger cache lines
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_read_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_block_m: tl.constexpr,  # Stride block to align divided by BLOCK_M
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    USE_READ_CACHE_INDICES: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    tl.static_assert(
        not (USE_READ_CACHE_INDICES and IS_APC_ENABLED),
        "Independent prefill sources do not support checkpoint tables",
    )
    tl.static_assert(
        (KERNEL_WIDTH >= 2) and (KERNEL_WIDTH <= 5),
        "Prefill supports convolution widths 2 through 5",
    )
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1  # can be passed via argument if it's not the same as this value

    id = tl.program_id(0)
    task_dim = tl.num_programs(0)
    ND = tl.cdiv(dim, BLOCK_N)
    chunk_num = NT * ND

    rem = chunk_num % task_dim
    block = chunk_num // task_dim + (id < rem)
    block_start = id * block + tl.where(id < rem, 0, rem)
    for offset in range(block_start, block_start + block):
        i_t = offset // ND
        i_d = offset % ND
        idx_seq = tl.load(batch_ptr + i_t).to(tl.int64)
        chunk_offset = tl.load(token_chunk_offset_ptr + i_t)

        idx_feats = i_d * BLOCK_N + tl.arange(0, BLOCK_N)
        if idx_seq != pad_slot_id:
            sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
            sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
            seqlen = sequence_end_index - sequence_start_index
            B_size: tl.constexpr = stride_block_m * BLOCK_M

            if IS_APC_ENABLED:
                current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
                current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
                sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

                sequence_completed_offset_token = sequence_completed_index % B_size
                seq_completed_offset = B_size - sequence_completed_offset_token
                seq_end_offset = (seqlen - seq_completed_offset) % B_size
                last_full_block_token_index = sequence_end_index - seq_end_offset
                if seq_end_offset == 0:
                    last_full_block_token_index = last_full_block_token_index - B_size

                n_block_to_fill = current_last_index - current_first_index

                conv_state_init_index = tl.load(initial_state_idx + idx_seq)
            else:
                n_block_to_fill = 0
                current_last_index = 0
                conv_state_init_index = 0
                current_first_index = 0
                last_full_block_token_index = 0

            token_offset = BLOCK_M * chunk_offset
            segment_len = min(BLOCK_M, seqlen - token_offset)

            x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim  # [BLOCK_N,]

            conv_states_output_coord = tl.load(
                conv_state_indices_ptr + idx_seq * stride_cache_indices + current_last_index
            ).to(tl.int64)
            if USE_READ_CACHE_INDICES:
                # Later chunks read history from x; only chunk zero needs the
                # independently selected initial state.
                conv_states_input_coord = conv_states_output_coord
                if chunk_offset == 0:
                    conv_states_input_coord = tl.load(read_cache_indices_ptr + idx_seq * stride_read_cache_indices).to(
                        tl.int64
                    )
                process_coord = conv_states_output_coord
            else:
                conv_states_input_coord = tl.load(
                    conv_state_indices_ptr + idx_seq * stride_cache_indices + conv_state_init_index
                ).to(tl.int64)
                process_coord = conv_states_input_coord

            if not HAS_NULL_BLOCK or process_coord != null_block_id:
                conv_states_base = (
                    conv_states_ptr
                    + (conv_states_input_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)
                )  # [BLOCK_N,]

                w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

                if chunk_offset == 0:
                    load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
                    if USE_READ_CACHE_INDICES:
                        load_init_state &= conv_states_input_coord != null_block_id
                    # Preserve the original checkpoint writeback for ordinary
                    # prefill; distinct I/O always publishes to the destination.
                    final_state_base = conv_states_base
                    if USE_READ_CACHE_INDICES:
                        final_state_base = (
                            conv_states_ptr
                            + conv_states_output_coord * stride_conv_state_seq
                            + idx_feats * stride_conv_state_dim
                        )
                    if load_init_state:
                        prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
                        mask_w = idx_feats < dim
                        if KERNEL_WIDTH == 2:
                            conv_states_ptrs = prior_tokens  # [BLOCK_N]
                            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
                        if KERNEL_WIDTH == 3:
                            conv_states_ptrs = prior_tokens  # [BLOCK_N]
                            col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
                        if KERNEL_WIDTH == 4:
                            conv_states_ptrs = prior_tokens  # [BLOCK_N]
                            col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                            col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
                        if KERNEL_WIDTH == 5:
                            conv_states_ptrs = prior_tokens  # [BLOCK_N]
                            col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                            col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                            col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
                    else:
                        if KERNEL_WIDTH >= 2:  # STRATEGY1
                            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                        if KERNEL_WIDTH >= 3:  # STRATEGY1
                            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                        if KERNEL_WIDTH >= 4:  # STRATEGY1
                            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                        if KERNEL_WIDTH >= 5:  # STRATEGY1
                            col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

                    if state_len <= seqlen:  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
                        idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
                        x_ptrs = (
                            x_ptr
                            + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                            + (idx_feats * stride_x_dim)[None, :]
                        )  # [BLOCK_M,BLOCK_N,]
                        mask_x = (
                            (idx_tokens_last >= 0)[:, None]
                            & (idx_tokens_last < seqlen)[:, None]
                            & (idx_feats < dim)[None, :]
                        )  # token-index  # token-index  # feature-index
                        loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                        idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                        conv_states_ptrs_target = (
                            conv_states_ptr
                            + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                            + (idx_feats * stride_conv_state_dim)
                        )[None, :] + (  # [BLOCK_N,]
                            idx_tokens_conv * stride_conv_state_tok
                        )[:, None]

                        mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                        tl.debug_barrier()  # Preserve the upstream compiler workaround.
                        tl.store(conv_states_ptrs_target, loaded_x, mask)

                    else:
                        if load_init_state:
                            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                            conv_states_ptrs_source = (
                                conv_states_ptr
                                + (conv_states_input_coord * stride_conv_state_seq)
                                + (idx_feats * stride_conv_state_dim)[None, :]
                                + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                            )  # [BLOCK_M, BLOCK_N]
                            mask = (
                                (conv_states_input_coord < num_cache_lines)
                                & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                                & (idx_feats < dim)[None, :]
                            )
                            conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                            VAL = state_len - seqlen

                            x_ptrs = (
                                x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                            )  # [BLOCK_M, BLOCK_N]

                            mask_x = (
                                (idx_tokens_conv - VAL >= 0)[:, None]
                                & (idx_tokens_conv - VAL < seqlen)[:, None]
                                & (idx_feats < dim)[None, :]
                            )  # token-index  # token-index  # feature-index
                            loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                            tl.debug_barrier()  # Complete source loads before merging and storing.
                            new_conv_state = tl.where(
                                mask, conv_state, loaded_x
                            )  # BUG in 'tl.where'  which requires a barrier before this
                            conv_states_ptrs_target = (
                                final_state_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                            )  # [BLOCK_M, BLOCK_N]
                            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                            tl.store(conv_states_ptrs_target, new_conv_state, mask)
                        else:  # load_init_state == False
                            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                            VAL = state_len - seqlen

                            x_ptrs = (
                                x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                            )  # [BLOCK_M, BLOCK_N]

                            mask_x = (
                                (idx_tokens_conv - VAL >= 0)[:, None]
                                & (idx_tokens_conv - VAL < seqlen)[:, None]
                                & (idx_feats < dim)[None, :]
                            )  # token-index  # token-index  # feature-index
                            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                            conv_states_ptrs_target = (
                                final_state_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                            )  # [BLOCK_M, BLOCK_N]
                            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                            tl.store(conv_states_ptrs_target, new_conv_state, mask)

                else:  # chunk_offset > 0
                    load_init_state = True
                    prior_tokens = x_base + (token_offset - 1) * stride_x_token
                    mask_w = idx_feats < dim
                    if KERNEL_WIDTH == 2:
                        conv_states_ptrs = prior_tokens  # [BLOCK_N]
                        col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                    if KERNEL_WIDTH == 3:
                        conv_states_ptrs = prior_tokens  # [BLOCK_N]
                        col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
                        col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                    if KERNEL_WIDTH == 4:
                        conv_states_ptrs = prior_tokens  # [BLOCK_N]
                        col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
                        col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
                        col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                    if KERNEL_WIDTH == 5:
                        conv_states_ptrs = prior_tokens  # [BLOCK_N]
                        col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
                        col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
                        col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
                        col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

                    if (chunk_offset - 1) < n_block_to_fill:
                        idx_tokens_last = (
                            last_full_block_token_index - (n_block_to_fill - chunk_offset) * B_size - state_len
                        ) + tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
                        x_ptrs = (
                            x_ptr + (idx_tokens_last * stride_x_token)[:, None] + (idx_feats * stride_x_dim)[None, :]
                        )  # [BLOCK_M,BLOCK_N,]

                        mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[
                            None, :
                        ]  # token-index  # token-index  # feature-index
                        loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                        idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                        conv_states_output_coord = tl.load(
                            conv_state_indices_ptr
                            + idx_seq * stride_cache_indices
                            + current_first_index
                            + (chunk_offset - 1)
                        ).to(tl.int64)

                        conv_states_ptrs_target = (
                            conv_states_ptr
                            + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                            + (idx_feats * stride_conv_state_dim)
                        )[None, :] + (  # [BLOCK_N,]
                            idx_tokens_conv * stride_conv_state_tok
                        )[:, None]

                        mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                        tl.debug_barrier()  # Preserve the upstream compiler workaround.
                        tl.store(conv_states_ptrs_target, loaded_x, mask)

                if HAS_BIAS:
                    bias = bias_ptr + idx_feats
                    mask_bias = idx_feats < dim
                    acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)  # [BLOCK_N]
                else:
                    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

                x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

                mask_w = idx_feats < dim
                if KERNEL_WIDTH >= 2:
                    w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
                    w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
                    w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
                    w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
                if KERNEL_WIDTH >= 3:
                    w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
                    w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
                if KERNEL_WIDTH >= 4:
                    w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
                    w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
                if KERNEL_WIDTH == 5:
                    w_ptrs = w_base + (4 * stride_w_width)
                    w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
                mask_x_1d = idx_feats < dim
                for idx_token in range(segment_len):
                    acc = acc_preload

                    matrix_w = w_col0
                    matrix_x = col0
                    for j in tl.static_range(KERNEL_WIDTH):
                        if KERNEL_WIDTH == 2:
                            if j == 1:  # KERNEL_WIDTH-1:
                                matrix_w = w_col1
                                x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                                matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                        elif KERNEL_WIDTH == 3:
                            if j == 1:
                                matrix_w = w_col1
                                matrix_x = col1
                            elif j == 2:
                                matrix_w = w_col2
                                x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                                matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                        elif KERNEL_WIDTH == 4:
                            if j == 1:
                                matrix_w = w_col1
                                matrix_x = col1
                            elif j == 2:
                                matrix_w = w_col2
                                matrix_x = col2
                            elif j == 3:
                                matrix_w = w_col3
                                x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                                matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

                        elif KERNEL_WIDTH == 5:
                            if j == 1:
                                matrix_w = w_col1
                                matrix_x = col1
                            elif j == 2:
                                matrix_w = w_col2
                                matrix_x = col2
                            elif j == 3:
                                matrix_w = w_col3
                                matrix_x = col3
                            elif j == 4:
                                matrix_w = w_col4
                                x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                                matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

                        acc += matrix_x * matrix_w  # [BLOCK_N]

                    if KERNEL_WIDTH == 2:
                        col0 = matrix_x
                    elif KERNEL_WIDTH == 3:
                        col0 = col1
                        col1 = matrix_x
                    elif KERNEL_WIDTH == 4:
                        col0 = col1
                        col1 = col2
                        col2 = matrix_x
                    elif KERNEL_WIDTH == 5:
                        col0 = col1
                        col1 = col2
                        col2 = col3
                        col3 = matrix_x

                    if SILU_ACTIVATION:
                        acc = acc / (1 + tl.exp(-acc))
                    mask_1d = (idx_token < segment_len) & (idx_feats < dim)  # token-index  # feature-index
                    o_ptrs = (
                        o_ptr
                        + (sequence_start_index + token_offset + idx_token) * stride_o_token
                        + (idx_feats * stride_o_dim)
                    )

                    tl.store(o_ptrs, acc, mask=mask_1d)


# This kernel is adapted from vLLM's FLA Triton ops:
# https://github.com/vllm-project/vllm/tree/v0.18.0/vllm/model_executor/layers/fla/ops
# Upstream license: Apache License, Version 2.0.
# Modified for xLLM MLU TMO integration.
@triton.jit()
def tmo_causal_conv1d_update_decode_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.int64,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.int64,
    # others
    null_block_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_B: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    num_n = (dim + BLOCK_N - 1) // BLOCK_N

    NEED_MASK_N = dim % BLOCK_N != 0
    total_blocks = batch * ((dim + BLOCK_N - 1) // BLOCK_N)

    arange_batch = tl.arange(0, BLOCK_B)
    mask_batch = arange_batch < batch
    # Pre-load weights outside the persistent loop
    if IS_APC_ENABLED:
        conv_state_inits = tl.load(initial_state_idx + arange_batch, mask=mask_batch)
        current_last_indexs = tl.load(block_idx_last_scheduled_token + arange_batch, mask=mask_batch)
    else:
        conv_state_inits = 0
        current_last_indexs = 0

    # conv_state_init = 0
    # current_last_index = 0
    if IS_APC_ENABLED:
        pass
    else:
        conv_states_input_coords = tl.load(
            conv_state_indices_ptr + arange_batch * stride_state_indices,
            mask=mask_batch,
            cache_modifier=".cg",
        ).to(tl.int64)
    w_base = (
        w_ptr
        + (tl.arange(0, ((dim + BLOCK_N - 1) // BLOCK_N) * BLOCK_N) * stride_w_dim)[:, None]
        + tl.arange(0, KERNEL_WIDTH)[None, :]
    )  # [BLOCK_N, KERNEL_WIDTH]
    mask_wraw = (tl.arange(0, ((dim + BLOCK_N - 1) // BLOCK_N) * BLOCK_N) < dim)[:, None]
    w_raw = tl.load(w_base, mask_wraw, other=0.0, cache_modifier=".cg")
    ws = tl.trans(w_raw)  # [KERNEL_WIDTH, BLOCK_N*num_n]
    # mask_w = (tl.arange(0, BLOCK_N) < dim)[:, None]
    for flat_pid in range(pid, total_blocks, num_jobs):
        idx_seq = flat_pid // num_n
        pid_n = flat_pid % num_n

        # [BLOCK_N,] elements along the feature-dimension (channel)
        idx_feats = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_bias = idx_feats < dim

        # cache_idx
        if IS_APC_ENABLED:
            conv_state_init = conv_state_inits[idx_seq]
            current_last_index = current_last_indexs[idx_seq]
            conv_states_input_coord = tl.load(
                conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
            ).to(tl.int64)
            conv_states_offset = tl.load(
                conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
            ).to(tl.int64)
        else:
            conv_state_init = 0
            current_last_index = 0
            conv_states_input_coord = conv_states_input_coords[idx_seq]
            conv_states_offset = conv_states_input_coords[idx_seq]

        should_process = True
        if HAS_NULL_BLOCK:
            if conv_states_input_coord == null_block_id:
                should_process = False

        if should_process:
            if IS_VARLEN:
                query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
                query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
                actual_seqlen = query_end_index - query_start_index
                actual_state_len = state_len - (seqlen - actual_seqlen)
                x_offset = query_start_index * stride_x_token
                o_offset = query_start_index * stride_o_token
            else:
                query_start_index = idx_seq * seqlen
                query_end_index = query_start_index + seqlen
                actual_seqlen = seqlen
                actual_state_len = state_len
                x_offset = idx_seq * stride_x_seq
                o_offset = idx_seq * stride_o_seq

            if query_start_index != query_end_index:
                if IS_SPEC_DECODING:
                    conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
                else:
                    conv_state_token_offset = 0

                # STEP 1: READ init_state data
                col_offsets = tl.arange(0, KERNEL_WIDTH - 1) * stride_conv_state_tok  # [KERNEL_WIDTH-1]
                conv_states_base = (
                    conv_state_ptr
                    + (conv_states_input_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[:, None]
                    + col_offsets[None, :]
                )

                prior_tokens = conv_states_base + (conv_state_token_offset * stride_conv_state_tok)[None, :]
                oldState_raw = tl.load(prior_tokens, mask_bias[:, None], 0.0, cache_modifier=".cg")
                oldState = tl.trans(oldState_raw)

                # STEP 2: assume state_len > seqlen
                x_ptrs = (
                    x_ptr
                    + x_offset
                    + (idx_feats * stride_x_dim)[None, :]
                    + (tl.arange(0, seqlen) * stride_x_token)[:, None]
                )  # [seqlen, BLOCK_N]
                xs = tl.load(
                    x_ptrs,
                    (tl.arange(0, seqlen) < actual_seqlen)[:, None] & mask_bias[None, :],
                    0.0,
                    cache_modifier=".cg",
                )

                tl.debug_barrier()

                FULL_LEN: tl.constexpr = KERNEL_WIDTH - 1 + seqlen
                new_conv_state = tl.empty((FULL_LEN, BLOCK_N), dtype=conv_states_base.dtype.element_ty)
                new_conv_state[0 : KERNEL_WIDTH - 1, :] = oldState
                new_conv_state[KERNEL_WIDTH - 1 : FULL_LEN, :] = xs

                # STEP 3: init accumulator
                if HAS_BIAS:
                    bias = bias_ptr + idx_feats
                    acc_preload = tl.load(bias, mask=mask_bias, other=0.0, cache_modifier=".cg").to(
                        tl.float32
                    )  # [BLOCK_N]
                else:
                    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

                # STEP 5: compute each token
                for idx_token in range(actual_seqlen):
                    mask_1d = (idx_token < actual_seqlen) & (idx_feats < dim)

                    acc = acc_preload

                    x_cur = xs[idx_token, :]

                    for idx_width in tl.static_range(KERNEL_WIDTH):
                        if idx_width == KERNEL_WIDTH - 1:
                            matrix_x = x_cur
                        else:
                            matrix_x = oldState[idx_width, :]
                        acc += ws[idx_width, pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N] * matrix_x

                    if KERNEL_WIDTH > 2:
                        oldState[0 : KERNEL_WIDTH - 2, :] = oldState[1 : KERNEL_WIDTH - 1, :]
                    oldState[KERNEL_WIDTH - 2, :] = x_cur
                    if SILU_ACTIVATION:
                        acc = acc / (1 + tl.exp(-acc))
                    o_ptrs = o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)

                    tl.store(o_ptrs, acc, mask=mask_1d)

                if IS_SPEC_DECODING:
                    oldstart = 1
                else:
                    oldstart = actual_seqlen
                cover_lo = actual_state_len - actual_seqlen  # = S - L, runtime but unrelated to a
                new_state_rows = tl.empty((state_len, BLOCK_N), dtype=oldState.dtype)
                FULL_LEN_I: tl.constexpr = FULL_LEN  # total rows of new_conv_state, used for clamping
                for gi in tl.static_range(state_len):
                    in_cover = (gi >= cover_lo) & (gi < state_len)
                    src_row = tl.where(
                        in_cover,
                        (gi - cover_lo) + (KERNEL_WIDTH - 1),
                        oldstart + gi,
                    )
                    # When cover_lo < 0 (varlen short sequence a<L, S-L<0), non-steady-state branch
                    # src_row may go out of [0, FULL_LEN); the baseline relies on the read side of tl.where(mask,...)
                    src_row = tl.minimum(tl.maximum(src_row, 0), FULL_LEN_I - 1)
                    new_state_rows[gi, :] = new_conv_state[src_row, :]
                new_state_data = tl.trans(new_state_rows)  # [BLOCK_N, state_len]

                idx_tokens_store = tl.arange(0, state_len)
                conv_state_ptrs_target = (
                    conv_state_ptr + (conv_states_offset * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim)
                )[:, None] + (idx_tokens_store * stride_conv_state_tok)[None, :]
                mask_store = (idx_tokens_store < actual_state_len)[None, :] & mask_bias[:, None]
                if NEED_MASK_N:
                    tl.store(conv_state_ptrs_target, new_state_data, mask_store)
                else:
                    tl.store(conv_state_ptrs_target, new_state_data, mask=idx_tokens_store[None, :] < actual_state_len)
